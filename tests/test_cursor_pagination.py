"""预约游标分页：页间不重叠、插入期间稳定快照、游标损坏返回可识别错误。"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import timedelta

from service_09252_008.application.booking_service import BookingService
from service_09252_008.application.catalog_service import CatalogService
from service_09252_008.application.cursor import InvalidCursorError
from service_09252_008.application.ports import ManualClock, SequentialIdGenerator, SystemClock, UuidIdGenerator
from service_09252_008.domain.errors import ValidationError
from service_09252_008.interfaces.http_api import create_server
from service_09252_008.persistence.sqlite_store import SQLiteStore
from service_09252_008.persistence.store import InMemoryStore
from tests.helpers import NOW, apply_payload, seed_catalog

SECRET = b"test-secret-key"


def _apply_n(bookings: BookingService, ids: dict, count: int, key_prefix: str) -> list[str]:
    created = []
    for i in range(count):
        view = bookings.apply(apply_payload(ids, f"{key_prefix}-{i}"))
        created.append(view["booking_id"])
    return created


def _service(store):
    clock = ManualClock(NOW)
    id_gen = SequentialIdGenerator()
    catalog = CatalogService(store, clock, id_gen)
    bookings = BookingService(store, clock, id_gen, cursor_secret=SECRET)
    return catalog, bookings, clock


def _walk_pages(bookings: BookingService, **kwargs) -> tuple[list[str], list[list[str]]]:
    """从头走到尾，返回（全部 booking_id，每页 booking_id）。"""
    seen: list[str] = []
    pages: list[list[str]] = []
    cursor: str | None = None
    while True:
        page = bookings.list_bookings_page(cursor=cursor, **kwargs)
        ids = [item["booking_id"] for item in page["items"]]
        pages.append(ids)
        seen.extend(ids)
        if not page["has_more"]:
            assert page["next_cursor"] is None
            break
        cursor = page["next_cursor"]
        assert cursor
    return seen, pages


class CursorPaginationTests(unittest.TestCase):
    def test_pages_cover_every_booking_without_overlap_in_memory(self) -> None:
        store = InMemoryStore()
        catalog, bookings, _ = _service(store)
        ids = seed_catalog(catalog, window_capacity=100)
        expected = set(_apply_n(bookings, ids, 25, "k-page"))
        seen, pages = _walk_pages(bookings, limit=7)
        self.assertEqual(set(seen), expected)
        self.assertEqual(len(seen), len(expected))  # 无重复
        self.assertTrue(all(len(p) <= 7 for p in pages))
        self.assertEqual([len(p) for p in pages[:-1]], [7] * (25 // 7))
        for earlier, later in zip(pages, pages[1:]):
            self.assertEqual(set(earlier) & set(later), set())

    def test_pages_cover_every_booking_without_overlap_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(f"{tmp}/page.db")
            try:
                catalog, bookings, _ = _service(store)
                ids = seed_catalog(catalog, window_capacity=100)
                expected = set(_apply_n(bookings, ids, 25, "k-page-sqlite"))
                seen, pages = _walk_pages(bookings, limit=7)
                self.assertEqual(set(seen), expected)
                self.assertEqual(len(seen), len(expected))
                for earlier, later in zip(pages, pages[1:]):
                    self.assertEqual(set(earlier) & set(later), set())
            finally:
                store.close()

    def test_ties_on_created_at_are_broken_by_booking_id(self) -> None:
        store = InMemoryStore()
        catalog, bookings, _ = _service(store)
        ids = seed_catalog(catalog, window_capacity=100)
        # 不推进时钟：所有预约 created_at 相同，只能靠 booking_id 决胜
        same_time = _apply_n(bookings, ids, 10, "k-tie")
        seen, _ = _walk_pages(bookings, limit=4)
        self.assertEqual(seen, sorted(same_time))

    def test_inserting_between_pages_never_overlaps_in_memory(self) -> None:
        """首页快照形成后穿插插入：既有预约不重不漏，锚点前的新行不补印。"""
        store = InMemoryStore()
        catalog, bookings, clock = _service(store)
        ids = seed_catalog(catalog, window_capacity=100)
        original: list[str] = []
        for i in range(12):
            original += _apply_n(bookings, ids, 1, f"k-mid-{i}")
            clock.advance(seconds=1)

        first = bookings.list_bookings_page(limit=5)
        first_ids = [item["booking_id"] for item in first["items"]]
        self.assertEqual(len(first_ids), 5)

        # 翻页期间插入两条：一条 created_at 早于首页锚点，一条晚于全部既有行
        clock.set(NOW - timedelta(minutes=1))
        before_id = _apply_n(bookings, ids, 1, "k-before")[0]
        clock.set(NOW + timedelta(minutes=10))
        after_id = _apply_n(bookings, ids, 1, "k-after")[0]

        cursor = first["next_cursor"]
        rest_ids = list(first_ids)
        while True:
            page = bookings.list_bookings_page(cursor=cursor, limit=5)
            page_ids = [item["booking_id"] for item in page["items"]]
            # 关键不变量：后续任何页都不与首页重叠，本页自身也不重复
            self.assertEqual(set(page_ids) & set(first_ids), set())
            self.assertEqual(len(page_ids), len(set(page_ids)))
            rest_ids.extend(page_ids)
            if not page["has_more"]:
                break
            cursor = page["next_cursor"]

        # 12 条既有预约全部且仅出现一次
        original_set = set(original)
        originals_seen = [b for b in rest_ids if b in original_set]
        self.assertEqual(len(originals_seen), 12)
        self.assertEqual(len(set(originals_seen)), 12)
        # 晚到的新行在尾部出现一次；早于锚点的新行不会在后续页“补印”
        self.assertEqual(rest_ids.count(after_id), 1)
        self.assertEqual(rest_ids[-1], after_id)
        self.assertNotIn(before_id, rest_ids)
        # 从头重翻（新快照）可以看到早插入的行
        fresh, _ = _walk_pages(bookings, limit=5)
        self.assertIn(before_id, fresh)

    def test_sqlite_concurrent_inserts_do_not_overlap_pages(self) -> None:
        """写线程持续插入时，键集分页每页运行在同一快照，页间无重复。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(f"{tmp}/conc.db")
            try:
                clock = ManualClock(NOW)
                id_gen = SequentialIdGenerator()
                catalog = CatalogService(store, clock, id_gen)
                bookings = BookingService(store, clock, id_gen, cursor_secret=SECRET)
                ids = seed_catalog(catalog, window_capacity=1000)
                _apply_n(bookings, ids, 20, "k-seed")

                stop = threading.Event()

                def writer() -> None:
                    # 写者使用独立时钟/ID 生成器与同一存储，模拟另一请求来源
                    w_bookings = BookingService(store, SystemClock(), UuidIdGenerator(), cursor_secret=SECRET)
                    for i in range(200):
                        if stop.is_set():
                            break
                        w_bookings.apply(apply_payload(ids, f"k-writer-{i}"))

                thread = threading.Thread(target=writer)
                thread.start()
                seen: list[str] = []
                try:
                    cursor: str | None = None
                    while True:
                        page = bookings.list_bookings_page(cursor=cursor, limit=6)
                        seen.extend(item["booking_id"] for item in page["items"])
                        if not page["has_more"]:
                            break
                        cursor = page["next_cursor"]
                finally:
                    stop.set()
                    thread.join(timeout=10)
                self.assertEqual(len(seen), len(set(seen)), "页间出现重叠")
            finally:
                store.close()

    def test_status_filter_is_applied_and_bound_to_cursor(self) -> None:
        store = InMemoryStore()
        catalog, bookings, _ = _service(store)
        ids = seed_catalog(catalog, window_capacity=2)  # 容量 2：超出部分进入候补
        _apply_n(bookings, ids, 9, "k-filter")
        waitlisted, pages = _walk_pages(bookings, limit=4, status="WAITLISTED")
        self.assertTrue(waitlisted)
        self.assertEqual(len(waitlisted), len(set(waitlisted)))
        for page_ids in pages:
            for booking_id in page_ids:
                self.assertEqual(bookings.get_booking(booking_id)["status"], "WAITLISTED")
        # 无过滤游标不能配合 status 使用（scope 不一致）
        unfiltered = bookings.list_bookings_page(limit=2)
        with self.assertRaises(InvalidCursorError) as ctx:
            bookings.list_bookings_page(cursor=unfiltered["next_cursor"], status="WAITLISTED")
        self.assertEqual(ctx.exception.details["reason"], "scope_mismatch")
        # 未知状态值是普通的输入校验错误
        with self.assertRaises(ValidationError):
            bookings.list_bookings_page(status="NOPE")

    def test_damaged_cursors_are_rejected_with_identifiable_error(self) -> None:
        store = InMemoryStore()
        catalog, bookings, _ = _service(store)
        ids = seed_catalog(catalog, window_capacity=100)
        _apply_n(bookings, ids, 3, "k-bad")
        good = bookings.list_bookings_page(limit=1)["next_cursor"]
        assert good

        cases = (
            ("not-a-cursor", "malformed"),
            ("bkgcur.%%%.sig", "bad_signature"),
            (good[:-2] + ("aa" if good[-2:] != "aa" else "bb"), "bad_signature"),
            (good.replace(".", "_", 1), "malformed"),
        )
        for token, reason in cases:
            with self.subTest(token=token):
                with self.assertRaises(InvalidCursorError) as ctx:
                    bookings.list_bookings_page(cursor=token)
                self.assertEqual(ctx.exception.code, "invalid_cursor")
                self.assertEqual(ctx.exception.details["reason"], reason)

        # 为另一密钥签发的游标 -> 签名失败
        other = BookingService(store, ManualClock(NOW), SequentialIdGenerator(), cursor_secret=b"other-secret")
        with self.assertRaises(InvalidCursorError):
            other.list_bookings_page(cursor=good)

    def test_limit_validation(self) -> None:
        store = InMemoryStore()
        catalog, bookings, _ = _service(store)
        ids = seed_catalog(catalog)
        _apply_n(bookings, ids, 1, "k-limit")
        for bad in (0, -1, 101):
            with self.assertRaises(ValidationError):
                bookings.list_bookings_page(limit=bad)
        page = bookings.list_bookings_page(limit=1)
        self.assertEqual(len(page["items"]), 1)


class CursorPaginationHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        store = InMemoryStore()
        clock = ManualClock(NOW)
        id_gen = SequentialIdGenerator()
        cls.catalog = CatalogService(store, clock, id_gen)
        cls.bookings = BookingService(store, clock, id_gen, cursor_secret=b"http-secret")
        cls.ids = seed_catalog(cls.catalog, window_capacity=100)
        _apply_n(cls.bookings, cls.ids, 8, "k-http-page")
        cls.server = create_server("127.0.0.1", 0, cls.catalog, cls.bookings)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _get(self, path: str) -> tuple[int, dict]:
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_walk_bookings_over_http(self) -> None:
        seen: list[str] = []
        path = "/bookings?limit=3"
        while True:
            status, body = self._get(path)
            self.assertEqual(status, 200)
            self.assertLessEqual(len(body["items"]), 3)
            seen.extend(item["booking_id"] for item in body["items"])
            if not body["has_more"]:
                self.assertIsNone(body["next_cursor"])
                break
            path = f"/bookings?limit=3&cursor={body['next_cursor']}"
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(seen), 8)

    def test_damaged_cursor_is_identifiable_client_error(self) -> None:
        status, body = self._get("/bookings?cursor=bkgcur.tampered.payload")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_cursor")
        self.assertIn("reason", body["details"])

    def test_bad_limit_is_client_error(self) -> None:
        status, body = self._get("/bookings?limit=abc")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "validation_error")
        status, body = self._get("/bookings?limit=999")
        self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
