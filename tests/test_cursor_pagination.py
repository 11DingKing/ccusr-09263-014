"""预约游标分页：排序锚点、跨页不重叠、并发插入与游标损坏。

覆盖：
- 键集分页顺序稳定（created_at + booking_id），页间无重叠/无遗漏；
- 翻页之间插入新预约（内存与 SQLite 双后端）不造成前后页重叠或回跳；
- SQLite 读事务使用同一快照：并发表务提交对进行中的遍历不可见，
  且生产路径的 BEGIN IMMEDIATE 会直接阻塞并发写者；
- 游标被篡改、版本不符或由其他密钥签名时返回 invalid_cursor；
- HTTP 边界把损坏游标/非法 limit 映射为 400。
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timezone

from service_09252_008.application.booking_service import COLLECTION_BOOKINGS, BookingService
from service_09252_008.application.catalog_service import CatalogService
from service_09252_008.application.cursor import BookingCursor, decode_cursor, encode_cursor
from service_09252_008.application.ports import ManualClock, SequentialIdGenerator
from service_09252_008.domain.errors import CursorError, ValidationError
from service_09252_008.interfaces.http_api import create_server
from service_09252_008.persistence.sqlite_store import SQLiteStore
from service_09252_008.persistence.store import InMemoryStore
from tests.helpers import NOW, apply_payload, seed_catalog

CURSOR_SECRET = b"0123456789abcdef0123456789abcdef"


def _apply_n(bookings: BookingService, ids: dict, start: int, count: int) -> list[str]:
    return [
        bookings.apply(apply_payload(ids, f"k-page-apply-{i}"))["booking_id"]
        for i in range(start, start + count)
    ]


class CursorPaginationTests(unittest.TestCase):
    def setUp(self) -> None:
        # 显式装配：catalog 与 bookings 共享序列 ID 生成器，
        # 后续用不同时钟再建服务实例时也不会出现主键碰撞。
        self.store = InMemoryStore()
        self.clock = ManualClock(NOW)
        self.id_gen = SequentialIdGenerator()
        self.catalog = CatalogService(self.store, self.clock, self.id_gen)
        self.bookings = BookingService(
            self.store, self.clock, self.id_gen, cursor_secret=CURSOR_SECRET
        )
        self.ids = seed_catalog(self.catalog, window_capacity=100)

    def test_pages_are_stable_ordered_and_disjoint(self) -> None:
        all_ids = _apply_n(self.bookings, self.ids, 0, 7)
        pages: list[list[str]] = []
        cursor: str | None = None
        for _ in range(10):
            page = self.bookings.list_bookings_page(limit=3, cursor=cursor)
            pages.append([item["booking_id"] for item in page["items"]])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        self.assertEqual([len(p) for p in pages], [3, 3, 1])
        flattened = [bid for page in pages for bid in page]
        self.assertEqual(flattened, all_ids)  # 同一时刻申请，按 booking_id 决胜
        self.assertEqual(len(set(flattened)), len(flattened))  # 无重复即无重叠
        self.assertEqual(set(flattened), set(all_ids))  # 无遗漏

    def test_inserting_between_pages_never_overlaps(self) -> None:
        original_ids = _apply_n(self.bookings, self.ids, 0, 5)

        first = self.bookings.list_bookings_page(limit=3)
        first_ids = [item["booking_id"] for item in first["items"]]
        self.assertEqual(first_ids, original_ids[:3])

        # 翻页过程中插入两条新预约（时间戳相同，booking_id 必然排在旧锚点之后）
        inserted_ids = _apply_n(self.bookings, self.ids, 100, 2)

        second = self.bookings.list_bookings_page(limit=3, cursor=first["next_cursor"])
        second_ids = [item["booking_id"] for item in second["items"]]
        # 第二页紧接着第一页锚点：原列表第 4、5 条仍紧跟其后，新插入行追加在尾，
        # 标准键集语义——关键是与第一页不重叠。
        self.assertEqual(second_ids, original_ids[3:5] + inserted_ids[:1])
        self.assertTrue(set(first_ids).isdisjoint(second_ids))

        third = self.bookings.list_bookings_page(limit=3, cursor=second["next_cursor"])
        third_ids = [item["booking_id"] for item in third["items"]]
        self.assertEqual(third_ids, inserted_ids[1:])
        self.assertIsNone(third["next_cursor"])
        self.assertTrue(set(first_ids).isdisjoint(third_ids))
        self.assertTrue(set(second_ids).isdisjoint(third_ids))

    def test_row_earlier_than_anchor_cannot_reappear(self) -> None:
        # 翻页途中插入一条“排在游标之前”的预约（更早创建时间）：
        # 它不会把已翻过的条目挤回后续页，遍历依旧无重叠、不回跳。
        _apply_n(self.bookings, self.ids, 0, 4)
        first = self.bookings.list_bookings_page(limit=2)
        first_ids = [item["booking_id"] for item in first["items"]]

        early_clock = ManualClock(datetime(2026, 9, 24, tzinfo=timezone.utc))  # 早于 NOW
        early_service = BookingService(
            self.store, early_clock, self.id_gen, cursor_secret=CURSOR_SECRET
        )
        early_id = early_service.apply(apply_payload(self.ids, "k-early-insert"))["booking_id"]

        seen_ids = set(first_ids)
        cursor = first["next_cursor"]
        while cursor:
            page = self.bookings.list_bookings_page(limit=10, cursor=cursor)
            page_ids = [item["booking_id"] for item in page["items"]]
            self.assertTrue(seen_ids.isdisjoint(page_ids))  # 与已交付页不重叠
            seen_ids.update(page_ids)
            cursor = page["next_cursor"]
        self.assertNotIn(early_id, seen_ids)  # 早于锚点的新行不会在后续页回跳
        self.assertEqual(seen_ids, {f"bkg_{i:04d}" for i in range(1, 5)})

    def test_filters_paginate_independently(self) -> None:
        _apply_n(self.bookings, self.ids, 0, 3)
        page = self.bookings.list_bookings_page(
            limit=2, status="REQUESTED", window_id=self.ids["window_id"]
        )
        self.assertEqual(len(page["items"]), 2)
        self.assertTrue(all(item["status"] == "REQUESTED" for item in page["items"]))
        empty = self.bookings.list_bookings_page(status="CANCELLED")
        self.assertEqual(empty["items"], [])
        self.assertIsNone(empty["next_cursor"])
        self.assertFalse(empty["has_more"])

    def test_invalid_status_and_limit_are_client_errors(self) -> None:
        with self.assertRaises(ValidationError):
            self.bookings.list_bookings_page(status="NOPE")
        for bad_limit in (0, -1, 201):
            with self.subTest(limit=bad_limit):
                with self.assertRaises(ValidationError):
                    self.bookings.list_bookings_page(limit=bad_limit)


class CursorCorruptionTests(unittest.TestCase):
    def test_roundtrip_and_tamper_detection(self) -> None:
        anchor = BookingCursor(NOW, "bkg_0007")
        token = encode_cursor(anchor, CURSOR_SECRET)
        decoded = decode_cursor(token, CURSOR_SECRET)
        assert decoded is not None
        self.assertEqual(decoded.to_anchor(), ("2026-09-25T00:00:00+00:00", "bkg_0007"))
        self.assertIsNone(decode_cursor(None, CURSOR_SECRET))

        _, payload_b64, sig_b64 = token.split(".")
        other_secret_token = encode_cursor(anchor, b"ffffffffffffffffffffffffffffffff")
        flipped = sig_b64[:-1] + ("A" if sig_b64[-1] != "A" else "B")
        cases = [
            "",
            "garbage",
            "v2.aaa.bbb",
            "v1.aaa.bbb",
            token + "x",
            f"v1.{payload_b64}.{flipped}",
            other_secret_token,
        ]
        for bad in cases:
            with self.subTest(cursor=bad[:20]):
                with self.assertRaises(CursorError):
                    decode_cursor(bad, CURSOR_SECRET)

    def test_service_rejects_corrupt_cursor(self) -> None:
        store = InMemoryStore()
        clock = ManualClock(NOW)
        id_gen = SequentialIdGenerator()
        catalog = CatalogService(store, clock, id_gen)
        bookings = BookingService(store, clock, id_gen, cursor_secret=CURSOR_SECRET)
        ids = seed_catalog(catalog)
        _apply_n(bookings, ids, 0, 1)
        for bad in ("garbage", "v1.aa.bb", "", "v9.x.y"):
            with self.subTest(cursor=bad):
                with self.assertRaises(CursorError):
                    bookings.list_bookings_page(cursor=bad)


class SqlitePaginationTests(unittest.TestCase):
    def _build(self, tmp: str) -> tuple[CatalogService, BookingService, SQLiteStore]:
        store = SQLiteStore(f"{tmp}/booking.db")
        clock = ManualClock(NOW)
        ids_gen = SequentialIdGenerator()
        catalog = CatalogService(store, clock, ids_gen)
        bookings = BookingService(store, clock, ids_gen, cursor_secret=CURSOR_SECRET)
        return catalog, bookings, store

    def test_inserting_between_pages_on_sqlite_never_overlaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog, bookings, store = self._build(tmp)
            ids = seed_catalog(catalog, window_capacity=100)
            original = _apply_n(bookings, ids, 0, 5)

            first = bookings.list_bookings_page(limit=2)
            first_ids = [i["booking_id"] for i in first["items"]]
            self.assertEqual(first_ids, original[:2])

            inserted = _apply_n(bookings, ids, 200, 3)

            seen = list(first_ids)
            cursor = first["next_cursor"]
            while cursor:
                page = bookings.list_bookings_page(limit=2, cursor=cursor)
                seen.extend(i["booking_id"] for i in page["items"])
                cursor = page["next_cursor"]

            self.assertEqual(len(seen), len(set(seen)))  # 无重叠
            self.assertEqual(set(seen), set(original) | set(inserted))  # 无遗漏
            self.assertEqual(seen[:2], original[:2])  # 已交付的首页不回跳
            store.close()

    def test_sqlite_scan_uses_one_snapshot_for_inflight_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog, bookings, store = self._build(tmp)
            ids = seed_catalog(catalog, window_capacity=100)
            original = _apply_n(bookings, ids, 0, 4)

            # 在主连接上开启一个读事务，快照就此固定
            conn = store._conn
            conn.execute("BEGIN")
            try:
                page1 = store.scan(COLLECTION_BOOKINGS, limit=2)
                self.assertEqual([r["booking_id"] for r in page1], original[:2])

                # 另一个连接在此期间提交一条新预约
                other = sqlite3.connect(f"{tmp}/booking.db", timeout=5)
                new_row = {
                    "booking_id": "bkg_concurrent_01",
                    "status": "REQUESTED",
                    "created_at": "2026-09-26T00:00:00+00:00",
                    "window_id": ids["window_id"],
                }
                other.execute(
                    "INSERT INTO records (collection, key, data) VALUES (?, ?, ?)",
                    (COLLECTION_BOOKINGS, "bkg_concurrent_01", json.dumps(new_row)),
                )
                other.commit()
                other.close()

                # 同一快照内继续键集翻页：并发提交不可见，不会插入遍历中间
                anchor = (page1[-1]["created_at"], page1[-1]["booking_id"])
                page2 = store.scan(COLLECTION_BOOKINGS, limit=50, cursor=anchor)
                snapshot_ids = {r["booking_id"] for r in page1} | {r["booking_id"] for r in page2}
                self.assertEqual(snapshot_ids, set(original))
                self.assertNotIn("bkg_concurrent_01", snapshot_ids)
            finally:
                conn.rollback()

            # 新快照可以看到并发提交的行
            fresh = store.scan(COLLECTION_BOOKINGS, limit=50)
            self.assertIn("bkg_concurrent_01", [r["booking_id"] for r in fresh])
            store.close()

    def test_immediate_transaction_blocks_concurrent_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog, bookings, store = self._build(tmp)
            ids = seed_catalog(catalog, window_capacity=100)
            _apply_n(bookings, ids, 0, 1)

            # 生产路径持有的是 BEGIN IMMEDIATE 写事务：持锁期间，
            # 其他连接的 IMMEDIATE 事务立即被拒，无法把写入插进同一读取窗口。
            with store.transaction():
                other = sqlite3.connect(f"{tmp}/booking.db", timeout=0)
                try:
                    with self.assertRaises(sqlite3.OperationalError):
                        other.execute("BEGIN IMMEDIATE")
                finally:
                    other.close()
            store.close()

    def test_filter_scan_in_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            catalog, bookings, store = self._build(tmp)
            ids = seed_catalog(catalog, window_capacity=100)
            _apply_n(bookings, ids, 0, 3)
            page = bookings.list_bookings_page(
                limit=1, status="REQUESTED", window_id=ids["window_id"]
            )
            self.assertEqual(len(page["items"]), 1)
            self.assertTrue(page["has_more"])
            store.close()


class HttpPaginationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        store = InMemoryStore()
        clock = ManualClock(NOW)
        id_gen = SequentialIdGenerator()
        catalog = CatalogService(store, clock, id_gen)
        bookings = BookingService(store, clock, id_gen, cursor_secret=CURSOR_SECRET)
        cls.ids = seed_catalog(catalog, window_capacity=100)
        cls.bookings = bookings
        _apply_n(bookings, cls.ids, 0, 5)
        cls.server = create_server("127.0.0.1", 0, catalog, bookings)
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

    def test_walk_pages_over_http_without_overlap(self) -> None:
        seen: list[str] = []
        path = "/bookings?limit=2"
        while True:
            status, body = self._get(path)
            self.assertEqual(status, 200)
            seen.extend(item["booking_id"] for item in body["items"])
            if not body["has_more"]:
                self.assertIsNone(body["next_cursor"])
                break
            path = f"/bookings?limit=2&cursor={body['next_cursor']}"
        # 遍历覆盖服务端当前全部预约，且无重复（无重叠）
        expected = {b["booking_id"] for b in self.bookings.list_bookings()}
        self.assertEqual(set(seen), expected)
        self.assertEqual(len(seen), len(set(seen)))

    def test_corrupt_cursor_is_identifiable_client_error(self) -> None:
        status, body = self._get("/bookings?cursor=tampered-value")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_cursor")

    def test_empty_cursor_is_identifiable_client_error(self) -> None:
        status, body = self._get("/bookings?cursor=")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_cursor")

    def test_bad_limit_is_client_error(self) -> None:
        status, body = self._get("/bookings?limit=abc")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "validation_error")

    def test_new_booking_during_http_walk_does_not_overlap(self) -> None:
        status, first = self._get("/bookings?limit=2")
        self.assertEqual(status, 200)
        first_ids = [item["booking_id"] for item in first["items"]]
        _apply_n(self.bookings, self.ids, 900, 2)
        status, second = self._get(f"/bookings?limit=2&cursor={first['next_cursor']}")
        self.assertEqual(status, 200)
        self.assertTrue(set(first_ids).isdisjoint(item["booking_id"] for item in second["items"]))


if __name__ == "__main__":
    unittest.main()
