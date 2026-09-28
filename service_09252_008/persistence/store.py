"""存储端口与内存实现。

存储层只面向“集合 + 键 + JSON 文档”，不理解领域语义；
事务边界由应用服务控制，两个实现都保证：
- ``transaction`` 可重入（嵌套时并入外层事务）；
- 外层事务异常时整体回滚；
- 读写对调用方返回深拷贝，避免共享可变状态。
"""
from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Protocol


class Store(Protocol):
    """文档型存储端口。"""

    def transaction(self) -> Iterator[None]:
        """进入事务上下文（可重入）。"""
        ...

    def get(self, collection: str, key: str) -> dict[str, Any] | None:
        ...

    def put(self, collection: str, key: str, record: dict[str, Any]) -> None:
        ...

    def delete(self, collection: str, key: str) -> None:
        ...

    def query(self, collection: str, **filters: Any) -> list[dict[str, Any]]:
        """按顶层字段等值过滤；无过滤条件时返回整个集合。"""
        ...

    def scan_page(
        self,
        collection: str,
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
        anchor_created_at: str | None = None,
        anchor_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """键集分页：按 ``(created_at, key)`` 升序取下一页。

        - ``limit`` 为期望取回的条数（含锚点之后的严格“下一条”起算），
          调用方一般传 ``page_size + 1`` 以判断是否还有下一页；
        - ``filters`` 为顶层字段等值过滤，必须与游标签发时的过滤条件一致；
        - 锚点为 ``None`` 时取首页；否则只返回严格晚于
          ``(anchor_created_at, anchor_key)`` 的记录。

        实现必须在同一读取快照内完成过滤、排序与限量，
        保证并发写入不会让相邻两页出现重叠或遗漏。
        """
        ...

    def close(self) -> None:
        ...


class InMemoryStore:
    """进程内存储：测试与演示用；通过快照实现事务回滚。"""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self._depth = 0
        self._snapshot: dict[str, dict[str, dict[str, Any]]] | None = None

    @contextmanager
    def transaction(self) -> Iterator[None]:
        # 排他锁贯穿整个事务：同一线程可重入，其他线程阻塞至外层事务结束，
        # 从而保证“读-判-写”序列的原子性（并发锁定只有一个成功者）。
        self._lock.acquire()
        try:
            outermost = self._depth == 0
            if outermost:
                self._snapshot = deepcopy(self._data)
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if outermost:
                    if self._snapshot is not None:
                        self._data = self._snapshot
                    self._snapshot = None
                raise
            else:
                self._depth -= 1
                if outermost:
                    self._snapshot = None
        finally:
            self._lock.release()

    def get(self, collection: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._data.get(collection, {}).get(key)
            return deepcopy(record) if record is not None else None

    def put(self, collection: str, key: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._data.setdefault(collection, {})[key] = deepcopy(record)

    def delete(self, collection: str, key: str) -> None:
        with self._lock:
            self._data.get(collection, {}).pop(key, None)

    def query(self, collection: str, **filters: Any) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._data.get(collection, {}).values())
        result = [r for r in records if all(r.get(field) == value for field, value in filters.items())]
        return deepcopy(result)

    def scan_page(
        self,
        collection: str,
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
        anchor_created_at: str | None = None,
        anchor_key: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = filters or {}
        with self._lock:
            # 持锁期间一次性取一致快照：过滤、键集比较、排序与限量都在这份
            # 深拷贝上完成，事务内的并发写入不会混入本次扫描（与 SQLite 的
            # 单条 SELECT 语句快照语义对齐）。
            entries = [
                (key, deepcopy(record))
                for key, record in self._data.get(collection, {}).items()
            ]
        rows: list[tuple[str, str, dict[str, Any]]] = []
        for key, record in entries:
            if not all(record.get(field) == value for field, value in filters.items()):
                continue
            created_at = str(record.get("created_at"))
            if anchor_created_at is not None and anchor_key is not None:
                if (created_at, key) <= (anchor_created_at, anchor_key):
                    continue
            rows.append((created_at, key, record))
        rows.sort(key=lambda item: (item[0], item[1]))
        return [deepcopy(record) for _, _, record in rows[:limit]]

    def close(self) -> None:  # pragma: no cover - 对称接口
        pass
