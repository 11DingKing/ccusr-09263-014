"""预约列表游标的编码与解码（仅标准库）。

游标承载键集分页的排序锚点 ``(created_at, booking_id)``：
- ``created_at`` 为预约创建时刻的 UTC ISO-8601 字符串（与持久化文档一致）；
- ``booking_id`` 是并列时的最终决胜键，保证排序全序、稳定且无重复。

完整性由 HMAC-SHA256 签名保护：任何截断、改写或伪造都会在解码时抛出
:class:`~service_09252_008.domain.errors.CursorError`，接口边界据此返回
可识别的客户端错误（``invalid_cursor``），而不会静默回退到首页。

编码布局（base64url，无填充）::

    v1.<urlsafe_b64(payload_json)>.<urlsafe_b64(hmac_sha256(secret, payload))>
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime
from typing import Any

from ..domain.errors import CursorError
from ..domain.models import dt_from_str, dt_to_str

_CURSOR_VERSION = "v1"
_PURPOSE = "bookings:keyset"


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    try:
        padded = text + "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise CursorError("cursor is not valid base64url") from exc


class BookingCursor:
    """预约列表的排序锚点。"""

    __slots__ = ("created_at", "booking_id")

    def __init__(self, created_at: datetime, booking_id: str) -> None:
        if created_at.tzinfo is None:
            raise ValueError("cursor created_at must be timezone-aware")
        if not isinstance(booking_id, str) or not booking_id:
            raise ValueError("cursor booking_id must be a non-empty string")
        self.created_at = created_at
        self.booking_id = booking_id

    def to_anchor(self) -> tuple[str, str]:
        """返回可直接与持久化文档比较的 ``(created_at ISO, booking_id)``。"""
        return dt_to_str(self.created_at), self.booking_id


def encode_cursor(cursor: BookingCursor, secret: bytes) -> str:
    """把排序锚点编码为带签名的不透明字符串。"""
    payload = json.dumps(
        {"p": _PURPOSE, "ts": dt_to_str(cursor.created_at), "id": cursor.booking_id},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hmac.new(secret, payload, hashlib.sha256).digest()
    return f"{_CURSOR_VERSION}.{_b64encode(payload)}.{_b64encode(signature)}"


def decode_cursor(token: str | None, secret: bytes) -> BookingCursor | None:
    """解码并校验游标；``None`` 表示首页。损坏时抛出 :class:`CursorError`。"""
    if token is None:
        return None
    if not isinstance(token, str) or not token:
        raise CursorError("cursor must be a non-empty string")
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != _CURSOR_VERSION:
        raise CursorError("cursor is malformed or uses an unsupported version")
    _version, payload_b64, signature_b64 = parts
    payload_raw = _b64decode(payload_b64)
    provided_signature = _b64decode(signature_b64)
    expected_signature = hmac.new(secret, payload_raw, hashlib.sha256).digest()
    if not hmac.compare_digest(provided_signature, expected_signature):
        raise CursorError("cursor signature does not match")
    try:
        payload: dict[str, Any] = json.loads(payload_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CursorError("cursor payload is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("p") != _PURPOSE:
        raise CursorError("cursor payload is not intended for this listing")
    ts_text = payload.get("ts")
    booking_id = payload.get("id")
    if not isinstance(ts_text, str) or not isinstance(booking_id, str) or not booking_id:
        raise CursorError("cursor payload is missing sort anchor fields")
    try:
        created_at = dt_from_str(ts_text)
    except ValueError as exc:
        raise CursorError(f"cursor timestamp is invalid: {exc}") from exc
    return BookingCursor(created_at, booking_id)
