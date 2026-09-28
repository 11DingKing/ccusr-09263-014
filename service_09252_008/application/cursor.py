"""不透明预约分页游标：编码/解码与完整性校验全部在 Python 侧完成。

游标载荷包含：
- 排序锚点 ``created_at`` / ``booking_id``（与列表的
  ``ORDER BY (created_at, booking_id)`` 完全一致）；
- ``scope``：本次列表过滤条件的指纹，游标跨条件复用时直接拒绝；
- 版本号，便于以后平滑演进。

载荷经 URL-safe base64 编码后追加 HMAC-SHA256 签名，客户端无法伪造，
任何截断、篡改、错配都会抛出 :class:`InvalidCursorError` —— 它是
:class:`~..domain.errors.ValidationError` 的子类，接口边界映射为 400，
客户端可凭错误码 ``invalid_cursor`` 识别并丢弃游标重新从头翻页。

存储层只按解码后的锚点做键集比较，永远不会把游标内容拼进 SQL。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass

from ..domain.errors import ValidationError
from ..domain.models import dt_from_str

CURSOR_VERSION = 1
_PREFIX = "bkgcur"


class InvalidCursorError(ValidationError):
    """游标损坏、签名不匹配、版本不支持或与查询参数不一致。"""

    code = "invalid_cursor"


@dataclass(frozen=True)
class CursorAnchor:
    """排序锚点：与 ``ORDER BY (created_at, booking_id)`` 一致。"""

    created_at: str
    booking_id: str


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def encode_cursor(anchor: CursorAnchor, scope: str, secret: bytes) -> str:
    """把锚点编码为带签名的不透明游标字符串。"""
    body = {
        "v": CURSOR_VERSION,
        "scope": scope,
        "created_at": anchor.created_at,
        "booking_id": anchor.booking_id,
    }
    payload = _b64encode(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(secret, payload.encode("ascii"), hashlib.sha256).digest()
    return f"{_PREFIX}.{payload}.{_b64encode(signature)}"


def decode_cursor(token: str, *, scope: str, secret: bytes) -> CursorAnchor:
    """解码并完整校验游标；任何形式的损坏都抛 :class:`InvalidCursorError`。"""

    def fail(reason: str) -> None:
        raise InvalidCursorError(
            f"invalid pagination cursor: {reason}", details={"reason": reason}
        )

    if not isinstance(token, str):
        fail("malformed")
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != _PREFIX or not parts[1] or not parts[2]:
        fail("malformed")
    _, payload_b64, signature_b64 = parts

    expected_signature = hmac.new(secret, payload_b64.encode("ascii"), hashlib.sha256).digest()
    try:
        signature = _b64decode(signature_b64)
    except (ValueError, TypeError):
        fail("bad_signature")
    if not hmac.compare_digest(signature, expected_signature):
        fail("bad_signature")

    try:
        body = json.loads(_b64decode(payload_b64).decode("utf-8"))
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        fail("malformed_payload")
    if not isinstance(body, dict) or body.get("v") != CURSOR_VERSION:
        fail("unsupported_version")
    if not isinstance(body.get("scope"), str) or body["scope"] != scope:
        fail("scope_mismatch")
    created_at = body.get("created_at")
    booking_id = body.get("booking_id")
    if not isinstance(created_at, str) or not created_at:
        fail("malformed_anchor")
    if not isinstance(booking_id, str) or not booking_id:
        fail("malformed_anchor")
    try:
        dt_from_str(created_at)
    except (ValueError, TypeError):
        fail("malformed_anchor")
    return CursorAnchor(created_at=created_at, booking_id=booking_id)
