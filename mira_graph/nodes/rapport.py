from __future__ import annotations

import re

from mira_graph.nodes._phase_base import run_phase
from mira_graph.state import MiraState

_NAME_PATTERNS = [
    re.compile(r"ชื่อ\s*([ก-๙a-zA-Z]{1,15})"),
    re.compile(r"เรียก(?:ผม|ฉัน|หนู)?ว่า\s*([ก-๙a-zA-Z]{1,15})"),
    re.compile(r"เรียกว่า\s*([ก-๙a-zA-Z]{1,15})"),
    re.compile(r"เรียก\s*([ก-๙]{1,12})\s*(?:ก็ได้|นะ|ครับ|ค่ะ|คะ)"),
]
_STRIP_SUFFIX = re.compile(
    r"(ก็ได้|ครับ|ค่ะ|คะ|นะคะ|นะครับ|นะ|จ้า|เลย|ด้วย|เองครับ|เองค่ะ)+$"
)


def _extract_name(text: str) -> str | None:
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            name = _STRIP_SUFFIX.sub("", m.group(1)).strip()
            if 1 <= len(name) <= 12:
                return name
    return None


async def checkin_node(state: MiraState) -> dict:
    result = await run_phase("CHECKIN", state, max_tokens=800)

    existing_name = state.get("user_name")
    tool_name = result.get("user_name")
    fallback_name = _extract_name(state.get("user_text") or "")

    final_name = existing_name or tool_name or fallback_name
    if final_name:
        result["user_name"] = final_name

    return result
