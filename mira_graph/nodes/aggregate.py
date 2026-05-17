"""Fan-in: wait for all 5 supervisors, parse outputs into state updates."""
import re

from mira_graph.state import MiraState

_THAI_NUMBERS = {
    "หนึ่ง": 1, "สอง": 2, "สาม": 3, "สี่": 4, "ห้า": 5,
    "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9, "สิบ": 10,
}
_MOOD_NUM_RE = re.compile(
    r'(?:ประมาณ|สัก|ระดับ|ให้|เลข|mood|score)?\s*(\d{1,2})\s*(?:ครับ|ค่ะ|คะ|เลย|แล้ว|คะแนน)?',
    re.IGNORECASE,
)


def _parse_field(text: str, field: str) -> str:
    for line in text.splitlines():
        if line.strip().lower().startswith(field.lower() + ":"):
            val = line.split(":", 1)[-1].strip()
            if val and val != "-":
                return val
    return ""


def _extract_mood_from_user(user_text: str) -> int | None:
    """Return explicit 1-10 mood score only when user states a number."""
    for m in _MOOD_NUM_RE.finditer(user_text):
        num = int(m.group(1))
        if 1 <= num <= 10:
            return num
    for word, num in _THAI_NUMBERS.items():
        if word in user_text:
            return num
    return None


async def aggregate_node(state: MiraState) -> dict:
    sups = []
    updates: dict = {}

    # === Empathy — count supervisor, but do NOT auto-set mood_score ===
    # mood_score is only set when user explicitly says a number (below)
    emp = state.get("supervisor_empathy") or ""
    if emp:
        sups.append("empathy")

    # === mood_score — only from user's explicit number ===
    if state.get("mood_score") is None:
        num = _extract_mood_from_user(state.get("user_text") or "")
        if num is not None:
            updates["mood_score"] = num
            print(f"  [AGG] mood_score = {num} (user explicit)")

    # === Belief → automatic thought + distortion ===
    bel = state.get("supervisor_belief") or ""
    if bel:
        sups.append("belief")
        if _parse_field(bel, "distortion").lower() == "yes":
            dist_type = _parse_field(bel, "type")
            quote = _parse_field(bel, "quote").strip('"\'')
            if quote and not state.get("automatic_thought"):
                updates["automatic_thought"] = quote
                updates["distortion"] = dist_type
                print(f"  [AGG] AT = '{quote[:50]}' ({dist_type})")

    # === Reflection → main concern (S2+ only) ===
    ref = state.get("supervisor_reflection") or ""
    if ref:
        sups.append("reflection")
        if state.get("phase") != "CHECKIN" and not state.get("main_concern"):
            summary = _parse_field(ref, "summary")
            if summary:
                updates["main_concern"] = summary
                print(f"  [AGG] concern = '{summary[:50]}'")

    if state.get("supervisor_strategy"):
        sups.append("strategy")
    if state.get("supervisor_encouragement"):
        sups.append("encouragement")


    print(f"  [AGGREGATE] {len(sups)}/5 supervisors: {sups}")
    return updates
