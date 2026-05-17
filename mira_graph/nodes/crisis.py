"""Crisis detection — instant regex check, no LLM call"""
import re

from mira_graph.state import MiraState

# Direct / active suicidal ideation — HIGH severity
CRISIS_KEYWORDS_HIGH = [
    "ฆ่าตัวตาย",
    "อยากตาย",
    "ไม่อยากมีชีวิต",
    "ไม่อยากอยู่แล้ว",
    "อยากจบชีวิต",
    "ทำร้ายตัวเอง",
    "ฉันจะตาย",
    "อยากหายไป",
    "จบมัน",
]

# Passive / indirect suicidal ideation — MODERATE severity
CRISIS_KEYWORDS_MODERATE = [
    "อยู่ไปทำไม",
    "จะอยู่ไปทำไม",
    "เกิดมาทำไม",
    "ไม่มีประโยชน์ที่จะอยู่",
    "เป็นภาระ",
    "ตัวเองเป็นภาระ",
    "ไม่มีใครต้องการ",
    "หายไปจากโลก",
    "ไม่อยากตื่น",
    "นอนหลับไปเลยไม่ต้องตื่น",
    "ชีวิตไม่มีความหมาย",
    "ไร้ค่า",
    "ชีวิตไร้จุดหมาย",
    "หมดหวัง",
    "ไม่ไหวแล้ว",
    "what's the point",
    "burden",
]

_RE_HIGH = re.compile("|".join(re.escape(k) for k in CRISIS_KEYWORDS_HIGH))
_RE_MODERATE = re.compile("|".join(re.escape(k) for k in CRISIS_KEYWORDS_MODERATE))

# Legacy alias for imports that reference CRISIS_KEYWORDS
CRISIS_KEYWORDS = CRISIS_KEYWORDS_HIGH + CRISIS_KEYWORDS_MODERATE
CRISIS_REGEX = re.compile("|".join(re.escape(k) for k in CRISIS_KEYWORDS))


def _find_keyword(text: str, pattern: re.Pattern) -> str | None:
    m = pattern.search(text)
    return m.group(0) if m else None


async def crisis_check_node(state: MiraState) -> dict:
    """Instant crisis detection. Sets crisis_detected + _crisis_severity + matched keyword."""
    user_text = state.get("user_text", "") or ""

    high_kw = _find_keyword(user_text, _RE_HIGH)
    if high_kw:
        print(f"  [CRISIS 🚨] HIGH — '{high_kw}'")
        return {
            "crisis_detected": True,
            "_crisis_severity": "high",
            "_crisis_keyword": high_kw,
            "parallel_results": ["crisis_done"],
        }

    mod_kw = _find_keyword(user_text, _RE_MODERATE)
    if mod_kw:
        print(f"  [CRISIS ⚠️] MODERATE — '{mod_kw}'")
        return {
            "crisis_detected": True,
            "_crisis_severity": "moderate",
            "_crisis_keyword": mod_kw,
            "parallel_results": ["crisis_done"],
        }

    print("  [CRISIS] clear")
    return {
        "crisis_detected": False,
        "_crisis_severity": None,
        "_crisis_keyword": None,
        "parallel_results": ["crisis_done"],
    }
