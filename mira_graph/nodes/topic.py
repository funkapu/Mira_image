"""Topic classifier — Groq Llama-3.3-70B categorizes user's main concern"""
import json
import re

from mira_graph.llm import call_llm, _TIMEOUT_SUBAGENT
from mira_graph.state import MiraState

TOPIC_PROMPT = """จัดประเภทหัวข้อหลักของข้อความนี้:

"{user_text}"

เลือก 1 หัวข้อ:
- work (งาน, อาชีพ, หัวหน้า, เพื่อนร่วมงาน)
- family (ครอบครัว, พ่อแม่, ลูก, พี่น้อง)
- relationship (ความรัก, แฟน, คู่รัก)
- health (สุขภาพ, ป่วย, ร่างกาย)
- study (เรียน, มหาลัย, สอบ)
- finance (เงิน, หนี้, รายได้)
- self (ตัวเอง, ความมั่นใจ, identity)
- other (อื่นๆ)

ตอบ JSON เท่านั้น:
{{"topic": "<หัวข้อ>", "confidence": <0-1>}}"""

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "work": ["งาน", "หัวหน้า", "บริษัท", "ออฟฟิศ", "เพื่อนร่วมงาน", "ลางาน", "ไล่ออก"],
    "family": ["พ่อ", "แม่", "ครอบครัว", "พี่", "น้อง", "ลูก"],
    "relationship": ["แฟน", "คู่รัก", "เลิก", "ชอบ", "รัก", "หึง"],
    "health": ["ป่วย", "หมอ", "โรค", "เจ็บ", "นอนไม่หลับ"],
    "study": ["เรียน", "สอบ", "มหาลัย", "โรงเรียน", "เกรด"],
    "finance": ["เงิน", "หนี้", "รายได้", "ค่าใช้จ่าย", "ยืม"],
    "self": ["ตัวเอง", "มั่นใจ", "คุณค่า", "ห่วย", "ไม่ดี"],
}


def _heuristic_topic(text: str) -> str:
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return topic
    return "other"


async def topic_classifier_node(state: MiraState) -> dict:
    """Classify user's main topic. Falls back to keyword heuristic on LLM failure."""
    user_text = state.get("user_text", "") or ""

    if not user_text or len(user_text) < 5:
        print("  [TOPIC] skipped")
        return {"parallel_results": ["topic_skipped"]}

    try:
        response, latency = await call_llm(
            messages=[{"role": "user", "content": TOPIC_PROMPT.format(user_text=user_text)}],
            max_tokens=60,
            json_mode=True,
            timeout=_TIMEOUT_SUBAGENT,
        )
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            m = re.search(r"\{[^}]+\}", response)
            result = json.loads(m.group(0)) if m else {}

        topic = result.get("topic") or _heuristic_topic(user_text)
        print(f"  [TOPIC] {topic} ({latency:.0f}ms)")

    except Exception as exc:
        topic = _heuristic_topic(user_text)
        print(f"  [TOPIC] LLM failed ({exc.__class__.__name__}), heuristic={topic}")

    return {
        "topic": topic,
        "parallel_results": ["topic_done"],
    }
