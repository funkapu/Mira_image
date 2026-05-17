"""Combined sub-agent — mood + topic + sentiment in 1 LLM call (saves 2 RPM/turn)"""
import json
import re

from mira_graph.llm import call_llm_fast, _TIMEOUT_SUBAGENT
from mira_graph.state import MiraState

COMBINED_PROMPT = """วิเคราะห์ข้อความนี้และตอบเป็น JSON:

"{user_text}"

ประเมิน 3 มิติ:
1. mood: คะแนน 1-10 (1=แย่มาก, 10=ดีมาก) — ถ้าผู้พูดบอกตัวเลขชัด ใช้ตัวเลขนั้น
2. topic: work | family | relationship | health | study | finance | self | other
3. sentiment: positive | neutral | negative

ตอบ JSON เท่านั้น:
{{
  "mood": <1-10 หรือ null>,
  "mood_label": "<คำสั้นๆภาษาไทย>",
  "topic": "<หัวข้อ>",
  "sentiment": "<positive|neutral|negative>",
  "intensity": <0-1>
}}"""

# Heuristic fallbacks when LLM fails
_MOOD_NUM = re.compile(r"(?:รู้สึก|mood)\s*([1-9]|10)\b|([1-9]|10)\s*/\s*10")

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "work": ["งาน", "หัวหน้า", "บริษัท", "ออฟฟิศ", "เพื่อนร่วมงาน"],
    "family": ["พ่อ", "แม่", "ครอบครัว", "พี่", "น้อง", "ลูก"],
    "relationship": ["แฟน", "คู่รัก", "เลิก", "รัก", "ชอบ"],
    "health": ["ป่วย", "หมอ", "โรค", "เจ็บ", "นอนไม่หลับ"],
    "study": ["เรียน", "สอบ", "มหาลัย", "โรงเรียน"],
    "finance": ["เงิน", "หนี้", "รายได้"],
    "self": ["ตัวเอง", "มั่นใจ", "คุณค่า", "ห่วย"],
}
_NEG_WORDS = ["เครียด", "แย่", "เหนื่อย", "เศร้า", "กังวล", "ห่วย", "โกรธ", "กลัว"]
_POS_WORDS = ["ดีขึ้น", "ขอบคุณ", "ดีใจ", "สุข", "โล่ง", "สบาย", "เข้าใจ"]


def _heuristic(text: str) -> tuple[int | None, str, str]:
    m = _MOOD_NUM.search(text)
    mood = int(m.group(1) or m.group(2)) if m else None

    topic = "other"
    for t, kws in _TOPIC_KEYWORDS.items():
        if any(kw in text for kw in kws):
            topic = t
            break

    neg = sum(1 for w in _NEG_WORDS if w in text)
    pos = sum(1 for w in _POS_WORDS if w in text)
    sentiment = "negative" if neg > pos else ("positive" if pos > neg else "neutral")

    return mood, topic, sentiment


async def analyze_node(state: MiraState) -> dict:
    """Single LLM call for mood + topic + sentiment. Heuristic fallback on failure."""
    user_text = state.get("user_text", "") or ""

    if len(user_text) < 5:
        print("  [ANALYZE] skipped (text too short)")
        return {"parallel_results": ["analyze_skipped"]}

    try:
        response, latency = await call_llm_fast(
            messages=[{"role": "user", "content": COMBINED_PROMPT.format(user_text=user_text)}],
            max_tokens=120,
            json_mode=True,
            timeout=_TIMEOUT_SUBAGENT,
        )

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            m = re.search(r"\{.*?\}", response, re.DOTALL)
            result = json.loads(m.group(0)) if m else {}

        mood = result.get("mood")
        mood_label = result.get("mood_label", "")
        topic = result.get("topic") or "other"
        sentiment = result.get("sentiment") or "neutral"
        intensity = result.get("intensity", 0.5)

        # Sanity-check: if explicit number in text, trust it over LLM
        heuristic_mood, _, _ = _heuristic(user_text)
        if heuristic_mood is not None:
            mood = heuristic_mood

        print(f"  [ANALYZE] mood={mood} \"{mood_label}\" | topic={topic} | sent={sentiment}({intensity:.2f}) ({latency:.0f}ms)")

        return {
            "mood_score": mood,
            "topic": topic,
            "sentiment": sentiment,
            "parallel_results": ["analyze_done"],
        }

    except Exception as exc:
        mood, topic, sentiment = _heuristic(user_text)
        print(f"  [ANALYZE] LLM failed ({exc.__class__.__name__}), heuristic: mood={mood} topic={topic} sent={sentiment}")
        return {
            "mood_score": mood,
            "topic": topic,
            "sentiment": sentiment,
            "parallel_results": ["analyze_failed"],
        }
