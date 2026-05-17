"""Mood detector — Groq Llama-3.3-70B extracts mood score 1-10"""
import json
import re

from mira_graph.llm import call_llm, _TIMEOUT_SUBAGENT
from mira_graph.state import MiraState

MOOD_PROMPT = """ประเมิน mood ของผู้พูดจากข้อความนี้:

"{user_text}"

ถ้าผู้พูดบอกตัวเลขชัดเจน (เช่น "รู้สึก 4" หรือ "4/10") ใช้ตัวเลขนั้นทันที
ถ้าไม่บอก ให้เดาจาก tone (1=แย่มาก, 10=ดีมาก)

ตอบ JSON เท่านั้น:
{{"mood": <1-10>, "confidence": <0-1>, "label": "<คำสั้นๆภาษาไทย>"}}"""

# Heuristic: detect explicit number in text like "รู้สึก 4" or "4/10"
_MOOD_NUM = re.compile(r"(?:รู้สึก|mood|คะแนน)\s*([1-9]|10)\b|([1-9]|10)\s*/\s*10")


def _heuristic_mood(text: str) -> int | None:
    m = _MOOD_NUM.search(text)
    if m:
        val = m.group(1) or m.group(2)
        return int(val)
    return None


async def mood_detector_node(state: MiraState) -> dict:
    """Extract mood score 1-10. Falls back to heuristic regex on LLM failure."""
    user_text = state.get("user_text", "") or ""

    if not user_text or len(user_text) < 3:
        print("  [MOOD] skipped (text too short)")
        return {"parallel_results": ["mood_skipped"]}

    try:
        response, latency = await call_llm(
            messages=[{"role": "user", "content": MOOD_PROMPT.format(user_text=user_text)}],
            max_tokens=80,
            json_mode=True,
            timeout=_TIMEOUT_SUBAGENT,
        )
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            m = re.search(r"\{[^}]+\}", response)
            result = json.loads(m.group(0)) if m else {}

        mood = result.get("mood") or _heuristic_mood(user_text)
        label = result.get("label", "")
        print(f"  [MOOD] {mood}/10 \"{label}\" ({latency:.0f}ms)")

    except Exception as exc:
        mood = _heuristic_mood(user_text)
        print(f"  [MOOD] LLM failed ({exc.__class__.__name__}), heuristic={mood}")

    return {
        "mood_score": mood,
        "parallel_results": ["mood_done"],
    }
