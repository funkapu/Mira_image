"""Sentiment analyzer — Groq Llama-3.3-70B, positive/neutral/negative + intensity"""
import json
import re

from mira_graph.llm import call_llm, _TIMEOUT_SUBAGENT
from mira_graph.state import MiraState

SENTIMENT_PROMPT = """ประเมิน sentiment ของข้อความ:

"{user_text}"

ตอบ JSON เท่านั้น:
{{"sentiment": "<positive|neutral|negative>", "intensity": <0-1>}}"""

_NEG_WORDS = ["เครียด", "แย่", "เหนื่อย", "เศร้า", "กังวล", "ห่วย", "ผิด", "โกรธ", "กลัว"]
_POS_WORDS = ["ดีขึ้น", "ขอบคุณ", "ดีใจ", "มีความสุข", "โล่ง", "สบายใจ", "เข้าใจ"]


def _heuristic_sentiment(text: str) -> str:
    neg = sum(1 for w in _NEG_WORDS if w in text)
    pos = sum(1 for w in _POS_WORDS if w in text)
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral"


async def sentiment_node(state: MiraState) -> dict:
    """Analyze sentiment polarity + intensity. Falls back to keyword heuristic on failure."""
    user_text = state.get("user_text", "") or ""

    if not user_text or len(user_text) < 3:
        print("  [SENTIMENT] skipped")
        return {"parallel_results": ["sentiment_skipped"]}

    try:
        response, latency = await call_llm(
            messages=[{"role": "user", "content": SENTIMENT_PROMPT.format(user_text=user_text)}],
            max_tokens=50,
            json_mode=True,
            timeout=_TIMEOUT_SUBAGENT,
        )
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            m = re.search(r"\{[^}]+\}", response)
            result = json.loads(m.group(0)) if m else {}

        sentiment = result.get("sentiment") or _heuristic_sentiment(user_text)
        intensity = result.get("intensity", 0.5)
        print(f"  [SENTIMENT] {sentiment} ({intensity:.2f}) ({latency:.0f}ms)")

    except Exception as exc:
        sentiment = _heuristic_sentiment(user_text)
        print(f"  [SENTIMENT] LLM failed ({exc.__class__.__name__}), heuristic={sentiment}")

    return {
        "sentiment": sentiment,
        "parallel_results": ["sentiment_done"],
    }
