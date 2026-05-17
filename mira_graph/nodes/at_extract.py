"""Automatic Thought Extraction Supervisor.

Cerebras-based supervisor that scans the latest user_text for automatic
thoughts (cognitive statements about self/others/future), identifies which
of the 11 distortions applies, and updates state[automatic_thought] +
state[distortion]. This unblocks judge.py's S2→S3 gate so Mira can actually
do CBT reframing instead of looping in S2 forever.

Sticky semantics:
  - Once an AT is captured, only overwrite if the new candidate has higher
    confidence AND is from a fresh user statement (not agreement with Mira).
"""
from __future__ import annotations

import re
from pathlib import Path

from mira_graph.llm import call_supervisor
from mira_graph.state import MiraState

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "sup_at.txt"

# Map English keys → Thai labels (for state[distortion] which lands in Mira's prompt)
_DISTORTION_LABELS = {
    "D1_all_or_nothing": "ความคิดขาว-ดำ",
    "D2_overgeneralization": "การสรุปเกินจริง",
    "D3_mental_filter": "กรองแต่ด้านลบ",
    "D4_disqualify_positive": "ปฏิเสธด้านบวก",
    "D5_mind_reading": "การอ่านใจคนอื่น",
    "D6_fortune_teller": "การทำนายอนาคตร้าย",
    "D7_magnification": "การขยายเกินจริง",
    "D8_emotional_reasoning": "การใช้อารมณ์เป็นเหตุผล",
    "D9_should_statements": "ความคิด ควร/ต้อง",
    "D10_labeling": "การติดป้ายตัวเอง",
    "D11_personalization": "การโทษตัวเองเกินควร",
}

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def _build_at_context(state: MiraState) -> str:
    lines = [f"Phase: {state.get('phase', 'CHECKIN')}"]

    existing_at = state.get("automatic_thought")
    existing_dist = state.get("distortion")
    if existing_at:
        lines.append(f"Already captured AT: {existing_at}")
        lines.append(f"Already captured distortion: {existing_dist or '?'}")
        lines.append("(already_captured = true — only overwrite if higher confidence)")
    else:
        lines.append("Already captured AT: (none yet)")

    msgs = state.get("messages", [])[-4:]
    user_lines = []
    mira_lines = []
    for m in msgs:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = (m.get("content") or "")[:80]
            if role == "user":
                user_lines.append(content)
            elif role == "assistant":
                mira_lines.append(content)
    if mira_lines:
        lines.append("Mira recently said (DO NOT extract from these):")
        for s in mira_lines:
            lines.append(f"  - {s}")
    if user_lines:
        lines.append("User recent messages:")
        for s in user_lines:
            lines.append(f"  - {s}")
    return "\n".join(lines)


def _parse_at_output(raw: str) -> dict:
    out = {
        "extracted_at": "none",
        "distortion": "none",
        "confidence": "low",
        "already_captured": False,
        "reasoning": "",
    }
    # Try JSON first
    s = raw.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            import json as _json
            data = _json.loads(s)
            if isinstance(data, dict):
                data = {str(k).lower(): v for k, v in data.items()}
                for key in out:
                    if key in data:
                        v = data[key]
                        if key == "already_captured":
                            out[key] = bool(v) if isinstance(v, bool) else str(v).strip().lower() == "true"
                        else:
                            out[key] = str(v).strip()
                return out
        except Exception:
            pass

    # Line-based parser — tolerant to markdown/bullets
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[\-*•·>\s]+", "", line)
        line = re.sub(r"\*+", "", line).strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        val = val.strip().strip('"').strip("'").strip(",").strip()
        if key == "extracted_at":
            out["extracted_at"] = val
        elif key == "distortion":
            out["distortion"] = val
        elif key == "confidence":
            out["confidence"] = val.lower()
        elif key == "already_captured":
            out["already_captured"] = val.lower() == "true"
        elif key == "reasoning":
            out["reasoning"] = val
    return out


def _normalize_distortion(val: str) -> str:
    """Map 'D6', 'd6_fortune_teller', 'D6 fortune teller' → 'D6_fortune_teller'.
    Returns 'none' if unrecognizable."""
    if not val:
        return "none"
    v = val.strip().strip('"').strip("'").lower()
    if v in ("none", "[]", "null", ""):
        return "none"
    for full in _DISTORTION_LABELS:
        if v == full.lower():
            return full
    short = re.match(r"^(d\d{1,2})\b", v)
    if short:
        prefix = short.group(1).upper()
        for full in _DISTORTION_LABELS:
            if full.startswith(prefix + "_"):
                return full
    return "none"


def _is_meaningful_at(at: str) -> bool:
    """Reject obvious non-AT outputs the model might still emit."""
    if not at or at.strip().lower() in ("none", "null", "n/a", ""):
        return False
    # Too short to be a thought
    if len(at.strip()) < 4:
        return False
    return True


# Keyword-based AT detector — fires when LLM supervisor misses clear cognitive
# statements like "ผมโง่" / "ผมเป็นคนล้มเหลว" / "ผมต้องทำให้เสร็จ".
# Each tuple: (regex pattern, distortion key)
_AT_KEYWORD_PATTERNS = [
    # D10 labeling
    (re.compile(r"(เป็นคนล้มเหลว|ล้มเหลว|ผมโง่|ฉันโง่|หนูโง่|กูโง่|ไร้ค่า|ไม่ดีพอ|ไม่เก่งพอ)"), "D10_labeling"),
    # D9 should statements
    (re.compile(r"(ผมต้อง|ฉันต้อง|หนูต้อง|กูต้อง|ผมควร|ฉันควร|หนูควร)\S+"), "D9_should_statements"),
    # D6 fortune teller
    (re.compile(r"(กลัวว่า|กลัวจะ|กลัวทำไม่ทัน|จะแย่แน่|จะพังแน่|คงไม่)"), "D6_fortune_teller"),
    # D5 mind reading
    (re.compile(r"(เพื่อนคง|พ่อแม่คง|ทุกคนคง|คนอื่นคง|คนรอบข้างคง)"), "D5_mind_reading"),
    # D7 magnification / catastrophizing
    (re.compile(r"(ชีวิตจบ|ชีวิตพัง|จบเห่|ไม่มีทางออก|ทำอะไรไม่ได้แล้ว)"), "D7_magnification"),
    # D2 overgeneralization
    (re.compile(r"(ทุกครั้งที่|ผมพลาดเสมอ|ฉันพลาดเสมอ|ทำอะไรก็ไม่ได้เรื่อง|ทำอะไรก็ไม่สำเร็จ)"), "D2_overgeneralization"),
]


def _keyword_extract_at(text: str) -> tuple[str, str] | None:
    """Last-resort regex AT detection. Returns (matched_phrase, distortion_key) or None.

    Used when the LLM supervisor returns "no AT" but user clearly stated one.
    """
    for pattern, dist in _AT_KEYWORD_PATTERNS:
        m = pattern.search(text)
        if m:
            # Capture surrounding context (~30 chars on each side) as the AT phrase
            start = max(0, m.start() - 5)
            end = min(len(text), m.end() + 25)
            phrase = text[start:end].strip()
            # Trim to first sentence boundary if any
            for sep in ["?", "!", ".", "。", "。"]:
                if sep in phrase:
                    phrase = phrase.split(sep)[0].strip()
            if len(phrase) >= 4:
                return phrase, dist
    return None


async def at_supervisor(state: MiraState) -> dict:
    """Detect automatic thought + distortion, update state if higher confidence.

    Always returns at minimum the existing AT/distortion (sticky behavior).
    """
    user_text = (state.get("user_text") or "").strip()
    existing_at = state.get("automatic_thought")
    existing_dist = state.get("distortion")
    existing_conf = state.get("_at_confidence") or "low"

    base_update: dict = {
        "automatic_thought": existing_at,
        "distortion": existing_dist,
        "_at_confidence": existing_conf,
    }

    # Skip very short replies (greetings, "ครับ", "ใช่") — never contain ATs.
    # Only worth a Cerebras call when user gives meaningful content.
    if len(user_text) < 15:
        return base_update

    # Skip in S3/S4 if AT already captured — don't waste calls on confirmation
    phase = state.get("phase", "CHECKIN")
    if existing_at and phase in ("WORK", "WRAP"):
        return base_update

    # Trim long inputs (same reasoning as PHQ-9 supervisor)
    truncated = user_text if len(user_text) <= 280 else user_text[:280] + "..."

    try:
        prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        raw = await call_supervisor(prompt, truncated, _build_at_context(state), max_tokens=400)
    except Exception as e:
        print(f"  [SUP:AT] error: {e}")
        return base_update

    print(f"  [SUP:AT RAW]\n{raw}\n  [/SUP:AT RAW]")

    parsed = _parse_at_output(raw)
    new_at = parsed["extracted_at"].strip()
    new_dist_key = _normalize_distortion(parsed["distortion"])
    new_conf = parsed["confidence"]

    # Reject if not meaningful or low confidence
    if not _is_meaningful_at(new_at) or new_conf == "low":
        print(f"  [SUP:AT] LLM no AT — falling back to keyword detection")
        # Keyword-based last-resort fallback — catches clear ATs LLM missed
        kw_match = _keyword_extract_at(user_text)
        if kw_match:
            phrase, dist_key = kw_match
            label = _DISTORTION_LABELS.get(dist_key, dist_key)
            # Only use keyword fallback if no AT captured yet
            if not existing_at:
                print(f"  [SUP:AT] KEYWORD CAPTURED: '{phrase[:50]}' → {label}")
                return {
                    "automatic_thought": phrase,
                    "distortion": label,
                    "_at_confidence": "medium",
                }
        print(f"  [SUP:AT] no AT — at={new_at[:30]!r} conf={new_conf}")
        return base_update

    # α-fix: allow overwrite when NEW signal is materially stronger.
    # Original strict-greater rule blocked better ATs from overwriting weaker
    # first-captured ones (observed: 'ทำอะไรก็ไม่มีแรง' high stuck across turns,
    # rejecting 'ตัวเองโง่ไร้ค่า' which is a cleaner self-labeling AT).
    _SELF_LABEL_TOKENS = (
        "โง่", "ไร้ค่า", "ผิดพลาด", "ล้มเหลว",
        "ไม่ดีพอ", "ไร้ความสามารถ", "เป็นภาระ", "เกลียดตัวเอง",
    )
    if existing_at and _CONFIDENCE_RANK.get(new_conf, 0) <= _CONFIDENCE_RANK.get(existing_conf, 0):
        # Strong-signal bypass: NEW must still be high confidence AND either
        # 1.5× length of existing OR contain a clear self-label token.
        is_high = new_conf == "high"
        len_bypass = len(new_at) > 1.5 * max(len(existing_at), 1)
        token_bypass = any(t in new_at for t in _SELF_LABEL_TOKENS)
        if is_high and (len_bypass or token_bypass):
            reason = "stronger-length" if len_bypass else "self-label-token"
            print(f"  [SUP:AT] BYPASS sticky ({reason}): {existing_at!r} → {new_at!r}")
        else:
            print(f"  [SUP:AT] keep existing AT (new conf {new_conf} ≤ existing {existing_conf})")
            return base_update

    new_dist_label = _DISTORTION_LABELS.get(new_dist_key, new_dist_key)
    print(f"  [SUP:AT] CAPTURED: '{new_at[:50]}' → {new_dist_label} ({new_conf})")
    return {
        "automatic_thought": new_at,
        "distortion": new_dist_label,
        "_at_confidence": new_conf,
    }
