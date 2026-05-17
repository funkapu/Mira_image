"""Output filter — strip artifacts, extract meta-tags, cap length for voice"""
import os
import re

from mira_graph.llm import clean_response
from mira_graph.state import MiraState

STAGE_DISABLED = os.environ.get("MIRA_STAGE_DISABLED", "0") == "1"

# Matches leaked tool args: trailing quote/paren/bracket combos
_LEAK_TAIL_RE = re.compile(r'[\"\'][\)\]]?\s*$')
_LEAK_HEAD_RE = re.compile(r'^\s*[\"\'\(]')
# Matches JSON-looking fragments — but NOT EMOTION/NEXT_STAGE tags (handled separately)
_JSON_FRAG_RE = re.compile(r'\{[^}]{0,80}\}')

# Meta tags emitted by the model (handled before _strip_leaks).
# Tolerant: accepts NEXT_STAGE / NEXTSTAGE / NEXT STAGE
_EMOTION_TAG_RE = re.compile(r"\[EMOTION:\s*([a-zA-Z]+)\s*\]\s*", re.IGNORECASE)
_STAGE_TAG_RE = re.compile(r"\[NEXT[_\s]?STAGE:\s*([A-Z0-9_\s]+?)\s*\]\s*", re.IGNORECASE)
_STAGE_NORMALIZE_RE = re.compile(r"^(S\d)([A-Z]+)$")
_VALID_STAGES = {"STAY", "CHECKIN", "EXPLORE", "WORK", "WRAP", "END"}


def _normalize_stage(raw: str) -> str:
    """Normalize stage names: handles legacy 'S2EXPLORE' → 'EXPLORE'."""
    raw = raw.upper().replace(" ", "")
    m = _STAGE_NORMALIZE_RE.match(raw)
    if m:
        raw = f"{m.group(1)}_{m.group(2)}"
    return raw

# Forbidden questions — the model is instructed never to ask these, but as a
# last line of defense we strip them from the spoken response and (if we have
# a captured automatic thought) replace with a reflection-style statement.
_FORBIDDEN_PATTERNS = [
    re.compile(r"(มีความคิด|ความคิด).{0,15}(ผ่าน|ใน)\s*หัว", re.IGNORECASE),
    re.compile(r"ตอนนั้น.{0,20}คิด(อะไร)?", re.IGNORECASE),
    re.compile(r"คิด(อะไร)?\s*อยู่", re.IGNORECASE),
    re.compile(r"คิดยังไง", re.IGNORECASE),
    re.compile(r"รู้สึกยังไง", re.IGNORECASE),
    re.compile(r"เป็นยังไงบ้าง", re.IGNORECASE),
]


def _strip_leaks(text: str) -> str:
    text = _JSON_FRAG_RE.sub("", text)
    text = _LEAK_TAIL_RE.sub("", text)
    text = _LEAK_HEAD_RE.sub("", text)
    return text.strip()


def _extract_meta_tags(text: str) -> tuple[str, str | None]:
    """Strip [EMOTION:] and [NEXT_STAGE:] from text. Return (cleaned, stage)."""
    text = _EMOTION_TAG_RE.sub("", text)

    stage: str | None = None
    m = _STAGE_TAG_RE.search(text)
    if m:
        raw = _normalize_stage(m.group(1))
        if raw in _VALID_STAGES:
            stage = raw
        else:
            stage = "STAY"
            print(f"  [STAGE] invalid value {m.group(1)!r} → fallback to STAY")
        text = _STAGE_TAG_RE.sub("", text, count=1)
    return text.strip(), stage


def _has_forbidden(sentence: str) -> bool:
    return any(p.search(sentence) for p in _FORBIDDEN_PATTERNS)


def _strip_forbidden_questions(text: str, state: MiraState) -> str:
    """Remove sentences containing forbidden 'what were you thinking' questions."""
    if not text:
        return text
    # Split on common Thai/English sentence enders, keeping splits non-greedy
    parts = re.split(r"(?<=[?？!])\s+|(?<=ค่ะ)\s+|(?<=นะคะ)\s+|(?<=ครับ)\s+", text)
    kept = [s for s in parts if s.strip() and not _has_forbidden(s)]

    if len(kept) < len(parts):
        captured_at = state.get("automatic_thought")
        if captured_at and not any("คิดว่า" in s for s in kept):
            kept.append(f"ฟังดูเหมือนตอนนั้นคุณกำลังคิดว่า '{captured_at}' ใช่ไหมคะ?")
        elif not kept:
            kept.append("พี่อยู่ตรงนี้รับฟังนะคะ")
        print(f"  [FILTER ⚠️] forbidden Q removed: {len(parts)} → {len(kept)} sentences")
    return " ".join(s.strip() for s in kept).strip()


async def output_filter_node(state: MiraState) -> dict:
    raw = state.get("mira_response") or ""
    cleaned, stage = _extract_meta_tags(raw)

    response = clean_response(cleaned)
    response = _strip_leaks(response)
    response = _strip_forbidden_questions(response, state)

    # Cap at 2 sentences for voice
    if len(response) > 300:
        for sep in ("นะคะ", "ค่ะ", "ครับ"):
            idx = response.find(sep)
            if 0 < idx < len(response) - 1:
                second = response.find(sep, idx + 1)
                if second != -1:
                    response = response[: second + len(sep)].strip()
                    break

    if not response:
        response = "อืม เล่าต่อได้เลยนะคะ"

    out: dict = {"mira_response": response}
    if stage and not STAGE_DISABLED:
        out["_llm_phase_suggestion"] = stage
        print(f"  [STAGE 🤖] LLM suggested: {stage}")
    elif stage and STAGE_DISABLED:
        print(f"  [STAGE 🚫] disabled — ignored LLM suggestion: {stage}")
    return out
