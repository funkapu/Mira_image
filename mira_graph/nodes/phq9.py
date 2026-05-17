"""PHQ-9 Symptom Detection Supervisor.

Single-purpose Cerebras agent that scans the latest user_text for PHQ-9
depression signals (Q1-Q8 only — Q9 is handled by the Crisis supervisor).
Outputs a structured directive that the voice agent injects into Mira's
system prompt as a hard instruction, overriding the LoRA's default
advice-giving behavior.

Flow:
  user_text → call_supervisor(sup_phq9.txt) → parse → directive string
                                                     → state[phq9_directive]
"""
from __future__ import annotations

import re
from pathlib import Path

from mira_graph.llm import call_supervisor
from mira_graph.state import MiraState

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "sup_phq9.txt"

# Phases where probing is forbidden (CBT reframe/wrap focus on cognitive work).
_NO_PROBE_PHASES = ("WORK", "WRAP")

# Cap on total PHQ-9 probes per session. After this, supervisor stops firing
# and Mira returns to natural CBT flow (no more directive injection).
_MAX_PROBES_PER_SESSION = 2

# Trigger keywords. PHQ-9 supervisor only fires when user_text contains one of
# these — saves Cerebras quota and keeps Mira's CBT flow uninterrupted on
# turns where user isn't reporting symptoms.
_SYMPTOM_TRIGGERS = (
    # Q1 anhedonia
    "เบื่อ", "เซ็ง", "ไม่อยากทำ", "ไม่สนุก",
    # Q2 mood
    "เศร้า", "หดหู่", "สิ้นหวัง", "ไม่มีความสุข", "ใจหาย",
    # Q3 sleep
    "นอนไม่หลับ", "นอนเยอะ", "ตื่นกลางดึก", "หลับยาก", "นอนไม่พอ",
    # Q4 fatigue
    "เหนื่อย", "หมดแรง", "หมดพลัง", "เพลีย", "ไม่มีแรง", "อ่อนล้า",
    # Q5 appetite
    "ไม่อยากกิน", "กินเยอะ", "น้ำหนักลด", "น้ำหนักขึ้น",
    # Q6 self-worth
    "ไร้ค่า", "ผิดหวังในตัวเอง", "รู้สึกผิด", "ตัวเองไม่ดี",
    # Q7 concentration
    "สมาธิ", "คิดไม่ออก", "ตัดสินใจไม่ได้", "หลงๆลืมๆ",
    # Q8 psychomotor
    "ช้าลง", "กระสับกระส่าย", "อยู่นิ่งไม่ได้", "ขยับช้า",
)


def _has_symptom_keyword(text: str) -> bool:
    return any(kw in text for kw in _SYMPTOM_TRIGGERS)

# Human-readable Thai labels for each symptom (used in the directive Mira reads).
_SYMPTOM_LABELS = {
    "Q1_anhedonia": "ความรู้สึกเบื่อหน่าย/ไม่อยากทำอะไร",
    "Q2_mood": "อารมณ์เศร้า/หดหู่",
    "Q3_sleep": "ปัญหาการนอน",
    "Q4_fatigue": "ความเหนื่อย/ไม่มีแรง",
    "Q5_appetite": "การกิน/ความอยากอาหาร",
    "Q6_self_worth": "ความรู้สึกต่อตัวเอง",
    "Q7_concentration": "สมาธิ/การคิด",
    "Q8_psychomotor": "การเคลื่อนไหว/กระสับกระส่าย",
}

# Probe templates — Mira will phrase her own variation around these intents.
_PROBE_TEMPLATES = {
    "duration": "เป็นมานานหรือยังคะ?",
    "frequency": "บ่อยไหมคะ? เกือบทุกวันเลยเหรอคะ?",
    "impact": "กระทบกับชีวิตประจำวันบ้างไหมคะ?",
}


def _build_phq9_context(state: MiraState) -> str:
    """Tell the LLM what Mira already probed so it doesn't repeat."""
    lines = [f"Phase: {state.get('phase', 'CHECKIN')}"]
    asked = state.get("phq9_probes_asked") or []
    if asked:
        lines.append(f"Already probed: {', '.join(asked)}")
    else:
        lines.append("Already probed: (none yet)")

    # Recent context for symptom continuity
    msgs = state.get("messages", [])[-4:]
    user_lines = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            user_lines.append((m.get("content") or "")[:80])
    if user_lines:
        lines.append("Recent user messages:")
        for u in user_lines:
            lines.append(f"  - {u}")
    return "\n".join(lines)


def _parse_supervisor_output(raw: str) -> dict:
    """Extract the 5 fields from the supervisor's output.

    Tolerates: JSON, markdown (`**field**:`), case variants, leading bullet/dash,
    quoted values, "Q4" without "_fatigue" suffix.
    """
    out = {
        "detected": [],
        "chosen_symptom": "none",
        "chosen_probe": "none",
        "already_provided": [],
        "reasoning": "",
    }

    # Try JSON first if response looks like JSON
    s = raw.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            import json as _json
            data = _json.loads(s)
            if isinstance(data, dict):
                # Lowercase keys
                data = {str(k).lower(): v for k, v in data.items()}
                for key in out:
                    if key in data:
                        v = data[key]
                        if key in ("detected", "already_provided"):
                            if isinstance(v, list):
                                out[key] = [str(x) for x in v if x and str(x).lower() != "none"]
                            elif isinstance(v, str):
                                out[key] = _split_list(v)
                        else:
                            out[key] = str(v).strip()
                # Normalize symptom/probe matching
                out["chosen_symptom"] = _normalize_symptom(out["chosen_symptom"])
                out["chosen_probe"] = out["chosen_probe"].lower()
                return out
        except Exception:
            pass  # fall through to line-based parser

    # Line-based parser — tolerates markdown, bullets, etc.
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        # Strip leading bullets/dashes/numbering
        line = re.sub(r"^[\-*•·>\s]+", "", line)
        # Strip markdown bold/italic
        line = re.sub(r"\*+", "", line).strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        val = val.strip().strip('"').strip("'").strip(",").strip()
        if key == "detected":
            out["detected"] = _split_list(val)
        elif key in ("chosen_symptom", "symptom"):
            out["chosen_symptom"] = _normalize_symptom(val)
        elif key in ("chosen_probe", "probe"):
            out["chosen_probe"] = val.lower()
        elif key == "already_provided":
            out["already_provided"] = _split_list(val)
        elif key == "reasoning":
            out["reasoning"] = val
    return out


_SYMPTOM_KEYS = {
    "Q1_anhedonia", "Q2_mood", "Q3_sleep", "Q4_fatigue",
    "Q5_appetite", "Q6_self_worth", "Q7_concentration", "Q8_psychomotor",
}
_QSHORT_TO_FULL = {k.split("_")[0]: k for k in _SYMPTOM_KEYS}


def _normalize_symptom(val: str) -> str:
    """Map 'Q4', 'q4_fatigue', 'Q4 fatigue', 'Q4_FATIGUE' → 'Q4_fatigue'.
    Returns 'none' if unrecognizable."""
    if not val:
        return "none"
    v = val.strip().strip('"').strip("'").lower()
    if v in ("none", "[]", "null"):
        return "none"
    # Direct match
    for full in _SYMPTOM_KEYS:
        if v == full.lower():
            return full
    # "Q4" → look up Q4_fatigue
    short = re.match(r"^(q[1-9])\b", v)
    if short:
        return _QSHORT_TO_FULL.get(short.group(1).upper(), "none")
    return "none"


_LIST_BRACKETS = re.compile(r"^\[(.*)\]$")


def _split_list(val: str) -> list[str]:
    """Parse '[Q1_anhedonia, Q4_fatigue]' or 'Q1_anhedonia, Q4_fatigue' or 'none' into list."""
    val = val.strip()
    m = _LIST_BRACKETS.match(val)
    if m:
        val = m.group(1)
    if not val or val.lower() == "none":
        return []
    return [v.strip() for v in val.split(",") if v.strip() and v.strip().lower() != "none"]


def _build_directive(symptom: str, probe: str) -> str:
    """Soft hint appended to Mira's system_prompt — gives Mira a probe suggestion
    without overriding the natural CBT flow. Mira can use it or ignore it."""
    label = _SYMPTOM_LABELS.get(symptom, symptom)
    template = _PROBE_TEMPLATES.get(probe, "เล่าเพิ่มได้ไหมคะ?")
    return (
        "\n\n=== PHQ-9 hint (ทางเลือก) ===\n"
        f"ผู้ใช้แสดงสัญญาณ: {label}\n"
        f"ถ้าจังหวะเหมาะ ลองถาม: \"{template}\"\n"
        f"ตอบ empathy 1 ประโยคก่อน → แล้วตามด้วยคำถามนี้หรือคำถาม CBT อื่นที่เหมาะสม\n"
    )


async def phq9_supervisor(state: MiraState) -> dict:
    """Detect PHQ-9 symptom in user_text and emit a soft probe hint.

    Trigger-based: only fires when user_text contains a symptom keyword.
    Otherwise returns no directive → Mira does natural CBT.
    """
    # NSC demo: PHQ-9 disabled to keep conversation natural.
    # Crisis detection still happens via _detect_crisis (text-based regex).
    return {
        "phq9_directive": None,
        "phq9_detected_symptom": None,
        "phq9_chosen_probe": None,
    }
    # ↓ Original code below (unreachable, kept for easy revert)
    phase = state.get("phase", "CHECKIN")
    user_text = (state.get("user_text") or "").strip()
    asked = list(state.get("phq9_probes_asked") or [])

    base_update: dict = {
        "phq9_directive": None,
        "phq9_detected_symptom": None,
        "phq9_chosen_probe": None,
        "phq9_probes_asked": asked,
    }

    # Hard guards — never call the LLM if obviously not applicable
    if phase in _NO_PROBE_PHASES:
        return base_update
    if len(user_text) < 5:
        return base_update

    # KEYWORD GATE — supervisor only fires when user mentions a symptom.
    # Plain conversation/greeting/event description: skip entirely, Mira goes
    # natural CBT without any directive.
    if not _has_symptom_keyword(user_text):
        return base_update

    # PROBE CAP — after N probes, stop. No more directive injection;
    # Mira returns to normal CBT flow.
    if len(asked) >= _MAX_PROBES_PER_SESSION:
        print(f"  [SUP:PHQ9] cap reached ({len(asked)}/{_MAX_PROBES_PER_SESSION}) — releasing")
        return base_update

    # Trim very long inputs — supervisor only needs ~250 chars to detect symptoms.
    # Long Thai rants burn token budget and can leave content empty after thinking.
    truncated_text = user_text if len(user_text) <= 280 else user_text[:280] + "..."

    # Call the LLM
    try:
        prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        raw = await call_supervisor(prompt, truncated_text, _build_phq9_context(state), max_tokens=600)
    except Exception as e:
        print(f"  [SUP:PHQ9] error: {e}")
        return base_update

    # DEBUG: log raw output so we can see if format matches our parser
    print(f"  [SUP:PHQ9 RAW]\n{raw}\n  [/SUP:PHQ9 RAW]")

    parsed = _parse_supervisor_output(raw)
    chosen_symptom = parsed["chosen_symptom"]
    chosen_probe = parsed["chosen_probe"]

    # Sanity: drop if no symptom or no probe
    if (
        chosen_symptom in ("none", "", None)
        or chosen_probe in ("none", "", None)
        or chosen_symptom not in _SYMPTOM_LABELS
        or chosen_probe not in _PROBE_TEMPLATES
    ):
        print(f"  [SUP:PHQ9] no probe — sym={chosen_symptom} probe={chosen_probe} reason={parsed['reasoning'][:50]}")
        return base_update

    probe_id = f"{chosen_symptom}_{chosen_probe}"

    # Already-asked guard (defense in depth — supervisor should already filter)
    if probe_id in asked:
        print(f"  [SUP:PHQ9] {probe_id} already asked — skipping")
        return base_update

    asked.append(probe_id)
    directive = _build_directive(chosen_symptom, chosen_probe)
    print(f"  [SUP:PHQ9] {chosen_symptom} → probe={chosen_probe} | reason={parsed['reasoning'][:60]}")

    return {
        "phq9_directive": directive,
        "phq9_detected_symptom": chosen_symptom,
        "phq9_chosen_probe": probe_id,
        "phq9_probes_asked": asked,
    }
