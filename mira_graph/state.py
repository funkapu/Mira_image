from __future__ import annotations

import operator
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class MiraState(TypedDict):
    user_text: str | None
    user_audio_b64: str | None  # Phase 2
    messages: Annotated[list[BaseMessage], add_messages]
    phase: Literal["CHECKIN", "EXPLORE", "WORK", "WRAP", "CRISIS", "END"]
    phase_turn_count: int
    total_turns: int
    parallel_results: Annotated[list, operator.add]
    crisis_detected: bool
    mood_score: int | None
    sentiment: str | None
    topic: str | None
    user_name: str | None
    main_concern: str | None
    automatic_thought: str | None
    distortion: str | None
    mira_response: str | None
    should_transition: bool
    next_phase: str | None
    _happy_path: bool
    _voice_mode: bool
    # Supervisor advice (set in parallel by 5 supervisor agents)
    supervisor_empathy: str | None
    supervisor_belief: str | None
    supervisor_reflection: str | None
    supervisor_strategy: str | None
    supervisor_encouragement: str | None
    # Phase 3.1: quick acknowledgment sent immediately before full graph completes
    quick_response: str | None
    # Last N Mira responses — injected into prompt to prevent loops
    asked_questions: list | None
    # Tool call results — set by phase nodes, consumed + reset by judge
    _transition_requested: bool
    _transition_reason: str | None
    _crisis_severity: str | None
    _crisis_keyword: str | None
    # PHQ-9 supervisor outputs (set by phq9_supervisor, consumed by counsellor/voice agent)
    phq9_directive: str | None         # the inject-into-prompt block, or None
    phq9_detected_symptom: str | None  # e.g. "Q4_fatigue"
    phq9_chosen_probe: str | None      # e.g. "Q4_duration"
    phq9_probes_asked: list | None     # sticky across turns: ["Q3_duration", "Q4_frequency"]
    # AT supervisor confidence — used to decide if a new AT should overwrite existing
    _at_confidence: str | None         # "high" | "medium" | "low"
    # Counter: how many turns we've already asked for AT after PHQ-9 cap.
    # Capped to avoid frustrating users who can't articulate cognitions.
    _at_asking_count: int
    # ── Covert Intake (PHQ-9 wrapped in small-talk) ────────────────────────────
    session_type: str | None          # "intake" or "therapy"
    intake_started: bool              # has Mira sent the first greeting yet?
    intake_pending_score: bool        # last Mira turn was a PHQ-9 Q → score user reply
    intake_smalltalk_count: int       # rotates small-talk topic
    phq9_step: int                    # 0..9: number of PHQ-9 Qs asked
    phq9_scores: dict                 # {"Q1": 0..3, ...}
    phq9_total: int | None            # 0..27 after Q9
    phq9_q9_score: int | None         # for Q9 crisis trigger
