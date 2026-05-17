"""agent.py — RunPod entrypoint shim for the LiveKit voice agent.

The actual agent logic lives in voice/mira_agent_v*.py; this thin wrapper
exists so the Docker CMD (`python agent.py`) matches the RunPod deployment
template and lets us swap agent versions without touching Docker / scripts.

Version dispatch (set in the Pod template or start_agent.sh):
    MIRA_AGENT_VERSION=v5   # default — cascading streaming, latency-first
    MIRA_AGENT_VERSION=v4   # fallback — pre-cascading, kept for regression

Run modes (passed through to livekit-agents CLI):
    python agent.py start      # production worker — connects to LiveKit Cloud
    python agent.py dev        # local hot-reload session
    python agent.py download-files
"""
import logging
import os
import signal
import traceback

from livekit.agents import WorkerOptions, cli

_VERSION = os.environ.get("MIRA_AGENT_VERSION", "v5").lower()

if _VERSION == "v4":
    from voice.mira_agent_v4 import _prewarm, entrypoint
elif _VERSION == "v5":
    from voice.mira_agent_v5_cascading import _prewarm, entrypoint
else:
    logging.getLogger("agent").warning(
        "Unknown MIRA_AGENT_VERSION=%r — falling back to v5", _VERSION,
    )
    from voice.mira_agent_v5_cascading import _prewarm, entrypoint


def _sigterm_trace(signum, frame):
    logging.getLogger("agent").critical(
        "SIGTERM received — stack trace:\n%s",
        "".join(traceback.format_stack(frame)),
    )
    # Re-raise so livekit-agents' own handler fires next
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.raise_signal(signal.SIGTERM)


signal.signal(signal.SIGTERM, _sigterm_trace)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=_prewarm, num_idle_processes=1, job_memory_warn_mb=5000, initialize_process_timeout=60.0))
