"""
RAG Integration Wrapper for Mira Agent v5

This module provides a drop-in replacement for call_ultravox_pure_stream
that adds RAG compose pipeline with crisis detection.

Usage in mira_agent_v5_cascading.py:

    # Add at top:
    import sys
    sys.path.insert(0, '/app')
    from rag_integration import call_with_rag_compose

    # In _handle_therapy(), replace line 672:
    # stream = call_ultravox_pure_stream(system_prompt, text, history)
    # with:
    stream = call_with_rag_compose(system_prompt, text, history)

Feature flag: MIRA_RAG_COMPOSE=1 to enable (default: 0)
"""

import os
import sys
import logging
from typing import AsyncIterator

# Add workspace to path for RAG modules
sys.path.insert(0, '/app')

# Import RAG compose
try:
    from rag.compose import get_mira_response as get_rag_response
    _RAG_AVAILABLE = True
except ImportError as e:
    _RAG_AVAILABLE = False
    logging.warning(f"RAG compose not available: {e}")

# Import original LLM call
from mira_graph.llm import call_ultravox_pure_stream

# Feature flag
USE_RAG = os.environ.get("MIRA_RAG_COMPOSE", "0") == "1" and _RAG_AVAILABLE

logger = logging.getLogger(__name__)


async def call_with_rag_compose(
    system_prompt: str,
    user_text: str,
    history: list,
    graph_phase: str | None = None,
    audio_b64: str | None = None,
) -> AsyncIterator[str]:
    """
    RAG-enhanced LLM call with crisis detection.

    Flow:
    1. Check if RAG enabled (MIRA_RAG_COMPOSE=1)
    2. If enabled:
       a. Crisis detection → hardcoded response (bypass LLM)
       b. RAG retrieval → enhanced prompt → LLM
    3. If disabled: fallback to original call_ultravox_pure_stream

    Args:
        system_prompt: Original system prompt (may be replaced by RAG)
        user_text: User's input text
        history: Conversation history
        graph_phase: Phase from heuristic_judge graph state (e.g. "CHECKIN", "EXPLORE").
                     If None, compose falls back to stateless detect_phase().
        audio_b64: Base64-encoded audio for Ultravox audio tower. When present,
                   the audio stream path is used instead of pure text.

    Yields:
        Sentences from LLM response (streaming)
    """

    if not USE_RAG:
        logger.debug("[RAG] Disabled - using legacy path")
        if audio_b64:
            from mira_graph.llm import call_ultravox_audio_stream
            async for sentence in call_ultravox_audio_stream(system_prompt, audio_b64, history):
                yield sentence
        else:
            async for sentence in call_ultravox_pure_stream(system_prompt, user_text, history):
                yield sentence
        return

    # RAG path
    logger.info(f"[RAG] Enabled - processing: {user_text[:50]}...")

    try:
        compose_phase = None
        if graph_phase is not None:
            from rag.phase_detector import map_graph_phase
            compose_phase = map_graph_phase(graph_phase)
            logger.info("[RAG] graph_phase=%s -> compose_phase=%s", graph_phase, compose_phase)

        result = get_rag_response(user_text, history=history, phase=compose_phase)

        if result['source'] == 'crisis':
            logger.warning(f"[RAG] Crisis detected: severity={result.get('crisis_severity') or result.get('severity')}, keywords={result.get('matched_keywords', [])}")

            yield result['response']

        else:
            logger.info(f"[RAG] Retrieved {len(result['references_used'])} references")
            for ref in result['references_used']:
                logger.debug(f"[RAG]   - score={ref['score']:.3f}, topics={ref['topics'][:2]}")

            # Cache-friendly path: stable system prompt + dynamic user prefix.
            # The stable_system_prompt is a pure function of phase, so it's
            # identical across every turn within the same phase. vLLM's prefix
            # cache reuses the whole prefix instead of re-prefilling 2000 tokens.
            stable_system = result.get('stable_system_prompt')
            dynamic_prefix = result.get('dynamic_user_prefix', '')

            if stable_system:
                # Hash debug — verify stability across turns of same phase.
                import hashlib as _hashlib
                prefix_hash = _hashlib.md5(stable_system[:2000].encode('utf-8')).hexdigest()[:8]
                logger.info(f"[CACHE-DEBUG] system_prompt prefix hash: {prefix_hash} (phase={result.get('phase')}, len={len(stable_system)})")

                # Prepend dynamic context to user message; keep system stable.
                user_text_with_ctx = (
                    (dynamic_prefix + user_text) if dynamic_prefix else user_text
                )
                if audio_b64:
                    from mira_graph.llm import call_ultravox_audio_stream
                    async for sentence in call_ultravox_audio_stream(stable_system, audio_b64, history, user_prefix=dynamic_prefix):
                        yield sentence
                else:
                    async for sentence in call_ultravox_pure_stream(stable_system, user_text_with_ctx, history):
                        yield sentence
            else:
                # Backward compat: fall back to monolithic reasoning_prompt.
                rag_prompt = result['reasoning_prompt']
                logger.debug(f"[RAG] Prompt length: {len(rag_prompt)} chars")
                if audio_b64:
                    from mira_graph.llm import call_ultravox_audio_stream
                    async for sentence in call_ultravox_audio_stream(rag_prompt, audio_b64, history):
                        yield sentence
                else:
                    async for sentence in call_ultravox_pure_stream(rag_prompt, user_text, history):
                        yield sentence

    except Exception as e:
        # Fallback to legacy on error
        logger.error(f"[RAG] Error in RAG pipeline: {e}", exc_info=True)
        logger.warning("[RAG] Falling back to legacy path")

        if audio_b64:
            from mira_graph.llm import call_ultravox_audio_stream
            async for sentence in call_ultravox_audio_stream(system_prompt, audio_b64, history):
                yield sentence
        else:
            async for sentence in call_ultravox_pure_stream(system_prompt, user_text, history):
                yield sentence


# Convenience function for testing
def test_rag_integration():
    """Test RAG integration without full agent"""
    print("=" * 60)
    print("RAG Integration Test")
    print("=" * 60)
    print(f"RAG Available: {_RAG_AVAILABLE}")
    print(f"USE_RAG: {USE_RAG}")
    print(f"MIRA_RAG_COMPOSE env: {os.environ.get('MIRA_RAG_COMPOSE', 'not set')}")

    if _RAG_AVAILABLE:
        # Test crisis detection
        test_queries = [
            "ผมเครียดงาน NSC มาก",
            "ผมไม่อยากตื่นมาแล้ว",
        ]

        for query in test_queries:
            print(f"\nTest: {query}")
            result = get_rag_response(query)
            print(f"  Source: {result['source']}")
            if result['source'] == 'crisis':
                print(f"  Severity: {result['crisis_severity']}")
                print(f"  Response: {result['response'][:100]}...")
            else:
                print(f"  References: {len(result['references_used'])}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_rag_integration()
