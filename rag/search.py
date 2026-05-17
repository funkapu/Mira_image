"""
search_technique: RAG search over cbt_manual_thai.json
Used by mira_graph/nodes/aggregate.py to retrieve matching CBT technique
given a belief_type + situation query.

Expected return schema (per aggregate.py usage):
    [
      {
        "score": float,
        "doc": {
          "situation_th": str,          # display label (user_input[:60])
          "recommended_techniques": [   # at least 1 entry
            {
              "technique": str,
              "question_th": str,
              "ref": str,
            }
          ],
          "avoid_techniques": [         # may be empty list
            {
              "technique": str,
              "reason": str,
            }
          ],
          "good_responses": [str],      # may be empty list
          "bad_responses": [str],       # may be empty list
        }
      },
      ...
    ]
"""

import json
import re
import numpy as np
from functools import lru_cache
from pathlib import Path

_CORPUS_PATH = Path(__file__).parent / "corpus.json"
_FAISS_PATH  = Path(__file__).parent / "corpus.faiss"

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_corpus():
    with open(_CORPUS_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


@lru_cache(maxsize=1)
def _load_index():
    import faiss
    return faiss.read_index(str(_FAISS_PATH))


# ---------------------------------------------------------------------------
# Schema adapter: cbt_manual_thai entry → aggregate.py expected doc shape
# ---------------------------------------------------------------------------

# Tag → human-readable technique label mapping
_TAG_TO_LABEL = {
    "all_or_nothing":          "All-or-Nothing Thinking challenge",
    "cognitive_restructuring": "Cognitive Restructuring",
    "evidence_examination":    "Evidence Examination",
    "catastrophizing":         "De-catastrophizing",
    "probability_estimation":  "Probability Estimation",
    "mind_reading":            "Perspective-Taking / Mind-reading challenge",
    "alternative_interpretation": "Alternative Interpretation",
    "fortune_telling":         "Fortune-telling challenge",
    "challenge_certainty":     "Challenge Certainty",
    "personalization":         "De-personalization",
    "responsibility_pie":      "Responsibility Pie",
    "should_statements":       "Should-statement challenge",
    "reframe_preference":      "Reframing",
    "socratic_questioning":    "Socratic Questioning",
    "validation":              "Validation",
    "behavioral_activation":   "Behavioral Activation",
}

# technique_inferred → default Socratic question in Thai
_TECHNIQUE_QUESTIONS = {
    "cbt":                    "ความคิดนั้น... มันจริงทั้งหมดไหมคะ หรือมีหลักฐานอีกมุม?",
    "reframing":              "ถ้าเพื่อนสนิทคิดแบบนี้กับตัวเอง น้องจะบอกเขาว่ายังไงคะ?",
    "socratic_questioning":   "มีครั้งไหนบ้างที่มันไม่เป็นแบบนั้น?",
    "validation":             "ฟังดูหนักมากเลยนะคะ วันนี้รู้สึกยังไงบ้างคะ?",
    "behavioral_activation":  "ลองนึกถึงกิจกรรมเล็กๆ ที่เคยทำแล้วรู้สึกดีขึ้นได้ไหมคะ?",
    "general":                "ความคิดนั้นมาบ่อยไหมคะ?",
}


def _entry_to_doc(entry: dict) -> dict:
    """Convert a cbt_manual_thai entry to the schema aggregate.py expects."""
    technique = entry.get("technique_inferred", "general")
    tags = entry.get("tags", [])

    # Build recommended_techniques from tags + technique_inferred
    rec_techniques = []
    for tag in tags:
        label = _TAG_TO_LABEL.get(tag)
        if label:
            rec_techniques.append({
                "technique": label,
                "question_th": _TECHNIQUE_QUESTIONS.get(technique,
                               _TECHNIQUE_QUESTIONS["general"]),
                "ref": entry.get("id", ""),
            })
    if not rec_techniques:
        rec_techniques.append({
            "technique": _TAG_TO_LABEL.get(technique, technique),
            "question_th": _TECHNIQUE_QUESTIONS.get(technique,
                           _TECHNIQUE_QUESTIONS["general"]),
            "ref": entry.get("id", ""),
        })

    # avoid_techniques: techniques NOT matching this entry's inferred one
    avoid = []
    if technique == "cbt":
        avoid.append({
            "technique": "Pure validation",
            "reason": "ช่วง WORK ต้องใช้ Socratic — validation ซ้ำจะ block progress",
        })
    elif technique == "validation":
        avoid.append({
            "technique": "Socratic challenge too early",
            "reason": "ยังอยู่ช่วง rapport/check-in — challenge ก่อนพร้อมทำให้ disengage",
        })

    counselor_resp = entry.get("counselor_response", "")
    lines = [l.strip() for l in counselor_resp.splitlines() if l.strip()]

    # Only include good_responses if they are Thai; English examples from the
    # English-dominant corpus (mentalchat16k_real, amod) would introduce English
    # text into the Thai system prompt and risk language drift.
    _thai_re = re.compile(r"[฀-๿]")
    thai_lines = [l for l in lines if _thai_re.search(l)]
    good = thai_lines[:2] if thai_lines else []

    return {
        "situation_th": entry.get("user_input", "")[:60],
        "recommended_techniques": rec_techniques,
        "avoid_techniques": avoid,
        "good_responses": good,
        "bad_responses": [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_technique(query: str, top_k: int = 1) -> list[dict]:
    """
    Search cbt_manual_thai corpus for the most relevant CBT technique.

    Args:
        query: Free-form text (e.g. "all_or_nothing ผมไม่ดีพอ").
        top_k: Number of results to return.

    Returns:
        List of dicts with keys: score (float), doc (dict per schema above).
        Empty list on any error.
    """
    try:
        encoder = _load_encoder()
        index   = _load_index()
        corpus  = _load_corpus()

        emb = encoder.encode([query])[0].astype("float32")
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        scores, indices = index.search(emb.reshape(1, -1), min(top_k * 4, len(corpus)))

        results = []
        seen_techniques = set()
        for score, idx in zip(scores[0], indices[0]):
            if len(results) >= top_k:
                break
            entry = corpus[int(idx)]
            tech = entry.get("technique_inferred", "general")
            if tech in seen_techniques:
                continue
            seen_techniques.add(tech)
            results.append({
                "score": float(score),
                "doc": _entry_to_doc(entry),
            })

        return results

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("[rag.search] search_technique error: %s", e)
        return []
