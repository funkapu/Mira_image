"""
Emotion detection for Mira's TTS routing.

Uses XLM-EMO-T (MilaNLProc, ACL 2022) for multilingual
emotion classification with crisis-keyword override.
"""

import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Lazy-load — only import torch/transformers when needed
_pipe = None
_pipe_load_failed = False
_MODEL_NAME = "MilaNLProc/xlm-emo-t"


def _get_pipe():
    """Lazy load emotion classifier (singleton, fail-safe)."""
    global _pipe, _pipe_load_failed
    
    if _pipe is not None:
        return _pipe
    if _pipe_load_failed:
        return None  # already failed, don't retry
    
    try:
        import torch
        from transformers import pipeline
        
        logger.info(f"Loading emotion classifier: {_MODEL_NAME}")
        _pipe = pipeline(
            "text-classification",
            model=_MODEL_NAME,
            device=0 if torch.cuda.is_available() else -1,
        )
        logger.info("Emotion classifier loaded ✓")
        return _pipe
    except Exception as e:
        logger.error(f"Failed to load emotion classifier: {e}")
        _pipe_load_failed = True
        return None


# XLM-EMO-T labels → Mira's emotion system
# Mira emotion options (must match TTS): sad, calm, neutral, happy, fearful
EMOTION_MAP = {
    "joy":     "happy",
    "sadness": "sad",
    "anger":   "sad",      # anger treated as distress
    "fear":    "fearful",
}

# Crisis keywords — ALWAYS force 'sad' (safety critical)
CRISIS_KEYWORDS = [
    "อยากตาย", "ฆ่าตัวตาย", "ไม่อยากอยู่", "ทรมาน",
    "หมดหวัง", "กระโดด", "ตัดข้อมือ", "จะตาย",
    "ฆ่าตัว", "ไม่อยากตื่น", "อยากหายไป",
]

# Mira's context indicators
MIRA_CRISIS_INDICATORS = ["1323", "สายด่วน", "เพื่อนรักผู้พิทักษ์"]
MIRA_CALM_INDICATORS = ["ยินดี", "ขอบคุณ", "ไม่เป็นไร"]

CONFIDENCE_THRESHOLD = 0.50


def detect_emotion_from_user(user_text: str) -> str:
    """
    Detect emotion from user's input (primary signal).
    
    Returns: one of {sad, calm, neutral, happy, fearful}
    """
    if not user_text or not user_text.strip():
        return "calm"
    
    text_lower = user_text.lower()
    
    # Crisis override (safety critical)
    for kw in CRISIS_KEYWORDS:
        if kw in text_lower:
            return "sad"
    
    # ML classifier
    pipe = _get_pipe()
    if pipe is not None:
        try:
            result = pipe(user_text, truncation=True, max_length=256)[0]
            label = result["label"].lower()
            score = result["score"]
            
            if score >= CONFIDENCE_THRESHOLD:
                return EMOTION_MAP.get(label, "calm")
        except Exception as e:
            logger.warning(f"Emotion classification failed: {e}")
    
    return "calm"


def detect_emotion_smart(
    user_text: str,
    mira_text: str = "",
) -> str:
    """
    Smart emotion detection using user input + Mira context.
    
    Logic:
      1. Mira mentions 1323/hotline → 'sad' (crisis context preserved)
      2. Use user's emotion as primary signal
      3. Override to 'calm' if Mira purely acknowledges + user not sad/fearful
    
    Returns: emotion label for TTS
    """
    # Mira mentions crisis hotline → preserve sad
    if mira_text:
        for kw in MIRA_CRISIS_INDICATORS:
            if kw in mira_text:
                return "sad"
    
    # User emotion (primary)
    user_emotion = detect_emotion_from_user(user_text)
    
    # Mira pure acknowledgment + user not distressed → calm
    if mira_text and user_emotion not in ("sad", "fearful"):
        mira_lower = mira_text.lower()
        is_acknowledgment = any(kw in mira_lower for kw in MIRA_CALM_INDICATORS)
        if is_acknowledgment:
            return "calm"
    
    return user_emotion


# Public API
__all__ = [
    "detect_emotion_from_user",
    "detect_emotion_smart",
    "EMOTION_MAP",
    "CRISIS_KEYWORDS",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_pairs = [
        ("สวัสดีครับ",            "สวัสดีค่ะ น้อง", "happy or calm"),
        ("วันนี้เหนื่อยมาก",      "(sighs) เหนื่อยขนาดนี้", "sad"),
        ("ผมไม่เก่งเลย",          "รู้สึกว่าตัวเอง 'ไม่เก่ง'", "sad"),
        ("ไม่รู้ครับ",            "พี่เข้าใจนะคะ", "calm"),
        ("อยากตายแล้ว",          "นี่หนักมาก ลองโทร 1323", "sad"),
        ("ผมจะกระโดดจริงๆ",      "น้อง โทร 1323 ตอนนี้", "sad"),
        ("ขอบคุณค่ะ พี่",         "ยินดีค่ะ น้อง", "calm"),
        ("ดีใจมาก",              "(chuckle) ดีใจที่ได้ยินนะ", "happy"),
        ("กลัวจะสอบไม่ผ่าน",      "ความกลัวสอบ", "fearful"),
    ]
    
    print(f"{'USER':<25} → {'MIRA':<35} → EMOTION (expected)")
    print("=" * 90)
    
    for user, mira, expected in test_pairs:
        em = detect_emotion_smart(user, mira)
        print(f"{user[:23]:<25} → {mira[:33]:<35} → {em:<8} ({expected})")
