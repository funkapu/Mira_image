#!/usr/bin/env python3
"""
Crisis Detection Layer - BEFORE RAG retrieval
Patient safety critical - must catch suicidal ideation
"""

import re
from typing import Dict, List

# Layer 1: Direct crisis keywords (highest priority)
CRISIS_KEYWORDS_THAI = [
    # Suicide/death wishes
    'อยากตาย', 'อยากฆ่าตัวตาย', 'ฆ่าตัวตาย', 'ตัดชีวิต',
    'จบชีวิต', 'ไม่อยากอยู่', 'ไม่อยากมีชีวิต',

    # Hopelessness idioms (Thai-specific)
    'ไม่อยากตื่น', 'ไม่อยากตื่นมา', 'ไม่อยากตื่นมาแล้ว',
    'ไม่อยากเจอวันพรุ่งนี้', 'หายไป', 'หายตัวไป',
    'จบ', 'จบลง', 'สิ้นสุด',

    # Self-harm
    'ทำร้ายตัวเอง', 'กรีดข้อมือ', 'กิน paracetamol',
    'กระโดด', 'แขวนคอ',

    # Despair
    'ไม่มีใครรัก', 'ไม่มีใครเข้าใจ', 'ไม่มีใครต้องการ',
    'ไม่มีความหมาย', 'ทุกอย่างไร้ค่า', 'หมดหวัง',

    # Crisis English variants (might mix)
    'kill myself', 'suicide', 'end it all', 'die', 'want to die'
]

# Layer 2: Severity scoring
CRITICAL_KEYWORDS = [
    'อยากตาย', 'ฆ่าตัวตาย', 'ไม่อยากตื่น', 'ไม่อยากตื่นมา',
    'ทำร้ายตัวเอง', 'จบชีวิต', 'กรีดข้อมือ', 'แขวนคอ',
    'kill myself', 'suicide', 'end it all'
]

HIGH_RISK = [
    'หายไป', 'หายตัวไป', 'จบ', 'จบลง', 'หมดหวัง',
    'ไม่ไหวแล้ว', 'ไม่อยากอยู่', 'ไม่มีใครรัก',
    'ไม่มีความหมาย', 'ทุกอย่างไร้ค่า', 'want to die'
]

MODERATE_RISK = [
    'ไม่มีใครเข้าใจ', 'ไม่มีใครต้องการ', 'สิ้นสุด',
    'ไม่อยากเจอวันพรุ่งนี้'
]


def detect_crisis(user_text: str) -> Dict:
    """
    Detect crisis/suicidal ideation in user input

    Returns:
    {
        "is_crisis": bool,
        "severity": "critical" | "high" | "moderate" | "none",
        "matched_keywords": [...],
        "action": "override_with_crisis_response" | "use_rag"
    }
    """
    text_lower = user_text.lower()
    matched = []
    severity = "none"

    # Check critical keywords (immediate danger)
    for kw in CRITICAL_KEYWORDS:
        if kw in text_lower:
            matched.append(kw)
            severity = "critical"

    # Check high risk (severe distress)
    if severity != "critical":
        for kw in HIGH_RISK:
            if kw in text_lower:
                matched.append(kw)
                severity = "high"

    # Check moderate risk (warning signs)
    if severity == "none":
        for kw in MODERATE_RISK:
            if kw in text_lower:
                matched.append(kw)
                severity = "moderate"

    # Check all crisis keywords (general)
    if severity == "none":
        for kw in CRISIS_KEYWORDS_THAI:
            if kw in text_lower:
                matched.append(kw)
                severity = "moderate"

    is_crisis = severity != "none"

    return {
        "is_crisis": is_crisis,
        "severity": severity,
        "matched_keywords": matched,
        "action": "override_with_crisis_response" if is_crisis else "use_rag"
    }


def test_crisis_detector():
    """Test crisis detection with various inputs"""

    test_cases = [
        # Critical
        ("ผมไม่อยากตื่นมาแล้ว", "critical"),
        ("ผมอยากตาย", "critical"),
        ("ผมคิดจะฆ่าตัวตาย", "critical"),
        ("ผมทำร้ายตัวเอง", "critical"),

        # High risk
        ("ผมรู้สึกหมดหวัง", "high"),
        ("ผมอยากหายไป", "high"),
        ("ผมไม่ไหวแล้ว", "high"),
        ("ไม่มีใครรักผม", "high"),

        # Moderate
        ("ไม่มีใครเข้าใจผม", "moderate"),
        ("ผมรู้สึกว่าทุกอย่างไร้ค่า", "moderate"),

        # Should NOT trigger (normal distress)
        ("ผมเครียดงาน NSC มาก", "none"),
        ("ผมเบื่อทุกอย่าง", "none"),
        ("ผมไม่ดีพอ", "none"),
        ("ผมเหนื่อย", "none"),
        ("ผมนอนไม่หลับ", "none"),
    ]

    print("=" * 60)
    print("Crisis Detector Test")
    print("=" * 60)

    passed = 0
    failed = 0

    for text, expected in test_cases:
        result = detect_crisis(text)
        status = "✓" if result['severity'] == expected else "✗"

        if result['severity'] == expected:
            passed += 1
        else:
            failed += 1

        print(f"\n{status} Input: '{text}'")
        print(f"  Expected: {expected}")
        print(f"  Got: {result['severity']}")

        if result['matched_keywords']:
            print(f"  Matched: {result['matched_keywords']}")

        if result['severity'] != expected:
            print(f"  ⚠️  MISMATCH!")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return passed, failed


if __name__ == "__main__":
    test_crisis_detector()
