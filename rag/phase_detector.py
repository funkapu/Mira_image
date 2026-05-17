"""
CBT phase + topic detection from conversation state.
Used by smart retriever to filter relevant references.
"""

import re
from typing import Optional


def detect_topic(user_text: str) -> Optional[str]:
    """
    Detect topic from Thai/English text.
    Returns one of: work, stress, family, anxiety, relationships,
                    depression, sleep, trauma, or None.
    """
    text_lower = user_text.lower()

    topic_keywords = {
        'work': [
            'งาน', 'ทำงาน', 'บริษัท', 'หัวหน้า', 'เพื่อนร่วมงาน',
            'เลย์ออฟ', 'โปรเจกต์', 'meeting', 'deadline',
            'work', 'job', 'boss', 'project', 'nsc', 'แข่ง', 'ส่งงาน'
        ],
        'stress': [
            'เครียด', 'กดดัน', 'ทับทวม', 'ภาระ', 'หนัก',
            'stress', 'pressure', 'overwhelmed', 'burden'
        ],
        'family': [
            'พ่อ', 'แม่', 'พี่', 'น้อง', 'ครอบครัว', 'ลูก',
            'family', 'parent', 'mother', 'father', 'sibling'
        ],
        'anxiety': [
            'กังวล', 'วิตก', 'หวาดกลัว', 'panic', 'ใจสั่น',
            'anxiety', 'anxious', 'worry', 'fear', 'หายใจไม่ออก'
        ],
        'relationships': [
            'แฟน', 'เพื่อน', 'คน', 'รัก', 'เลิก', 'ทะเลาะ',
            'relationship', 'partner', 'friend', 'breakup', 'fight'
        ],
        'depression': [
            'ซึมเศร้า', 'หมดหวัง', 'ไร้ค่า', 'เศร้า', 'ไม่มีแรง',
            'depressed', 'hopeless', 'worthless', 'sad', 'numb'
        ],
        'sleep': [
            'นอน', 'หลับ', 'ไม่ได้นอน', 'ตื่น', 'ฝันร้าย',
            'sleep', 'insomnia', 'tired', 'exhausted'
        ],
        'trauma': [
            'ทำร้าย', 'ถูกข่ม', 'บาดเจ็บ', 'อันตราย',
            'trauma', 'abuse', 'assault', 'harm'
        ],
    }

    scores = {}
    for topic, keywords in topic_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score

    if not scores:
        return None

    return max(scores, key=scores.get)


# Maps heuristic_judge graph phases -> compose/retriever phases
_GRAPH_TO_COMPOSE_PHASE = {
    "CHECKIN": "rapport",
    "EXPLORE": "assessment",
    "WORK": "intervention",
    "WRAP": "rapport",
    "CRISIS": "crisis",
    "END": "rapport",
}


def map_graph_phase(graph_phase: str) -> str:
    """Map a heuristic_judge graph phase to a compose/retriever phase.
    
    Args:
        graph_phase: Phase from MiraState (e.g. "CHECKIN", "EXPLORE").
    
    Returns:
        Compose phase string ("rapport", "assessment", "intervention", "skills", "crisis").
        Falls back to "rapport" for unknown phases.
    """
    return _GRAPH_TO_COMPOSE_PHASE.get(graph_phase, "rapport")


def detect_phase(
    conversation_history: list,
    current_user_msg: str
) -> str:
    """
    Detect CBT phase based on conversation state + content.
    Returns: 'rapport', 'assessment', 'intervention', 'skills', or 'crisis'.

    Note: Crisis is also handled by crisis_detector.py before reaching here.
    This is defensive detection only.
    """
    user_turns = sum(1 for m in conversation_history if m.get('role') == 'user')
    msg_lower = current_user_msg.lower()

    crisis_kw = [
        'อยากตาย', 'ฆ่าตัวตาย', 'ไม่อยากตื่น',
        'อยากจบ', 'จบชีวิต', 'ไม่อยากอยู่'
    ]
    if any(kw in msg_lower for kw in crisis_kw):
        return 'crisis'

    if user_turns <= 1:
        return 'rapport'

    distortion_kw = [
        'ทุกคน', 'ไม่มีใคร', 'ไม่เคย', 'เสมอ', 'ทุกอย่าง',
        'พังหมด', 'จบแล้ว', 'ไม่มีทาง', 'ทำลายชีวิต',
        'เพราะผม', 'ผมทำให้', 'ผมผิด', 'เป็นความผิดผม',
        'ต้อง', 'ควรจะ', 'น่าจะ',
        'ไร้ค่า', 'ไร้ประโยชน์', 'ไม่ดีพอ', 'ห่วย', 'โง่',
    ]
    distortion_count = sum(1 for kw in distortion_kw if kw in msg_lower)

    if distortion_count >= 1 and user_turns >= 2:
        return 'intervention'

    skill_kw = [
        'ทำยังไง', 'จะ', 'ช่วย', 'แนะนำ', 'ควรทำ',
        'how do', 'what should', 'help me'
    ]
    if any(kw in msg_lower for kw in skill_kw) and user_turns >= 3:
        return 'skills'

    if user_turns <= 4:
        return 'assessment'

    return 'intervention'