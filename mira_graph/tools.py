"""Tool definitions for Mira's autonomous decision-making"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "track_mood",
            "description": (
                "บันทึก mood score ของ user ใน turn นี้ "
                "เรียกเมื่อ user แสดงอารมณ์ชัดเจน (เช่น บอก 1-10 หรือบอกความรู้สึกชัด)"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mood": {
                        "type": "integer",
                        "description": "1-10 (1=worst, 10=best)",
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "label": {
                        "type": "string",
                        "description": "คำสั้น ๆ อธิบาย mood เช่น 'เครียด', 'ดี'",
                    },
                },
                "required": ["mood"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_concern",
            "description": (
                "บันทึกเรื่องหลักที่ user กังวล "
                "เรียกเมื่อ user เปิดเผยปัญหาหลัก (เช่น 'เครียดเรื่องงาน')"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concern": {
                        "type": "string",
                        "description": "ปัญหา/เรื่องที่ user กังวล",
                    },
                    "topic": {
                        "type": "string",
                        "enum": ["work", "family", "relationship", "health", "study", "finance", "self", "other"],
                    },
                },
                "required": ["concern", "topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_automatic_thought",
            "description": (
                "บันทึก automatic thought ของ user "
                "เรียกเมื่อจับ AT ได้ (เช่น 'ตัวเองห่วยมาก', 'ทำอะไรก็ไม่ดี')"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "automatic thought ที่ user แสดง",
                    },
                    "distortion": {
                        "type": "string",
                        "enum": [
                            "all_or_nothing", "overgeneralization", "mental_filter",
                            "disqualifying_positive", "jumping_to_conclusions",
                            "magnification", "emotional_reasoning", "should_statements",
                            "labeling", "personalization", "unknown",
                        ],
                        "description": "Burns' cognitive distortion type",
                    },
                },
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transition_phase",
            "description": (
                "เปลี่ยนไป CBT phase ถัดไป "
                "เรียกเมื่อ user แสดงเกณฑ์ครบของ phase ปัจจุบัน:\n"
                "- S1: user บอกชื่อ + mood\n"
                "- S2: user เล่าเหตุการณ์ + AT + emotion\n"
                "- S3: user เริ่มเห็น distortion / มอง angle ใหม่\n"
                "- S4: user รับข้อสรุป พร้อมจบ"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "เหตุผลสั้น ๆ ว่าทำไมพร้อม transition",
                    },
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_crisis_protocol",
            "description": (
                "เรียกใช้เมื่อตรวจพบความเสี่ยง self-harm หรือ suicide ideation "
                "(เช่น 'อยากตาย', 'ไม่อยากอยู่')"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "indicators": {
                        "type": "string",
                        "description": "คำหรือ context ที่บ่งบอก",
                    },
                },
                "required": ["severity"],
            },
        },
    },
]


def execute_tool(tool_name: str, args: dict, state: dict) -> dict:
    """Execute a tool call — return state updates"""

    if tool_name == "track_mood":
        mood = args.get("mood")
        label = args.get("label", "")
        print(f"    [TOOL] track_mood({mood}, '{label}')")
        return {"mood_score": mood}

    elif tool_name == "track_concern":
        concern = args.get("concern", "")
        topic = args.get("topic", "other")
        print(f"    [TOOL] track_concern('{concern[:40]}', '{topic}')")
        return {"main_concern": concern, "topic": topic}

    elif tool_name == "track_automatic_thought":
        thought = args.get("thought", "")
        distortion = args.get("distortion", "unknown")
        print(f"    [TOOL] track_automatic_thought('{thought[:40]}', '{distortion}')")
        return {"automatic_thought": thought, "distortion": distortion}

    elif tool_name == "transition_phase":
        reason = args.get("reason", "user ready")
        print(f"    [TOOL] transition_phase('{reason[:60]}')")
        return {"_transition_requested": True, "_transition_reason": reason}

    elif tool_name == "trigger_crisis_protocol":
        severity = args.get("severity", "medium")
        indicators = args.get("indicators", "")
        print(f"    [TOOL] trigger_crisis_protocol({severity}, '{indicators[:40]}')")
        return {"crisis_detected": True, "_crisis_severity": severity}

    print(f"    [TOOL ⚠️] unknown tool: {tool_name}")
    return {}
