"""Build mixed calibration set: 128 EN from corpus.json + 128 synthetic Thai CBT.

Output: /workspace/awq_work/calib_data.jsonl  (one JSON per line, key="text")
"""
import json
import random
from pathlib import Path
from transformers import AutoTokenizer

random.seed(42)

MODEL_PATH = "/workspace/models/ultravox/merged-full-v2"
CORPUS_PATH = "/app/rag/corpus.json"
OUT_PATH = "/workspace/awq_work/calib_data.jsonl"
SYS_PROMPT = "คุณคือพี่มิร่า เพื่อนคู่ใจ AI ที่ใช้ CBT framework"
MAX_LEN = 2048
N_EN = 128
N_TH = 128

# ---------- Thai synthetic prompts ----------
# Templates cover the dominant Thai-CBT user topics: anxiety, depression, work
# stress, relationships, family, self-worth, sleep, rumination, anger, grief.
THAI_OPENERS = [
    "พี่มิร่าคะ", "พี่มิร่าครับ", "หนูอยากปรึกษาหน่อย",
    "ผมไม่รู้จะทำยังไงแล้วครับ", "หนูรู้สึก", "พี่ช่วยหนูหน่อยได้ไหม",
    "ช่วงนี้", "ผมว่าผม", "หนูเริ่มจะ", "ตอนนี้",
]

THAI_FEELINGS = [
    "เครียดมากเลยค่ะ", "ไม่ไหวแล้วครับ", "เศร้าจนไม่อยากลุกจากเตียง",
    "หงุดหงิดกับตัวเองตลอดเวลา", "รู้สึกว่าตัวเองไม่มีค่า",
    "นอนไม่หลับมาหลายวันแล้ว", "กลัวอนาคตมาก", "คิดมากจนปวดหัว",
    "เหนื่อยทั้งกายทั้งใจ", "รู้สึกเหงาแม้อยู่กับคนอื่น",
    "ใจสั่นทุกครั้งที่ต้องออกจากบ้าน", "อยากร้องไห้แต่ร้องไม่ออก",
    "โกรธตัวเองที่ทำพลาด", "รู้สึกผิดที่ไม่สามารถช่วยพ่อแม่ได้",
    "กลัวว่าจะทำให้คนอื่นผิดหวัง", "รู้สึกว่าไม่มีใครเข้าใจ",
]

THAI_CONTEXTS = [
    "เพราะงานเยอะมาก เจ้านายก็กดดันตลอด",
    "ตั้งแต่เลิกกับแฟน ก็คิดเรื่องเขาทุกวัน",
    "พ่อแม่คาดหวังกับผมสูง แต่ผมรู้สึกทำไม่ได้",
    "เพื่อนที่ทำงานนินทาลับหลัง รู้สึกไม่ปลอดภัย",
    "เรียนหนัก สอบใกล้แล้วยังเตรียมตัวไม่ทัน",
    "ตกงานมา 3 เดือน หางานก็ไม่ได้",
    "ทะเลาะกับแม่เรื่องเดิมๆ ทุกวัน",
    "เปรียบเทียบตัวเองกับเพื่อนใน social media ตลอด",
    "ย้ายมาทำงานต่างจังหวัด คิดถึงครอบครัวมาก",
    "เพิ่งเสียคุณยายไป ใจหายมาก",
    "ลูกป่วยหนัก รักษาก็ไม่หาย ไม่รู้จะทำยังไง",
    "ทำธุรกิจขาดทุน หนี้สินเยอะมาก",
    "รู้สึกว่าชีวิตไม่มีเป้าหมาย ตื่นมาก็ไม่รู้จะทำอะไร",
    "เพื่อนสนิทเลิกคุยด้วย ไม่รู้ว่าทำผิดอะไร",
    "อกหักครั้งนี้รู้สึกแย่กว่าทุกครั้ง",
    "พึ่งหย่ากับสามี ลูกก็ยังเล็ก ไม่รู้จะเลี้ยงคนเดียวยังไง",
]

THAI_QUESTIONS = [
    "หนูควรทำยังไงดีคะ",
    "ผมจะผ่านมันไปได้ไหมครับ",
    "พี่มีวิธีให้ใจสงบลงบ้างไหม",
    "หนูจะหยุดคิดเรื่องนี้ยังไงดี",
    "ผมเป็นโรคซึมเศร้าหรือเปล่า",
    "ทำไมหนูถึงรู้สึกแบบนี้คะ",
    "พี่ช่วยแนะนำหน่อยได้ไหมว่าควรเริ่มจากตรงไหน",
    "หนูควรไปหาหมอไหมคะ",
    "ผมต้องอดทนต่อไปอีกนานแค่ไหน",
    "มีวิธีนอนหลับโดยไม่ต้องกินยาไหมครับ",
    "",
    "",
]

def gen_thai_prompts(n):
    seen = set()
    out = []
    while len(out) < n:
        opener = random.choice(THAI_OPENERS)
        feeling = random.choice(THAI_FEELINGS)
        ctx = random.choice(THAI_CONTEXTS)
        q = random.choice(THAI_QUESTIONS)
        # 3 sentence stitching variants for diversity
        variant = random.randint(0, 2)
        if variant == 0:
            text = f"{opener} {feeling} {ctx} {q}".strip()
        elif variant == 1:
            text = f"{opener} {ctx} เลย{feeling} {q}".strip()
        else:
            text = f"{feeling} {ctx} {q}".strip()
        text = " ".join(text.split())  # collapse whitespace
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out

# ---------- main ----------
print("[1/4] loading corpus.json (EN)...")
corpus = json.load(open(CORPUS_PATH))
en_pool = [r["user_input"].strip() for r in corpus
           if isinstance(r.get("user_input"), str) and 30 < len(r["user_input"]) < 2000]
print(f"      eligible EN samples: {len(en_pool)}")

en_samples = random.sample(en_pool, N_EN)
print(f"      sampled {len(en_samples)} EN")

print("[2/4] generating Thai synthetic prompts...")
th_samples = gen_thai_prompts(N_TH)
print(f"      generated {len(th_samples)} unique TH")

print("[3/4] loading tokenizer + applying chat template...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

records = []
overlong = 0
for u in en_samples + th_samples:
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": u},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Length check (tokenized)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) > MAX_LEN:
        # Truncate user content if too long (rare, only on outliers)
        keep = MAX_LEN - 64
        text = tok.decode(ids[:keep], skip_special_tokens=False)
        overlong += 1
    records.append({"text": text, "n_tokens": min(len(ids), MAX_LEN)})

print(f"      truncated: {overlong}/{len(records)}")

print(f"[4/4] writing {OUT_PATH}...")
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Stats
import statistics as st
toks = [r["n_tokens"] for r in records]
print(f"\n[OK] wrote {len(records)} samples")
print(f"     token stats: min={min(toks)} median={int(st.median(toks))} "
      f"mean={int(st.mean(toks))} p95={sorted(toks)[int(0.95*len(toks))]} max={max(toks)}")
print(f"     EN: {N_EN}  TH: {N_TH}")

# Show 1 sample of each
print("\n--- EN sample ---")
print(records[0]["text"][:600])
print("\n--- TH sample ---")
print(records[N_EN]["text"][:600])
