# main.py — EduGen API (GPT-only, summarize without glossary/evidence) + Minimal Notes
# ---------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import os, re, json

# ---------- Bootstrapping ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")  # optional

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Please set it in .env")

client = OpenAI(api_key=OPENAI_API_KEY, project=OPENAI_PROJECT_ID) if OPENAI_PROJECT_ID else OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="EduGen API", version="3.6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Config ----------
NEAR_DUP_THRESHOLD = 0.78
MAX_TRIES_PER_BATCH = 4        # ขอซ้ำสูงสุดกี่ครั้งเพื่อให้ครบ n
CTX_CHAR_LIMIT = 15000         # context ต่อครั้ง
EXCLUDE_LIST_LIMIT = 30        # กัน prompt ยาวไป

# ---------- Minimal Notes Config ----------
NOTES_ROOT = os.path.join(os.getcwd(), "data", "notes")
os.makedirs(NOTES_ROOT, exist_ok=True)

def _require_user_id(req: Request) -> str:
    uid = (req.headers.get("X-User-Id") or "").strip()
    if not uid:
        raise HTTPException(401, "Missing X-User-Id header")
    if len(uid) > 128:
        raise HTTPException(400, "X-User-Id too long")
    return uid

def _note_path(user_id: str, file_id: str) -> str:
    safe_uid = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id)
    safe_fid = re.sub(r"[^A-Za-z0-9_.-]", "_", file_id)
    folder = os.path.join(NOTES_ROOT, safe_uid)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{safe_fid}.json")

# ---------- Utils ----------
def _sentences(text: str) -> List[str]:
    s = re.split(r"[。.!?]\s+|[\n\r]+", (text or "").strip())
    return [x.strip() for x in s if x.strip()]

def _clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n\n", text or "")
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()

def _strip_json_fence(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).strip()
        s = re.sub(r"```$", "", s).strip()
    return s

def _safe_json_loads(s: str, fallback: Union[dict, list, None] = None):
    try:
        return json.loads(_strip_json_fence(s))
    except Exception:
        return fallback if fallback is not None else {}

def _numbered_sentences(text: str, max_sentences: int = 800):
    sents = _sentences(text)
    sents = sents[:max_sentences]
    items = []
    for i, t in enumerate(sents, start=1):
        items.append({"id": i, "text": t})
    return items

def _truncate_text_chars(text: str, max_chars: int = 45000) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

# ---------- Near-duplicate helpers ----------
_STOP = set("คือ ของ และ หรือ ที่ ใน เป็น ได้ มี ใด ใดๆ อะไร อย่างไร ใคร ไหน ข้อใด ต่อไปนี้ มาก น้อย ไม่ ใช่ จาก ตาม เพื่อ เช่น ดังนั้น ดังกล่าว ซึ่ง โดย เพราะ ดังนั้นจึง".split())

def _tokenize(s: str) -> List[str]:
    return [
        w for w in re.sub(r"[^\w\s]", " ", (s or "").lower())
               .replace("ๆ"," ")
               .split()
        if w and w not in _STOP
    ]

def _jaccard(a: str, b: str) -> float:
    A, B = set(_tokenize(a)), set(_tokenize(b))
    if not A or not B: return 0.0
    inter = len(A & B)
    uni = len(A | B)
    return inter / uni if uni else 0.0

def _dice_bigram(a: str, b: str) -> float:
    def bi(x: str) -> List[str]:
        t = re.sub(r"\s+", " ", x).strip()
        return [t[i:i+2] for i in range(len(t)-1)] if len(t) > 1 else []
    A, B = bi(a), bi(b)
    if not A or not B: return 0.0
    from collections import Counter
    CA, CB = Counter(A), Counter(B)
    inter = 0
    for k, v in CA.items():
        inter += min(v, CB.get(k, 0))
    return (2 * inter) / (len(A) + len(B))

def _similar(a: str, b: str) -> float:
    return max(_jaccard(a, b), _dice_bigram(a, b))

def _filter_near_dups(items: List[Dict[str, Any]], exclude: List[str], threshold: float = NEAR_DUP_THRESHOLD) -> List[Dict[str, Any]]:
    """ตัดข้อที่ซ้ำ/ใกล้เคียงกับ exclude และกันซ้ำกันเองในชุด"""
    kept: List[Dict[str, Any]] = []
    for q in items:
        text = str(q.get("question") or "").strip()
        if not text:
            continue
        dup = False
        for e in exclude:
            if _similar(text, e) >= threshold:
                dup = True
                break
        if dup:
            continue
        for e in kept:  # กันซ้ำกันเองในชุด
            if _similar(text, str(e.get("question") or "")) >= threshold:
                dup = True
                break
        if not dup:
            kept.append(q)
    return kept

# ---------- Models ----------
class ContextIn(BaseModel):
    context: str

class QuizIn(BaseModel):
    context: str
    n: int = 5
    exclude: Optional[List[str]] = None      # รายการคำถามเดิม (ข้อความดิบ) เพื่อกันซ้ำ
    topics: Optional[List[str]] = None       # (ถ้ามี) บังคับให้สร้าง “หัวข้อละ 1 ข้อ”

class QAIn(BaseModel):
    context: str
    question: str

class SummarizeOut(BaseModel):
    overview: str
    key_points: List[str]
    sections: List[Dict[str, str]]   # {title, summary}
    data_points: List[Dict[str, str]]# {label, value, unit?}

# ---------- Endpoints ----------

@app.post("/pdf/extract")
async def pdf_extract(pdf: UploadFile = File(...)):
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "รองรับเฉพาะไฟล์ .pdf เท่านั้น")
    try:
        import pdfplumber
    except Exception:
        raise HTTPException(500, "กรุณาติดตั้ง pdfplumber: pip install pdfplumber")

    pages = []
    try:
        with pdfplumber.open(pdf.file) as doc:
            for p in doc.pages:
                pages.append(p.extract_text() or "")
    except Exception:
        raise HTTPException(422, "ไม่สามารถอ่านข้อความได้ (อาจเป็นไฟล์สแกน)")
    text = _clean_text("\n\n".join(pages))
    if not text:
        raise HTTPException(422, "ไม่สามารถอ่านข้อความได้ (อาจเป็นไฟล์สแกน)")
    return {"text": text}

# ---------- Summarization ----------
@app.post("/summarize", response_model=SummarizeOut)
def summarize(body: ContextIn):
    ctx_raw = (body.context or "").strip()
    if not ctx_raw:
        raise HTTPException(400, "context ว่าง")

    ctx = _clean_text(_truncate_text_chars(ctx_raw, 45000))
    sent_items = _numbered_sentences(ctx, max_sentences=800)
    if not sent_items:
        raise HTTPException(422, "เอกสารสั้นเกินไป")
    sent_block = "\n".join([f"[{it['id']}] {it['text']}" for it in sent_items])

    try:
        prompt_sections = f"""
คุณเป็นครูบรรณาธิการสรุปเอกสารแบบยึดตามข้อความเท่านั้น

กติกาหลัก:
- อ่านเฉพาะ "รายการประโยคมีเลขกำกับ" ด้านล่าง
- สกัดหัวข้อหลัก 5–9 หัวข้อ (เรียงจากกว้าง → ลึก หรือจากต้นเหตุ → ผลลัพธ์)
- แต่ละหัวข้อเขียนสรุปแบบย่อหน้า 1 ย่อหน้า ความยาว 3–6 ประโยค
- ครอบคลุมคำนิยาม/กระบวนการ/ตัวเลข/ผลกระทบ/ตัวอย่าง เท่าที่มี
- ห้ามแต่งเติมเนื้อหาที่ไม่มีในรายการ

รูปแบบคำตอบ: JSON เท่านั้น
{{
  "sections": [
    {{"title": "ชื่อหัวข้อหลัก", "summary": "สรุปแบบย่อหน้า 3–6 ประโยค"}}
  ]
}}

รายการประโยค:
{sent_block}
"""
        res1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_sections}],
            temperature=0.15,
            response_format={"type": "json_object"},
        )
        sec_json = _safe_json_loads(res1.choices[0].message.content, {"sections": []})
        sections = sec_json.get("sections", [])
        if not isinstance(sections, list):
            sections = []

        prompt_overview = f"""
คุณเป็นผู้ช่วยสรุประดับอาจารย์มหาวิทยาลัย ใช้เฉพาะข้อมูลจาก "รายการประโยค" และ "หัวข้อ" ด้านล่าง

สิ่งที่ต้องสร้าง (JSON เดียว):
- overview: 2–3 ย่อหน้า (180–350 คำ) ภาษาไทยลื่นไหล
- key_points: 6–10 ข้อ จุดละ ≤ 25 คำ
- data_points: 3–8 ค่า (หรือเท่าที่พบ) ในรูปแบบ {{"label","value","unit"}}

กติกาทั่วไป:
- อ้างอิงเฉพาะข้อความที่ให้มา ห้ามแต่งเติม
- ถ้าไม่พบ ให้คืนค่าเป็น [] แทน

โครง JSON:
{{
  "overview": "...",
  "key_points": ["..."],
  "data_points": [{{"label":"...","value":"...","unit":"..."}}]
}}

รายการประโยค:
{sent_block}

หัวข้อ (JSON จากขั้นก่อนหน้า):
{json.dumps({"sections": sections}, ensure_ascii=False)}
"""
        res2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "คุณสรุปได้กระชับ ชัด และยึดข้อความต้นฉบับเท่านั้น"},
                {"role": "user", "content": prompt_overview},
            ],
            temperature=0.15,
            response_format={"type": "json_object"},
        )
        ov_json = _safe_json_loads(
            res2.choices[0].message.content,
            {"overview": "", "key_points": [], "data_points": []},
        )

        def _norm_list(x):
            return x if isinstance(x, list) else []
        def _norm_str(x):
            return (x or "").strip()

        cleaned_sections: List[Dict[str, str]] = []
        for s in sections:
            if not isinstance(s, dict):
                continue
            title = _norm_str(s.get("title", ""))
            summary = _norm_str(s.get("summary", ""))
            if title and summary:
                cleaned_sections.append({"title": title, "summary": summary})

        dps_in = ov_json.get("data_points", [])
        cleaned_dps: List[Dict[str, str]] = []
        if isinstance(dps_in, list):
            for d in dps_in:
                if not isinstance(d, dict):
                    continue
                label = _norm_str(d.get("label", ""))
                value = _norm_str(d.get("value", ""))
                unit  = _norm_str(d.get("unit", ""))
                if label and value:
                    item: Dict[str, str] = {"label": label, "value": value}
                    if unit:
                        item["unit"] = unit
                    cleaned_dps.append(item)

        return {
            "overview": _norm_str(ov_json.get("overview", "")),
            "key_points": _norm_list(ov_json.get("key_points", [])),
            "sections": cleaned_sections,
            "data_points": cleaned_dps,
        }

    except Exception as e:
        raise HTTPException(500, f"Summarize failed: {e}")

# ---------- Topics extraction (NEW) ----------
class TopicsOut(BaseModel):
    topics: List[str]

@app.post("/quiz/topics", response_model=TopicsOut)
def quiz_topics(body: ContextIn):
    """แตกหัวข้อสำคัญจาก context ให้หลากหลาย (เอาไปสุ่มใช้สร้างคำถามหัวข้อละ 1 ข้อ)"""
    ctx = (body.context or "").strip()
    if not ctx:
        raise HTTPException(400, "context ว่าง")
    prompt = f"""
สกัดหัวข้อ/แนวคิดสำคัญจากเนื้อหาด้านล่าง (ไม่เกิน 30 หัวข้อ) โดยแต่ละหัวข้อควรเป็นคนละมุมมอง เช่น นิยาม หลักการทำงาน องค์ประกอบ ขั้นตอน ตัวเลข/มาตรฐาน ข้อดี-ข้อจำกัด ผลกระทบ เปรียบเทียบ กรณีศึกษา สมการ ฯลฯ
- ใช้ถ้อยคำสั้น ชัด อ่านแล้วรู้ว่าต่างหัวข้อกันจริง
- ตอบเป็น JSON เท่านั้น: {{"topics": ["หัวข้อ1","หัวข้อ2", "..."]}}

เนื้อหา:
{_truncate_text_chars(ctx, CTX_CHAR_LIMIT)}
"""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        data = _safe_json_loads(r.choices[0].message.content, {"topics": []})
        topics = [str(t).strip() for t in data.get("topics", []) if str(t).strip()]
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(500, f"Topics generation failed: {e}")

# ---------- Internal generators (ensure n by retry) ----------
def _gen_mcq_once(ctx: str, n: int, exclude_list: List[str], topic_hints: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    exclude_block = ""
    if exclude_list:
        exclude_block = "หลีกเลี่ยงการตั้งคำถามคล้ายกับ (ถ้าเจตนาเดียวกัน ให้เปลี่ยนหัวข้อ):\n" + "\n".join(f"- {q}" for q in exclude_list[:EXCLUDE_LIST_LIMIT]) + "\n"

    topic_block = ""
    if topic_hints:
        topic_block = "ให้สร้าง 'หัวข้อละ 1 ข้อ' จากหัวข้อต่อไปนี้ (ห้ามซ้ำ/ใกล้เคียงกัน):\n" + "\n".join(f"- {t}" for t in topic_hints[:n]) + "\n"

    prompt = f"""
สร้างข้อสอบแบบปรนัย {n} ข้อ จากเนื้อหาด้านล่าง
- ต้องมีคำตอบถูกเพียงข้อเดียว
- ห้ามใช้ตัวเลือกแบบรวม เช่น "ถูกทุกข้อ", "ทั้งหมด", "ทั้ง ก และ ข", "ก และ ข และ ค", "ไม่ถูกสักข้อ"
- กระจายหัวข้อให้หลากหลาย: นิยาม หลักการทำงาน องค์ประกอบ/โครงสร้าง ขั้นตอน ตัวเลข/ปี/มาตรฐาน ข้อดี-ข้อจำกัด เปรียบเทียบ ฯลฯ
{topic_block}{exclude_block}- หลีกเลี่ยงการแต่งเติมนอกเนื้อหา
- ตอบเป็น JSON เท่านั้น:
{{
  "questions": [
    {{
      "type": "mcq",
      "question": "....",
      "choices": ["ก) ...","ข) ...","ค) ...","ง) ..."],
      "answer": "ก|ข|ค|ง",
      "explain": "...",
      "topic": "หัวข้อย่อยของคำถามนี้"
    }}
  ]
}}
เนื้อหา:
{_truncate_text_chars(ctx, CTX_CHAR_LIMIT)}
"""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    data = _safe_json_loads(r.choices[0].message.content, {"questions": []})
    qs = data.get("questions", [])
    return _filter_near_dups(qs, exclude_list, threshold=NEAR_DUP_THRESHOLD)

def _gen_tf_once(ctx: str, n: int, exclude_list: List[str], topic_hints: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    exclude_block = ""
    if exclude_list:
        exclude_block = "หลีกเลี่ยงการตั้งคำถามคล้ายกับ (ถ้าเจตนาเดียวกัน ให้เปลี่ยนหัวข้อ):\n" + "\n".join(f"- {q}" for q in exclude_list[:EXCLUDE_LIST_LIMIT]) + "\n"

    topic_block = ""
    if topic_hints:
        topic_block = "ให้สร้าง 'หัวข้อละ 1 ข้อ' จากหัวข้อต่อไปนี้ (ห้ามซ้ำ/ใกล้เคียงกัน):\n" + "\n".join(f"- {t}" for t in topic_hints[:n]) + "\n"

    prompt = f"""
สร้างข้อสอบแบบ ถูก/ผิด (True/False) จำนวน {n} ข้อ จากเนื้อหาด้านล่าง
- ให้เหตุผลสั้น ๆ ทุกข้อ
- กระจายหัวข้อให้หลากหลาย: แนวคิด กลไก/เงื่อนไข ตัวเลข/ค่ามาตรฐาน ข้อควรระวัง เปรียบเทียบ ผลกระทบ ฯลฯ
{topic_block}{exclude_block}- หลีกเลี่ยงการแต่งเติมนอกเนื้อหา
- ตอบเป็น JSON เท่านั้น:
{{
  "questions": [
    {{"type": "tf", "question": "...", "answer": "true|false", "explain": "...", "topic": "หัวข้อย่อย"}}
  ]
}}
เนื้อหา:
{_truncate_text_chars(ctx, CTX_CHAR_LIMIT)}
"""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25,
        response_format={"type": "json_object"},
    )
    data = _safe_json_loads(r.choices[0].message.content, {"questions": []})
    qs = data.get("questions", [])
    return _filter_near_dups(qs, exclude_list, threshold=NEAR_DUP_THRESHOLD)

# ---------- MCQ Quiz (multi-pass; ensure n) ----------
@app.post("/quiz/mcq")
def quiz_mcq(body: QuizIn):
    ctx = (body.context or "").strip()
    n = max(1, min(10, int(body.n or 5)))
    if not ctx:
        raise HTTPException(400, "context ว่าง")

    exclude_list = [str(x).strip() for x in (body.exclude or []) if str(x).strip()]
    topics = [str(t).strip() for t in (body.topics or []) if str(t).strip()] or None

    # multi-pass เพื่อให้ "ได้ครบ n" แน่ๆ
    collected: List[Dict[str, Any]] = []
    tries = 0
    while len(collected) < n and tries < MAX_TRIES_PER_BATCH:
        need = n - len(collected)
        # ส่ง exclude = ข้อเดิมทั้งหมด + ที่เก็บได้ในรอบก่อน
        excludes_now = exclude_list + [str(q.get("question") or "") for q in collected]
        # ถ้ามี topics ให้ป้อนหัวข้อเท่าที่ต้องการในรอบนี้
        topic_hints = topics[:need] if topics else None

        batch = _gen_mcq_once(ctx, need, excludes_now, topic_hints)
        # เพิ่มเฉพาะที่ยังไม่ซ้ำกับ collected (กันอีกชั้น)
        for q in batch:
            if all(_similar(str(q.get("question","")), str(e.get("question",""))) < NEAR_DUP_THRESHOLD for e in collected):
                collected.append(q)
        # ตัด topics ที่ใช้ไปแล้ว (ถ้าส่งมา)
        if topics:
            used = set(str(q.get("topic","")).strip().lower() for q in collected)
            topics = [t for t in topics if str(t).strip().lower() not in used]
        tries += 1

    if not collected:
        return {"questions": []}

    return {"questions": collected[:n]}

# ---------- True/False Quiz (multi-pass; ensure n) ----------
@app.post("/quiz/tf")
def quiz_tf(body: QuizIn):
    ctx = (body.context or "").strip()
    n = max(1, min(10, int(body.n or 5)))
    if not ctx:
        raise HTTPException(400, "context ว่าง")

    exclude_list = [str(x).strip() for x in (body.exclude or []) if str(x).strip()]
    topics = [str(t).strip() for t in (body.topics or []) if str(t).strip()] or None

    collected: List[Dict[str, Any]] = []
    tries = 0
    while len(collected) < n and tries < MAX_TRIES_PER_BATCH:
        need = n - len(collected)
        excludes_now = exclude_list + [str(q.get("question") or "") for q in collected]
        topic_hints = topics[:need] if topics else None

        batch = _gen_tf_once(ctx, need, excludes_now, topic_hints)
        for q in batch:
            if all(_similar(str(q.get("question","")), str(e.get("question",""))) < NEAR_DUP_THRESHOLD for e in collected):
                collected.append(q)
        if topics:
            used = set(str(q.get("topic","")).strip().lower() for q in collected)
            topics = [t for t in topics if str(t).strip().lower() not in used]
        tries += 1

    if not collected:
        return {"questions": []}

    return {"questions": collected[:n]}

# ---------- Q/A ----------
@app.post("/qa")
def qa(body: QAIn):
    ctx = (body.context or "").strip()
    q = (body.question or "").strip()
    if not ctx or not q:
        raise HTTPException(400, "context/question ว่าง")

    prompt = f"""
ตอบคำถามโดยอ้างอิง "เฉพาะ" เนื้อหาที่ให้ด้านล่างเท่านั้น
ถ้าไม่พบคำตอบ ให้ตอบว่า: ไม่พบในเนื้อหาที่ให้มา

เนื้อหา:
{_truncate_text_chars(ctx, CTX_CHAR_LIMIT)}

คำถาม: {q}
ตอบ:
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
        )
        return {"answer": (res.choices[0].message.content or "").strip()}
    except Exception as e:
        raise HTTPException(500, f"QA failed: {e}")

# ---------- Minimal Notes API (NEW) ----------
class SimpleNoteIn(BaseModel):
    content: str

class SimpleNoteOut(BaseModel):
    file_id: str
    content: str
    updated_at: Optional[str] = None

@app.get("/notes/{file_id}", response_model=SimpleNoteOut)
def get_note(request: Request, file_id: str = Path(...)):
    uid = _require_user_id(request)
    p = _note_path(uid, file_id)
    if not os.path.exists(p):
        return {"file_id": file_id, "content": "", "updated_at": None}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "file_id": file_id,
            "content": str(data.get("content", "")),
            "updated_at": data.get("updated_at"),
        }
    except Exception:
        # ถ้าไฟล์เสีย คืนค่าเปล่าๆ แทน
        return {"file_id": file_id, "content": "", "updated_at": None}

@app.put("/notes/{file_id}", response_model=SimpleNoteOut)
def put_note(request: Request, file_id: str = Path(...), body: SimpleNoteIn = None):
    uid = _require_user_id(request)
    if body is None or not isinstance(body.content, str):
        raise HTTPException(400, "content ว่างหรือรูปแบบไม่ถูกต้อง")
    content = body.content.strip()
    p = _note_path(uid, file_id)
    payload = {"content": content, "updated_at": datetime.utcnow().isoformat() + "Z"}
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return {"file_id": file_id, **payload}
