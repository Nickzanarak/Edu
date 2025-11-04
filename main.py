# main.py — EduGen API (GPT-only, summarize without glossary/evidence)
# ---------------------------------------------------------------
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any, Union
import os, re, json

# ---------- Bootstrapping ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # บังคับให้มี API key (ไม่ใช้โหมดเดโม่)
    raise RuntimeError("OPENAI_API_KEY is missing. Please set it in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="EduGen API", version="3.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Utils ----------
def _sentences(text: str) -> List[str]:
    # แยกประโยคแบบง่าย + รองรับขึ้นบรรทัดใหม่
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

def _numbered_sentences(text: str, max_sentences: int = 500):
    # ทำรายการประโยคพร้อมเลขกำกับ (ใช้คุมไม่ให้แต่งเติม แต่จะไม่ส่งเลขออก)
    sents = _sentences(text)
    sents = sents[:max_sentences]
    items = []
    for i, t in enumerate(sents, start=1):
        items.append({"id": i, "text": t})
    return items

def _truncate_text_chars(text: str, max_chars: int = 28000) -> str:
    # กันข้อความยาวเกิน
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

# ---------- Models ----------
class ContextIn(BaseModel):
    context: str

class QuizIn(BaseModel):
    context: str
    n: int = 5

class QAIn(BaseModel):
    context: str
    question: str

# สำหรับ summarize (เอาท์พุต) — ไม่มี glossary / evidence / key_terms
class SummarizeOut(BaseModel):
    overview: str
    key_points: List[str]
    sections: List[Dict[str, str]]   # {title, summary}
    data_points: List[Dict[str, str]]# {label, value, unit?}

# ---------- Endpoints ----------

@app.post("/pdf/extract")
async def pdf_extract(pdf: UploadFile = File(...)):
    """Extract and clean text from PDF."""
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

# ---------- Summarization (Overview + Sections, no glossary/evidence) ----------
@app.post("/summarize", response_model=SummarizeOut)
def summarize(body: ContextIn):
    ctx_raw = (body.context or "").strip()
    if not ctx_raw:
        raise HTTPException(400, "context ว่าง")

    # ตัดความยาวเบื้องต้นกัน prompt เกิน
    ctx = _clean_text(_truncate_text_chars(ctx_raw, 28000))

    # 1) ทำ numbered sentences เพื่อคุมไม่ให้แต่งเติม (แต่จะไม่ส่งเลขออก)
    sent_items = _numbered_sentences(ctx, max_sentences=500)
    if not sent_items:
        raise HTTPException(422, "เอกสารสั้นเกินไป")

    # บล็อคที่โมเดลอ่าน: [id] sentence (ใช้เป็นขอบเขตเนื้อหา)
    sent_block = "\n".join([f"[{it['id']}] {it['text']}" for it in sent_items])

    try:
        # ---------- Pass A: Sectioning (title + summary เท่านั้น) ----------
        prompt_sections = f"""
คุณเป็นผู้ช่วยสรุปเอกสารแบบยึดตามข้อความเท่านั้น
กติกา:
- อ่านเฉพาะ "รายการประโยคมีเลขกำกับ" ด้านล่าง
- แยกหัวข้อหลัก 4–8 หัวข้อ
- แต่ละหัวข้อให้สรุป 1–2 บรรทัด อ่านง่าย กระชับ
- ห้ามแต่งเติมเนื้อหาที่ไม่มีในรายการ

ตอบเป็น JSON เท่านั้น ตามโครง:
{{
  "sections": [
    {{"title": "หัวข้อหลัก", "summary": "สรุปสั้น 1–2 บรรทัด"}}
  ]
}}

รายการประโยค:
{sent_block}
"""
        res1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_sections}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        sec_json = _safe_json_loads(res1.choices[0].message.content, {"sections": []})
        sections = sec_json.get("sections", [])
        if not isinstance(sections, list):
            sections = []

        # ---------- Pass B: Overview + Key bullets + Data points (ไม่มี evidence) ----------
        prompt_overview = f"""
คุณเป็นผู้ช่วยสรุประดับอาจารย์มหาวิทยาลัย
ใช้เฉพาะข้อความในรายการประโยค (มีเลขกำกับ) และข้อมูลหัวข้อที่ได้ (JSON) เพื่อเขียนสรุปภาพรวม
ห้ามเพิ่มเนื้อหาที่ไม่มีในรายการ

สิ่งที่ต้องสร้าง (ตอบ JSON เดียว):
- overview: ย่อหน้า 1–2 ย่อหน้า ภาษาไทย ลื่นไหล ชัดเจน
- key_points: bullet 5–8 ข้อ สั้น กระชับ
- data_points: ถ้ามีตัวเลข/หน่วย/สูตรสำคัญ ให้สกัดเป็นรายการ {{"label","value","unit"}}
กติกา: ถ้าไม่พบข้อมูลบางส่วน ให้ส่งเป็น [] แทน

โครง JSON:
{{
  "overview": "...",
  "key_points": ["...", "..."],
  "data_points": [{{"label":"...","value":"...","unit":"..."}}]
}}

รายการประโยค:
{sent_block}

หัวข้อ (JSON):
{json.dumps({"sections": sections}, ensure_ascii=False)}
"""
        res2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "คุณสรุปได้กระชับ ชัด ละเอียดพอดี และยึดข้อความต้นฉบับเท่านั้น"},
                {"role": "user", "content": prompt_overview},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        ov_json = _safe_json_loads(
            res2.choices[0].message.content,
            {"overview": "", "key_points": [], "data_points": []},
        )

        # sanitize เบื้องต้น
        def _norm_list(x):
            return x if isinstance(x, list) else []
        def _norm_str(x):
            return (x or "").strip()

        # sections เหลือแค่ title/summary
        cleaned_sections: List[Dict[str, str]] = []
        for s in sections:
            if not isinstance(s, dict): 
                continue
            title = _norm_str(s.get("title", ""))
            summary = _norm_str(s.get("summary", ""))
            if title and summary:
                cleaned_sections.append({"title": title, "summary": summary})

        # data_points ไม่มี evidence_ids
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
        # ไม่ใช้เดโม่; แจ้ง error ให้ frontend รับรู้
        raise HTTPException(500, f"Summarize failed: {e}")

# ---------- MCQ Quiz ----------
@app.post("/quiz/mcq")
def quiz_mcq(body: QuizIn):
    ctx = (body.context or "").strip()
    n = max(1, min(10, body.n))
    if not ctx:
        raise HTTPException(400, "context ว่าง")

    prompt = f"""
สร้างข้อสอบแบบปรนัย {n} ข้อ จากเนื้อหาด้านล่าง
- ทุกข้อมีตัวเลือก "ก/ข/ค/ง" และเฉลยเป็นตัวอักษรเดียว (ก/ข/ค/ง)
- ใส่คำอธิบายเฉลยสั้นๆ ทุกข้อ
- หลีกเลี่ยงการแต่งเติมนอกเนื้อหา
- ตอบเป็น JSON เท่านั้น ตามโครง:
{{
  "questions": [
    {{
      "type": "mcq",
      "question": "....",
      "choices": ["ก) ...","ข) ...","ค) ...","ง) ..."],
      "answer": "ก|ข|ค|ง",
      "explain": "..."
    }}
  ]
}}
เนื้อหา:
{_truncate_text_chars(ctx, 15000)}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        data = _safe_json_loads(res.choices[0].message.content, {"questions": []})
        return {"questions": data.get("questions", [])}
    except Exception as e:
        raise HTTPException(500, f"MCQ generation failed: {e}")

# ---------- True/False Quiz ----------
@app.post("/quiz/tf")
def quiz_tf(body: QuizIn):
    ctx = (body.context or "").strip()
    n = max(1, min(10, body.n))
    if not ctx:
        raise HTTPException(400, "context ว่าง")

    prompt = f"""
สร้างข้อสอบแบบ ถูก/ผิด (True/False) จำนวน {n} ข้อ จากเนื้อหาด้านล่าง
- ให้เหตุผลสั้นๆ ทุกข้อ
- หลีกเลี่ยงการแต่งเติมนอกเนื้อหา
- ตอบเป็น JSON เท่านั้น ตามโครง:
{{
  "questions": [
    {{"type": "tf", "question": "...", "answer": "true|false", "explain": "..."}}
  ]
}}
เนื้อหา:
{_truncate_text_chars(ctx, 15000)}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            response_format={"type": "json_object"},
        )
        data = _safe_json_loads(res.choices[0].message.content, {"questions": []})
        return {"questions": data.get("questions", [])}
    except Exception as e:
        raise HTTPException(500, f"TF generation failed: {e}")

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
{_truncate_text_chars(ctx, 15000)}

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
