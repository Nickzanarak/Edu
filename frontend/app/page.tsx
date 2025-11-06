"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode, ButtonHTMLAttributes } from "react";

/* ---------- Config ---------- */
const MAX_QUESTIONS = 15;         // ต่อชนิด (MCQ 15 + TF 15)
const BATCH_SIZE = 5;
const MAX_RETRY = 2;
const NEAR_DUP_TH = 0.78;

/* ---------- Types ---------- */
type QuizItem = { type: "mcq" | "tf"; question: string; choices?: string[]; answer: string; explain?: string };
type Section = { title: string; summary: string };
type DataPoint = { label: string; value: string; unit?: string };
type SummarizeResponse = { overview: string; key_points: string[]; sections: Section[]; data_points: DataPoint[] };

/* ---------- Type guards ---------- */
function hasArrayQuestions(x: unknown): x is { questions: unknown[] } {
  if (typeof x !== "object" || x === null) return false;
  const q = (x as { questions?: unknown }).questions;
  return Array.isArray(q);
}
function hasDetail(x: unknown): x is { detail?: string } {
  return typeof x === "object" && x !== null && "detail" in x;
}
function isStringArray(v: unknown): v is string[] {
  return Array.isArray(v) && v.every((x) => typeof x === "string");
}

/* ---------- Helpers ---------- */
const idxToLetter = ["ก", "ข", "ค", "ง"];
const letterToIdx: Record<string, number> = { ก: 0, ข: 1, ค: 2, ง: 3 };

const stripChoiceLabel = (s: string) => String(s).replace(/^\s*[กขคง]\)\s*/i, "").trim();
const toStr = (v: unknown) => (typeof v === "string" ? v : String(v ?? "")).trim();
function shuffle<T>(arr: T[]) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
const keyFor = (q: QuizItem) =>
  q.question.normalize("NFKC").replace(/\s+/g, " ").replace(/[^\p{L}\p{N}\s]/gu, "").trim().toLowerCase();

/* -------- Near-duplicate -------- */
const STOP = new Set(["คือ","ของ","และ","หรือ","ที่","ใน","เป็น","ได้","มี","ใด","ใดๆ","อะไร","อย่างไร","ใคร","ไหน","ข้อใด","ต่อไปนี้","มาก","น้อย","ไม่","ใช่","จาก","เพื่อ","เช่น","ซึ่ง","ดังนั้น","โดย","ดังกล่าว"]);
const tokenize = (s: string) => s.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}\s]/gu," ").split(/\s+/).filter(w => w && !STOP.has(w));
const jaccard = (a: string, b: string) => { const A=new Set(tokenize(a)),B=new Set(tokenize(b)); if(!A.size||!B.size) return 0; let inter=0; for(const w of A) if(B.has(w)) inter++; return inter/(A.size+B.size-inter); };
const diceBigram = (s: string, t: string) => { const bi=(x:string)=>{const z=x.replace(/\s+/g," ").trim(); const out:string[]=[]; for(let i=0;i<z.length-1;i++) out.push(z.slice(i,i+2)); return out;}; const A=bi(s),B=bi(t); if(!A.length||!B.length) return 0; const m=new Map<string,number>(); for(const x of A) m.set(x,(m.get(x)??0)+1); let inter=0; for(const y of B){const c=m.get(y)??0; if(c>0){inter++; m.set(y,c-1);}} return (2*inter)/(A.length+B.length); };
const similar = (a: string,b: string) => Math.max(jaccard(a,b), diceBigram(a,b));

/** ทำให้ payload จาก /quiz/* เป็น QuizItem[] ที่สะอาด */
function normalizeFromAPI(payload: unknown): QuizItem[] {
  const raw: unknown[] = hasArrayQuestions(payload) ? payload.questions : [];
  const out: QuizItem[] = [];
  for (const item of raw) {
    if (typeof item !== "object" || item === null) continue;
    const q = item as Record<string, unknown>;
    const type = toStr(q.type).toLowerCase();
    const question = toStr(q.question);
    let answer = toStr(q.answer).toLowerCase();
    const explain = q.explain !== undefined ? toStr(q.explain) : undefined;
    if (!type || !question) continue;

    if (type === "mcq") {
      const choicesRaw = isStringArray(q.choices) ? q.choices : [];
      const mapped = choicesRaw.map(stripChoiceLabel).filter(Boolean).slice(0, 4);
      const choices: string[] = mapped.length < 4 ? [...mapped, ...Array.from({ length: 4 - mapped.length }, (_, i) => `ตัวเลือกที่ ${mapped.length + i + 1}`)] : mapped;

      if (!["ก", "ข", "ค", "ง"].includes(answer)) {
        const num = Number(answer);
        if (Number.isFinite(num)) {
          const idx = num >= 1 && num <= 4 ? num - 1 : num;
          if (idx >= 0 && idx < 4) answer = idxToLetter[idx];
        }
        if (!["ก", "ข", "ค", "ง"].includes(answer)) {
          const idx = choices.findIndex((c) => c.replace(/\s+/g, "") === answer.replace(/\s+/g, ""));
          answer = idx >= 0 ? idxToLetter[idx] : "ก";
        }
      }
      out.push({ type: "mcq", question, choices, answer, explain });
      continue;
    }

    if (!["true", "false"].includes(answer)) {
      const trueSet = new Set(["true","t","1","yes","y","จริง","ถูก"]);
      const falseSet = new Set(["false","f","0","no","n","เท็จ","ผิด"]);
      if (trueSet.has(answer)) answer = "true"; else if (falseSet.has(answer)) answer = "false"; else answer = "false";
    }
    out.push({ type: "tf", question, answer, explain });
  }
  return out;
}

/** สลับช้อยส์เฉพาะ batch แล้ว remap เฉลย */
function shuffleAndRemapBatch(qs: QuizItem[]): QuizItem[] {
  return shuffle(qs).map((q) => {
    if (q.type !== "mcq" || !q.choices?.length) return q;
    const original = [...q.choices];
    const correctIdx = letterToIdx[q.answer] ?? -1;
    const correctText = correctIdx >= 0 && correctIdx < original.length ? original[correctIdx] : null;
    const newChoices = shuffle(original);
    let newAns = q.answer;
    if (correctText) {
      const newIdx = newChoices.findIndex((t) => t === correctText);
      if (newIdx >= 0) newAns = idxToLetter[newIdx] || newAns;
    }
    return { ...q, choices: newChoices, answer: newAns };
  });
}

/* ---------- UI primitives ---------- */
const Card = ({ className = "", children }: { className?: string; children: ReactNode }) => (
  <div className={`rounded-2xl border border-zinc-800/40 bg-zinc-900/40 backdrop-blur-sm p-5 ${className}`}>{children}</div>
);
const Label = ({ children }: { children: ReactNode }) => (
  <span className="inline-flex items-center gap-2 text-xs rounded-full px-2 py-1 bg-zinc-800/60 border border-zinc-700/60">
    {children}
  </span>
);
const PrimaryBtn = ({ children, className, ...props }:
  ButtonHTMLAttributes<HTMLButtonElement> & { children: ReactNode }) => (
  <button
    {...props}
    className={`px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium shadow-sm disabled:opacity-50 ${className}`}
  >
    {children}
  </button>
);

/* ---------- Simple Modal ---------- */
function Modal({
  open, onClose, title, children, rightInfo,
}: { open: boolean; onClose: () => void; title: string; children: ReactNode; rightInfo?: ReactNode }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div className="w-full max-w-3xl rounded-2xl border border-zinc-800 bg-zinc-900 shadow-2xl">
          <div className="flex items-center justify-between px-5 py-3 border-b border-zinc-800">
            <div className="font-semibold">{title}</div>
            <div className="flex items-center gap-3 text-xs text-zinc-400">{rightInfo}
              <button onClick={onClose} className="ml-3 rounded-lg px-2 py-1 hover:bg-zinc-800">✕</button>
            </div>
          </div>
          <div className="p-5">{children}</div>
        </div>
      </div>
    </div>
  );
}

/* ---------- Page ---------- */
export default function Home() {
  const API = process.env.NEXT_PUBLIC_API || "http://127.0.0.1:8000";

  // -------- Notes (keep hidden fileId; show as popup) --------
  const [userId, setUserId] = useState<string>("demo-user");
  useEffect(() => { try { const uid = localStorage.getItem("uid"); if (uid && uid.trim()) setUserId(uid.trim()); } catch {} }, []);
  const [fileId, setFileId] = useState<string>("manual");
  const [note, setNote] = useState<string>("");
  const [noteOpen, setNoteOpen] = useState(false);
  const [noteStatus, setNoteStatus] = useState<"idle"|"saving"|"saved"|"error">("idle");
  const [noteUpdatedAt, setNoteUpdatedAt] = useState<string | null>(null);
  const saveTimer = useRef<number | null>(null);
  const firstLoadRef = useRef<boolean>(true);

  const loadNote = async (fid: string) => {
    if (!fid) return;
    try {
      const res = await fetch(`${API}/notes/${encodeURIComponent(fid)}`, { headers: { "X-User-Id": userId || "demo-user" } });
      const json = await res.json();
      setNote(typeof json?.content === "string" ? json.content : "");
      setNoteUpdatedAt(typeof json?.updated_at === "string" ? json.updated_at : null);
      setNoteStatus("idle");
      firstLoadRef.current = false;
    } catch { setNote(""); setNoteUpdatedAt(null); setNoteStatus("error"); }
  };
  const autosaveNote = async (fid: string, content: string) => {
    if (!fid) return;
    try {
      setNoteStatus("saving");
      const res = await fetch(`${API}/notes/${encodeURIComponent(fid)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json", "X-User-Id": userId || "demo-user" },
        body: JSON.stringify({ content }),
      });
      const json = await res.json();
      setNoteUpdatedAt(typeof json?.updated_at === "string" ? json.updated_at : null);
      setNoteStatus("saved");
    } catch { setNoteStatus("error"); }
  };
  // debounce autosave 1.2s
  useEffect(() => {
    if (firstLoadRef.current) return;
    if (!fileId) return;
    if (saveTimer.current) window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => { autosaveNote(fileId, note); }, 1200);
    return () => { if (saveTimer.current) window.clearTimeout(saveTimer.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [note, fileId, userId]);

  /* -------- App states -------- */
  const [text, setText] = useState("");
  const [pdf, setPdf] = useState<File | null>(null);
  const [pdfText, setPdfText] = useState("");
  const [overview, setOverview] = useState("");
  const [keyPoints, setKeyPoints] = useState<string[]>([]);
  const [sections, setSections] = useState<Section[]>([]);
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
  const [questions, setQuestions] = useState<QuizItem[]>([]);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [score, setScore] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [qaInput, setQaInput] = useState("");
  const [qaAnswer, setQaAnswer] = useState("");

  const seenKeysRef = useRef<{ mcq: Set<string>; tf: Set<string> }>({ mcq: new Set(), tf: new Set() });
  const topicsRef = useRef<string[]>([]);

  const context = useMemo(() => {
    const t1 = text.trim(); const t2 = pdfText.trim();
    return [t1, t2].filter(Boolean).join("\n");
  }, [text, pdfText]);

  const countByType = (t: "mcq" | "tf") => questions.filter(q => q.type === t).length;
  const resetAllViews = () => { setError(null); setScore(null); setQuestions([]); setAnswers({}); setQaAnswer(""); seenKeysRef.current.mcq.clear(); seenKeysRef.current.tf.clear(); topicsRef.current = []; };
  const resetQuizOnly = () => { setScore(null); setQuestions([]); setAnswers({}); seenKeysRef.current.mcq.clear(); seenKeysRef.current.tf.clear(); topicsRef.current = []; };

  const ensureTopics = async () => {
    if (!context) return;
    if (topicsRef.current.length >= 10) return;
    try {
      const res = await fetch(`${API}/quiz/topics`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ context }) });
      const json = (await res.json()) as { topics?: string[] };
      if (Array.isArray(json.topics) && json.topics.length > 0) {
        const exist = new Set(topicsRef.current.map((t) => t.toLowerCase()));
        for (const t of json.topics) {
          const k = String(t).trim().toLowerCase();
          if (k && !exist.has(k)) { topicsRef.current.push(t); exist.add(k); }
        }
      }
    } catch { /* ignore */ }
  };

  // --- PDF Upload ---
  const uploadPdf = async (file: File | null) => {
    setPdf(file); setPdfText("");
    const fid = file ? (file.name.replace(/\.pdf$/i,"") || "pdf-file") : "manual";
    setFileId(fid);
    firstLoadRef.current = true;
    await loadNote(fid);

    if (!file) return;
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("pdf", file);
      const res = await fetch(`${API}/pdf/extract`, { method: "POST", body: fd });
      const json: unknown = await res.json();
      if (!res.ok) {
        const msg = hasDetail(json) && typeof json.detail === "string" ? json.detail : "อ่าน PDF ไม่สำเร็จ";
        throw new Error(msg);
      }
      const txt = (json as { text?: unknown }).text;
      setPdfText(typeof txt === "string" ? txt : "");
      resetQuizOnly();
      await ensureTopics();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  };

  // --- Summarize ---
  const summarize = async () => {
    if (!context) return;
    if (!pdf) { setFileId("manual"); firstLoadRef.current = true; await loadNote("manual"); }
    resetAllViews();
    setLoading(true);
    try {
      const res = await fetch(`${API}/summarize`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ context }) });
      const json: unknown = await res.json();
      if (!res.ok) {
        const msg = hasDetail(json) && typeof json.detail === "string" ? json.detail : "สรุปไม่สำเร็จ";
        throw new Error(msg);
      }
      const data = json as Partial<SummarizeResponse>;
      setOverview(typeof data.overview === "string" ? data.overview : "");
      setKeyPoints(Array.isArray(data.key_points) ? data.key_points.filter((x): x is string => typeof x === "string") : []);
      setSections(Array.isArray(data.sections) ? data.sections.filter((x): x is Section => !!x && typeof x === "object") : []);
      setDataPoints(Array.isArray(data.data_points) ? data.data_points.filter((x): x is DataPoint => !!x && typeof x === "object") : []);
      await ensureTopics();
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  };

  // --- Quiz generation ---
  const addMoreQuiz = async (type: "mcq" | "tf") => {
    if (!context) return;
    setError(null);
    const already = countByType(type);
    const remain = MAX_QUESTIONS - already;
    if (remain <= 0) return;
    const want = Math.min(BATCH_SIZE, remain);
    setLoading(true);
    await ensureTopics();
    const topicSlice = topicsRef.current.splice(0, want);

    try {
      let collected: QuizItem[] = [];
      for (let attempt = 0; attempt < 1 + MAX_RETRY && collected.length < want; attempt++) {
        const excludeTexts = [...questions.filter(q => q.type === type).map(q => q.question), ...collected.map(q => q.question)];
        const res = await fetch(`${API}/quiz/${type}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ context, n: want - collected.length, exclude: excludeTexts, topics: topicSlice.length ? topicSlice : undefined }),
        });
        const json: unknown = await res.json();
        if (!res.ok) {
          const msg = hasDetail(json) && typeof json.detail === "string" ? json.detail : "สร้างข้อสอบไม่สำเร็จ";
          throw new Error(msg);
        }
        const incoming = normalizeFromAPI(json).filter(q => q.type === type);
        const existingSameType = questions.filter(x => x.type === type);
        const bucket = seenKeysRef.current[type];
        const unique = incoming.filter((q) => {
          const k = keyFor(q);
          if (bucket.has(k)) return false;
          for (const e of existingSameType) if (similar(q.question, e.question) >= NEAR_DUP_TH) return false;
          for (const e of collected)       if (similar(q.question, e.question) >= NEAR_DUP_TH) return false;
          return true;
        });
        collected = [...collected, ...unique];
      }
      if (collected.length === 0) { setError(`ไม่พบข้อใหม่ของชนิด ${type.toUpperCase()} (อาจใกล้เคียงกับที่มีอยู่)`); return; }
      const batchReady = shuffleAndRemapBatch(collected.slice(0, want));
      setQuestions((prev) => [...prev, ...batchReady]);
      const bucket = seenKeysRef.current[type];
      batchReady.forEach((q) => bucket.add(keyFor(q)));
      setScore(null);
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  };

  // --- QA ---
  const askQA = async () => {
    if (!context || !qaInput.trim()) return;
    setLoading(true);
    setQaAnswer("");
    try {
      const res = await fetch(`${API}/qa`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ context, question: qaInput }) });
      const json: unknown = await res.json();
      if (!res.ok) {
        const msg = hasDetail(json) && typeof json.detail === "string" ? json.detail : "ถามไม่สำเร็จ";
        throw new Error(msg);
      }
      const ans = (json as { answer?: unknown }).answer;
      setQaAnswer(typeof ans === "string" ? ans : "");
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  };

  // --- Submit score ---
  const submit = () => {
    let correct = 0;
    questions.forEach((q, i) => { const u = (answers[i] || "").toLowerCase(); if (u === String(q.answer).toLowerCase()) correct++; });
    setScore(correct);
  };

  const isSubmitted = score !== null;
  const mcqCount = countByType("mcq");
  const tfCount  = countByType("tf");

  // โหลดโน้ตสำหรับ fileId เริ่มต้น
  useEffect(() => { firstLoadRef.current = true; loadNote(fileId); /* eslint-disable-next-line */}, [fileId, userId]);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800 bg-zinc-900/60 backdrop-blur sticky top-0 z-10">
        <div className="mx-auto max-w-5xl px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">EduGen — GPT Quiz Builder</h1>
            <p className="text-sm text-zinc-400">สรุป | ข้อสอบ | ถาม–ตอบ จากข้อความหรือ PDF</p>
          </div>
          <div className="flex items-center gap-3">
            {score !== null && (<Label>🏁 คะแนนรวม: <b className="text-emerald-400">{score}</b> / {questions.length}</Label>)}
            <PrimaryBtn onClick={() => setNoteOpen(true)} className="bg-zinc-700 hover:bg-zinc-600">📝 เปิดโน้ต</PrimaryBtn>
          </div>
        </div>
      </header>

      {/* Notes Modal */}
      <Modal
        open={noteOpen}
        onClose={() => setNoteOpen(false)}
        title="📝 โน้ตของฉัน"
        rightInfo={
          <>
            {noteStatus === "saving" && "กำลังบันทึก…"}
            {noteStatus === "saved" && (noteUpdatedAt ? `บันทึกล่าสุด: ${new Date(noteUpdatedAt).toLocaleString()}` : "บันทึกแล้ว")}
            {noteStatus === "error" && <span className="text-red-300">บันทึกล้มเหลว</span>}
          </>
        }
      >
        <textarea
          placeholder="จดสรุป/ประเด็นสำคัญ/ข้อสงสัย… (บันทึกอัตโนมัติ)"
          className="w-full h-60 rounded-xl bg-zinc-900 border border-zinc-800 px-3 py-3 outline-none focus:border-indigo-500"
          value={note}
          onChange={(e) => setNote(e.target.value)}
        />
        <div className="mt-2 text-xs text-zinc-500">
          * ระบบบันทึกอัตโนมัติและผูกโน้ตกับไฟล์ที่คุณใช้งานอยู่ (ถ้าอัปโหลด PDF จะใช้ชื่อไฟล์เป็นตัวระบุภายใน)
        </div>
      </Modal>

      <main className="mx-auto max-w-5xl px-6 py-6 space-y-6">
        {/* Input + Actions */}
        <Card>
          <textarea
            placeholder="พิมพ์เนื้อหาเพื่อใช้งานสรุป/ออกข้อสอบ หรือปล่อยว่างแล้วอัปโหลด PDF"
            className="w-full h-36 rounded-xl bg-zinc-900 border border-zinc-800 px-3 py-3 outline-none focus:border-indigo-500"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <PrimaryBtn onClick={summarize} disabled={loading}>สรุปจากข้อความ/ไฟล์</PrimaryBtn>

            <PrimaryBtn onClick={() => addMoreQuiz("mcq")} disabled={loading || mcqCount >= MAX_QUESTIONS}>
              {mcqCount >= MAX_QUESTIONS ? "MCQ ครบ 15 ข้อแล้ว" : `เพิ่มช้อยส์อีก ${BATCH_SIZE} ข้อ (${mcqCount} / ${MAX_QUESTIONS})`}
            </PrimaryBtn>

            <PrimaryBtn onClick={() => addMoreQuiz("tf")} disabled={loading || tfCount >= MAX_QUESTIONS}>
              {tfCount >= MAX_QUESTIONS ? "True/False ครบ 15 ข้อแล้ว" : `เพิ่มถูก/ผิดอีก ${BATCH_SIZE} ข้อ (${tfCount} / ${MAX_QUESTIONS})`}
            </PrimaryBtn>

            <PrimaryBtn onClick={async () => { topicsRef.current = []; await ensureTopics(); }} disabled={loading} className="bg-zinc-700 hover:bg-zinc-600">
              รีเฟรชหัวข้อ
            </PrimaryBtn>

            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => uploadPdf(e.target.files?.[0] || null)}
              className="text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-zinc-800 file:px-3 file:py-2 file:text-zinc-100 file:hover:bg-zinc-700 file:cursor-pointer"
            />
            {pdf && <span className="text-xs text-zinc-400">{pdf.name}</span>}
          </div>
          <div className="mt-2 text-xs text-zinc-400">
            * จำกัดสูงสุดชนิดละ {MAX_QUESTIONS} ข้อ — กดเพิ่มได้รอบละ {BATCH_SIZE} ข้อ · หัวข้อในคิว: {topicsRef.current.length}
          </div>
        </Card>

        {/* Error */}
        {error && <Card className="border-red-700/50 bg-red-900/20 text-red-200">⚠️ {error}</Card>}

        {/* Overview + Sections + Key points + Data points */}
        {(overview || keyPoints.length > 0 || sections.length > 0 || dataPoints.length > 0) && (
          <Card>
            <div className="mb-3"><h2 className="text-lg font-semibold">📘 ภาพรวมเนื้อหา</h2></div>
            {overview && <p className="text-sm leading-relaxed text-zinc-200 mb-4 whitespace-pre-line">{overview}</p>}
            {keyPoints.length > 0 && (
              <div className="mb-4">
                <h3 className="text-base font-semibold mb-2 text-indigo-400">✅ หัวใจสำคัญ (Key Points)</h3>
                <ul className="list-disc pl-6 space-y-1 text-sm">{keyPoints.map((p, i) => <li key={i}>{p}</li>)}</ul>
              </div>
            )}
            {sections.length > 0 && (
              <div className="mb-4">
                <h3 className="text-base font-semibold mb-2 text-indigo-400">🧩 หัวข้อสำคัญในไฟล์</h3>
                <ul className="space-y-2">
                  {sections.map((s, i) => (
                    <li key={i} className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-3 text-sm leading-snug">
                      <div className="font-medium">{s.title}</div>
                      <div className="text-zinc-300">{s.summary}</div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {dataPoints.length > 0 && (
              <div>
                <h3 className="text-base font-semibold mb-2 text-indigo-400">📏 ตัวเลข/ข้อมูลอ้างอิง</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead><tr className="text-left text-zinc-400"><th className="py-1 pr-4">รายการ</th><th className="py-1 pr-4">ค่า</th><th className="py-1 pr-4">หน่วย</th></tr></thead>
                    <tbody>{dataPoints.map((d, i) => (
                      <tr key={i} className="border-t border-zinc-800"><td className="py-2 pr-4">{d.label}</td><td className="py-2 pr-4">{d.value}</td><td className="py-2 pr-4">{d.unit || "-"}</td></tr>
                    ))}</tbody>
                  </table>
                </div>
              </div>
            )}
          </Card>
        )}

        {/* Q&A */}
        <Card>
          <div className="flex items-center gap-3">
            <input
              className="flex-1 rounded-xl bg-zinc-900 border border-zinc-800 px-3 py-2"
              placeholder="พิมพ์คำถามจากเนื้อหา…"
              value={qaInput}
              onChange={(e) => setQaInput(e.target.value)}
            />
            <PrimaryBtn onClick={askQA} disabled={loading}>ถามจากเนื้อหา</PrimaryBtn>
          </div>
          {qaAnswer && <div className="mt-3 rounded-lg border border-zinc-800 bg-zinc-900/60 p-3 text-sm">{qaAnswer}</div>}
        </Card>

        {/* Quiz */}
        {questions.length > 0 && (
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">แบบทดสอบ</h2>
              <Label>{questions.length} ข้อ</Label>
            </div>
            <ol className="space-y-4">
              {questions.map((q, idx) => {
                const selectedLetter = answers[idx];
                const hasSelected = !!selectedLetter;
                const isCorrect = hasSelected && String(selectedLetter).toLowerCase() === String(q.answer).toLowerCase();
                const headDotColor = hasSelected ? (isSubmitted ? (isCorrect ? "bg-emerald-400" : "bg-red-400") : "bg-zinc-500") : "bg-zinc-800";

                return (
                  <li key={idx} className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                    <div className="mb-3 font-medium flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <span aria-hidden className={`inline-block h-2.5 w-2.5 rounded-full ${headDotColor}`} />
                        <span>{idx + 1}. {q.question}</span>
                      </div>
                      {hasSelected && (<Label>คุณเลือก: <b className="text-zinc-100">{String(selectedLetter).toUpperCase()}</b></Label>)}
                    </div>

                    {q.type === "mcq" ? (
                      <div className="grid gap-2">
                        {(q.choices || []).map((c, i) => {
                          const letter = idxToLetter[i];
                          const selected = answers[idx] === letter;
                          const correctLetter = q.answer;
                          const isCorrectChoice = letter === correctLetter;
                          const wrongSelected = isSubmitted && selected && !isCorrectChoice;

                          let cls = "flex items-center gap-3 rounded-lg border px-3 py-2 cursor-pointer transition ";
                          if (!isSubmitted) cls += selected ? "border-emerald-500 bg-emerald-500/10" : "border-zinc-800 hover:bg-zinc-800/40";
                          else cls += isCorrectChoice ? "border-emerald-500 bg-emerald-500/10" : (wrongSelected ? "border-red-500 bg-red-500/10" : "border-zinc-800 opacity-70");

                          const Dot = (
                            <span className="relative inline-flex items-center justify-center mr-1.5">
                              <span className="h-3.5 w-3.5 rounded-full border border-zinc-600 bg-zinc-900" />
                              <span className={`absolute h-2 w-2 rounded-full transition ${selected ? "bg-black" : "bg-transparent"}`} />
                            </span>
                          );
                          const tailDot = !isSubmitted ? (selected ? "bg-zinc-400" : "") : (isCorrectChoice ? "bg-emerald-400" : (wrongSelected ? "bg-red-400" : ""));

                          return (
                            <label key={i} className={cls}>
                              <input type="radio" name={`q-${idx}`} value={letter} checked={selected} onChange={(e) => setAnswers((p) => ({ ...p, [idx]: e.target.value }))} className="sr-only" disabled={isSubmitted} />
                              {Dot}
                              <span><b>{letter})</b> {c}</span>
                              {tailDot && <span aria-hidden className={`ml-auto inline-block h-2.5 w-2.5 rounded-full ${tailDot}`} />}
                            </label>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="flex gap-3">
                        {["true", "false"].map((v) => {
                          const selected = answers[idx] === v;
                          const isCorrectChoice = q.answer.toLowerCase() === v;
                          const wrongSelected = isSubmitted && selected && !isCorrectChoice;

                          let cls = "flex items-center gap-3 rounded-lg border px-3 py-2 cursor-pointer transition ";
                          if (!isSubmitted) cls += selected ? "border-emerald-500 bg-emerald-500/10" : "border-zinc-800 hover:bg-zinc-800/40";
                          else cls += isCorrectChoice ? "border-emerald-500 bg-emerald-500/10" : (wrongSelected ? "border-red-500 bg-red-500/10" : "border-zinc-800 opacity-70");

                          const Dot = (
                            <span className="relative inline-flex items-center justify-center mr-1.5">
                              <span className="h-3.5 w-3.5 rounded-full border border-zinc-600 bg-zinc-900" />
                              <span className={`absolute h-2 w-2 rounded-full transition ${selected ? "bg-black" : "bg-transparent"}`} />
                            </span>
                          );
                          const tailDot = !isSubmitted ? (selected ? "bg-zinc-400" : "") : (isCorrectChoice ? "bg-emerald-400" : (wrongSelected ? "bg-red-400" : ""));

                          return (
                            <label key={v} className={cls}>
                              <input type="radio" name={`q-${idx}`} value={v} checked={selected} onChange={(e) => setAnswers((p) => ({ ...p, [idx]: e.target.value }))} className="sr-only" disabled={isSubmitted} />
                              {Dot}
                              <span>{v.toUpperCase()}</span>
                              {tailDot && <span aria-hidden className={`ml-auto inline-block h-2.5 w-2.5 rounded-full ${tailDot}`} />}
                            </label>
                          );
                        })}
                      </div>
                    )}

                    {isSubmitted && (
                      <div className="mt-3 text-sm">
                        เฉลย: <b className="text-emerald-400">{String(q.answer).toUpperCase()}</b>
                        {q.explain && <span className="text-zinc-400"> — {q.explain}</span>}
                      </div>
                    )}
                  </li>
                );
              })}
            </ol>

            <div className="mt-5 flex items-center justify-between">
              <div className="text-xs text-zinc-400">* ระบบจะเฉลยและแสดงคะแนนหลังจากกด “ตรวจคะแนน”</div>
              <PrimaryBtn onClick={submit} disabled={isSubmitted}>ตรวจคะแนน</PrimaryBtn>
            </div>

            {isSubmitted && (
              <div className="mt-4 rounded-xl border border-emerald-600/40 bg-emerald-900/20 p-4 text-emerald-300">
                คะแนนรวม: <b className="text-emerald-400">{score}</b> / {questions.length}
              </div>
            )}
          </Card>
        )}
      </main>
    </div>
  );
}
