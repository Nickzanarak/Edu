"use client";
import { useMemo, useState } from "react";
import type { ReactNode, ButtonHTMLAttributes } from "react";

/* ---------- Types ---------- */
type QuizItem = {
  type: "mcq" | "tf";
  question: string;
  choices?: string[];
  answer: string; // "ก" | "ข" | "ค" | "ง" | "true" | "false"
  explain?: string;
};

type Section = { title: string; summary: string };
type DataPoint = { label: string; value: string; unit?: string };
type SummarizeResponse = {
  overview: string;
  key_points: string[];
  sections: Section[];
  data_points: DataPoint[];
};

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

const stripChoiceLabel = (s: string) =>
  String(s).replace(/^\s*[กขคง]\)\s*/i, "").trim();

const toStr = (v: unknown) =>
  (typeof v === "string" ? v : String(v ?? "")).trim();

function shuffle<T>(arr: T[]) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

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
      const choices: string[] =
        mapped.length < 4
          ? [
              ...mapped,
              ...Array.from(
                { length: 4 - mapped.length },
                (_, i) => `ตัวเลือกที่ ${mapped.length + i + 1}`
              ),
            ]
          : mapped;

      if (!["ก", "ข", "ค", "ง"].includes(answer)) {
        const num = Number(answer);
        if (Number.isFinite(num)) {
          const idx = num >= 1 && num <= 4 ? num - 1 : num;
          if (idx >= 0 && idx < 4) answer = idxToLetter[idx];
        }
        if (!["ก", "ข", "ค", "ง"].includes(answer)) {
          const idx = choices.findIndex(
            (c) => c.replace(/\s+/g, "") === answer.replace(/\s+/g, "")
          );
          answer = idx >= 0 ? idxToLetter[idx] : "ก";
        }
      }
      out.push({ type: "mcq", question, choices, answer, explain });
      continue;
    }

    if (!["true", "false"].includes(answer)) {
      const trueSet = new Set(["true", "t", "1", "yes", "y", "จริง", "ถูก"]);
      const falseSet = new Set(["false", "f", "0", "no", "n", "เท็จ", "ผิด"]);
      if (trueSet.has(answer)) answer = "true";
      else if (falseSet.has(answer)) answer = "false";
      else answer = "false";
    }
    out.push({ type: "tf", question, answer, explain });
  }
  return out;
}

function shuffleAndRemap(qs: QuizItem[]): QuizItem[] {
  return shuffle(qs).map((q) => {
    if (q.type !== "mcq" || !q.choices?.length) return q;
    const original = [...q.choices];
    const correctIdx = letterToIdx[q.answer] ?? -1;
    const correctText =
      correctIdx >= 0 && correctIdx < original.length ? original[correctIdx] : null;

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
const PrimaryBtn = ({
  children, className, ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { children: ReactNode }) => (
  <button
    {...props}
    className={`px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium shadow-sm disabled:opacity-50 ${className}`}
  >
    {children}
  </button>
);

/* ---------- Page ---------- */
export default function Home() {
  const API = process.env.NEXT_PUBLIC_API || "http://127.0.0.1:8000";

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

  const context = useMemo(() => {
    const t1 = text.trim();
    const t2 = pdfText.trim();
    return [t1, t2].filter(Boolean).join("\n");
  }, [text, pdfText]);

  const resetView = () => {
    setError(null);
    setScore(null);
    setQuestions([]);
    setAnswers({});
    setQaAnswer("");
  };

  // --- PDF Upload ---
  const uploadPdf = async (file: File | null) => {
    setPdf(file);
    setPdfText("");
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
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  // --- Summarize ---
  const summarize = async () => {
    if (!context) return;
    resetView();
    setLoading(true);
    try {
      const res = await fetch(`${API}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context }),
      });
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
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  // --- Quiz generation ---
  const genQuiz = async (type: "mcq" | "tf") => {
    if (!context) return;
    resetView();
    setLoading(true);
    try {
      const res = await fetch(`${API}/quiz/${type}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context, n: 5 }),
      });
      const json: unknown = await res.json();
      if (!res.ok) {
        const msg = hasDetail(json) && typeof json.detail === "string" ? json.detail : "สร้างข้อสอบไม่สำเร็จ";
        throw new Error(msg);
      }
      const items = normalizeFromAPI(json);
      setQuestions(shuffleAndRemap(items));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  // --- QA ---
  const askQA = async () => {
    if (!context || !qaInput.trim()) return;
    setLoading(true);
    setQaAnswer("");
    try {
      const res = await fetch(`${API}/qa`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context, question: qaInput }),
      });
      const json: unknown = await res.json();
      if (!res.ok) {
        const msg = hasDetail(json) && typeof json.detail === "string" ? json.detail : "ถามไม่สำเร็จ";
        throw new Error(msg);
      }
      const ans = (json as { answer?: unknown }).answer;
      setQaAnswer(typeof ans === "string" ? ans : "");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  // --- Submit score ---
  const submit = () => {
    let correct = 0;
    questions.forEach((q, i) => {
      const u = (answers[i] || "").toLowerCase();
      if (u === String(q.answer).toLowerCase()) correct++;
    });
    setScore(correct);
  };

  const isSubmitted = score !== null;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <header className="border-b border-zinc-800 bg-zinc-900/60 backdrop-blur sticky top-0 z-10">
        <div className="mx-auto max-w-5xl px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">EduGen — GPT Quiz Builder</h1>
            <p className="text-sm text-zinc-400">สรุป | ข้อสอบ | ถาม–ตอบ จากข้อความหรือ PDF</p>
          </div>
          {score !== null && (
            <Label>🏁 คะแนนรวม: <b className="text-emerald-400">{score}</b> / {questions.length}</Label>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-6 space-y-6">
        {/* Input + Actions */}
        <Card>
          <textarea
            placeholder="พิมพ์เนื้อหาเพื่อใช้งานสรุป/ออกข้อสอบ หรือปล่อยว่างแล้วอัปโหลด PDF"
            className="w-full h-36 rounded-xl bg-zinc-900 border border-zinc-800 px-3 py-3 outline-none focus:border-indigo-500"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="mt-3 flex flex-wrap gap-3">
            <PrimaryBtn onClick={summarize} disabled={loading}>สรุปจากข้อความ/ไฟล์</PrimaryBtn>
            <PrimaryBtn onClick={() => genQuiz("mcq")} disabled={loading}>สร้างข้อสอบแบบช้อยส์ </PrimaryBtn>
            <PrimaryBtn onClick={() => genQuiz("tf")} disabled={loading}>สร้างข้อสอบแบบ ถูก/ผิด </PrimaryBtn>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => uploadPdf(e.target.files?.[0] || null)}
              className="text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-zinc-800 file:px-3 file:py-2 file:text-zinc-100 file:hover:bg-zinc-700 file:cursor-pointer"
            />
            {pdf && <span className="text-xs text-zinc-400">{pdf.name}</span>}
          </div>
        </Card>

        {/* Error */}
        {error && <Card className="border-red-700/50 bg-red-900/20 text-red-200">⚠️ {error}</Card>}

        {/* Overview + Sections + Key points + Data points */}
        {(overview || keyPoints.length > 0 || sections.length > 0 || dataPoints.length > 0) && (
          <Card>
            <div className="mb-3">
              <h2 className="text-lg font-semibold">📘 ภาพรวมเนื้อหา</h2>
            </div>
            {overview && <p className="text-sm leading-relaxed text-zinc-200 mb-4 whitespace-pre-line">{overview}</p>}
            {keyPoints.length > 0 && (
              <div className="mb-4">
                <h3 className="text-base font-semibold mb-2 text-indigo-400">✅ หัวใจสำคัญ (Key Points)</h3>
                <ul className="list-disc pl-6 space-y-1 text-sm">
                  {keyPoints.map((p, i) => <li key={i}>{p}</li>)}
                </ul>
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
                    <thead>
                      <tr className="text-left text-zinc-400">
                        <th className="py-1 pr-4">รายการ</th>
                        <th className="py-1 pr-4">ค่า</th>
                        <th className="py-1 pr-4">หน่วย</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dataPoints.map((d, i) => (
                        <tr key={i} className="border-t border-zinc-800">
                          <td className="py-2 pr-4">{d.label}</td>
                          <td className="py-2 pr-4">{d.value}</td>
                          <td className="py-2 pr-4">{d.unit || "-"}</td>
                        </tr>
                      ))}
                    </tbody>
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
                const isCorrect =
                  hasSelected &&
                  String(selectedLetter).toLowerCase() === String(q.answer).toLowerCase();

                const headDotColor = hasSelected
                  ? isSubmitted
                    ? isCorrect
                      ? "bg-emerald-400"
                      : "bg-red-400"
                    : "bg-zinc-500"
                  : "bg-zinc-800";

                return (
                  <li key={idx} className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                    <div className="mb-3 font-medium flex items-center justify-between gap-3">
                      {/* จุดนำหน้าหัวข้อ + ข้อความคำถาม */}
                      <div className="flex items-center gap-2">
                        <span aria-hidden className={`inline-block h-2.5 w-2.5 rounded-full ${headDotColor}`} />
                        <span>{idx + 1}. {q.question}</span>
                      </div>

                      {hasSelected && (
                        <Label>คุณเลือก: <b className="text-zinc-100">{String(selectedLetter).toUpperCase()}</b></Label>
                      )}
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
                          if (!isSubmitted) {
                            cls += selected ? "border-emerald-500 bg-emerald-500/10" : "border-zinc-800 hover:bg-zinc-800/40";
                          } else {
                            if (isCorrectChoice) cls += "border-emerald-500 bg-emerald-500/10";
                            else if (wrongSelected) cls += "border-red-500 bg-red-500/10";
                            else cls += "border-zinc-800 opacity-70";
                          }

                          // จุดดำ ๆ ด้านซ้าย (custom radio)
                          const Dot = (
                            <span className="relative inline-flex items-center justify-center mr-1.5">
                              <span className="h-3.5 w-3.5 rounded-full border border-zinc-600 bg-zinc-900" />
                              <span className={`absolute h-2 w-2 rounded-full transition ${selected ? "bg-black" : "bg-transparent"}`} />
                            </span>
                          );

                          // จุดสรุปด้านขวา (เขียว/แดงหลังส่ง)
                          const tailDot =
                            !isSubmitted
                              ? selected
                                ? "bg-zinc-400"
                                : ""
                              : isCorrectChoice
                              ? "bg-emerald-400"
                              : wrongSelected
                              ? "bg-red-400"
                              : "";

                          return (
                            <label key={i} className={cls}>
                              <input
                                type="radio"
                                name={`q-${idx}`}
                                value={letter}
                                checked={selected}
                                onChange={(e) => setAnswers((p) => ({ ...p, [idx]: e.target.value }))}
                                className="sr-only" // ซ่อน radio จริง
                                disabled={isSubmitted}
                              />
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
                          if (!isSubmitted) {
                            cls += selected ? "border-emerald-500 bg-emerald-500/10" : "border-zinc-800 hover:bg-zinc-800/40";
                          } else {
                            if (isCorrectChoice) cls += "border-emerald-500 bg-emerald-500/10";
                            else if (wrongSelected) cls += "border-red-500 bg-red-500/10";
                            else cls += "border-zinc-800 opacity-70";
                          }

                          const Dot = (
                            <span className="relative inline-flex items-center justify-center mr-1.5">
                              <span className="h-3.5 w-3.5 rounded-full border border-zinc-600 bg-zinc-900" />
                              <span className={`absolute h-2 w-2 rounded-full transition ${selected ? "bg-black" : "bg-transparent"}`} />
                            </span>
                          );

                          const tailDot =
                            !isSubmitted
                              ? selected
                                ? "bg-zinc-400"
                                : ""
                              : isCorrectChoice
                              ? "bg-emerald-400"
                              : wrongSelected
                              ? "bg-red-400"
                              : "";

                          return (
                            <label key={v} className={cls}>
                              <input
                                type="radio"
                                name={`q-${idx}`}
                                value={v}
                                checked={selected}
                                onChange={(e) => setAnswers((p) => ({ ...p, [idx]: e.target.value }))}
                                className="sr-only"
                                disabled={isSubmitted}
                              />
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
