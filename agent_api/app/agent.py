import os
import re
import httpx
import hashlib
from typing import List, Dict, Tuple

from .tools import Tools
from .query_planner import plan_queries, refine_queries
from .rerank import rerank
from .evidence import build_evidence_pack

SYSTEM_PROMPT = """You are an agentic RAG assistant.
You will be given:
- a running SUMMARY (persistent memory)
- NOTES (private working memory)
- RECENT CHAT history (sliding window)
- retrieved CONTEXT chunks with citations

Rules:
- Answer the user normally.
- Always cite sources like [1], [2] when using retrieved chunks.
- If you extract or infer structure (tables, counts), do so explicitly.
- Never reveal private NOTES verbatim. Use them only to stay consistent.

WICHTIG:
Du HAST Zugriff auf die Nutzerdaten über den Block "RETRIEVED CONTEXT".
Dieser Kontext stammt aus lokalen Dateien des Nutzers und ist bereits für dich geladen.
Du darfst daraus konkrete Dateinamen/Pfade/Details nennen, sofern sie im Kontext/Metadaten stehen.
Du darfst NICHT sagen, dass du keinen Zugriff auf Daten hast, wenn RETRIEVED CONTEXT nicht leer ist.
Wenn Details fehlen, stelle Rückfragen oder sage präzise, was im Kontext fehlt.
Wenn Treffer thematisch nahe sind (z.B. Offerte statt Rechnung), zeige sie und frage nach Präzisierung (Datum/Projekt), statt zu sagen, du hättest keinen Kontext.
BILINGUAL: "Rechnung" = "invoice" = "bill" = "receipt". Behandle englische und deutsche Begriffe als äquivalent.

EMAIL-REGELN (sehr wichtig):
E-Mail Details (Betreff, Datum, Betrag, Anhänge, Absender/Empfänger) dürfen nur genannt werden,
wenn sie im RETRIEVED CONTEXT oder in Metadatenfeldern (email_*) vorhanden sind.
Wenn solche Felder fehlen: sage "Metadaten fehlen in der Indexierung" und zeige die Top-Hits.
Erfinde niemals Betreff/Beträge/Anhänge.

Du arbeitest agentisch:
plane Suchvarianten,
nutze Retrieval,
lies Belege,
beantworte ausschließlich aus Belegen.
Wenn Belege fehlen: sage exakt "Nicht in den Dokumenten gefunden."

SPRACHE:
Antworte standardmäßig auf Deutsch.
Wenn Nutzer explizit Englisch verlangt ("in english", "auf englisch") -> antworte Englisch.

Output format requirement:
- You MAY include a private notes block wrapped in:
  <NOTES>...</NOTES>
  Then provide the user-facing answer AFTER that block.
- The server will strip <NOTES> before showing to the user.
"""

def approx_tokens(s: str) -> int:
    # fast heuristic: 1 token ~ 4 chars (roughly)
    if not s:
        return 0
    return max(1, len(s) // 4)

def clamp_tokens(text: str, max_tokens: int) -> str:
    # naive clamp by chars
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]  # keep tail (most recent)

def strip_notes(model_output: str) -> Tuple[str, str]:
    """
    Returns (notes, answer_without_notes).
    """
    if not model_output:
        return "", ""
    m = re.search(r"<NOTES>(.*?)</NOTES>", model_output, re.DOTALL | re.IGNORECASE)
    notes = m.group(1).strip() if m else ""
    answer = re.sub(r"<NOTES>.*?</NOTES>", "", model_output, flags=re.DOTALL | re.IGNORECASE).strip()
    return notes, answer

def _hits_preview(hits, n=5):
    lines=[]
    for i,h in enumerate(hits[:n],1):
        md=h.get("metadata") or {}
        p=md.get("original_path") or md.get("file_path") or "unknown"
        sn=(h.get("text") or "").replace("\n"," ")[:160]
        lines.append(f"{i}. {p} — {sn} …")
    return "\n".join(lines)

def _weak_evidence(hits):
    # simple gate: too few hits or all distances bad (if present)
    if not hits or len(hits) < 3:
        return True
    dists=[h.get("distance") for h in hits if h.get("distance") is not None]
    if dists:
        # Chroma distance: lower better; >1.2 often weak in many setups
        if min(dists) > 1.2:
            return True
    return False

class Agent:
    def __init__(self):
        self.ollama_base = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.llm_model = os.getenv("LLM_MODEL", "qwen2.5:14b")

        self.tools = Tools()

        self.context_max_tokens = int(os.getenv("CONTEXT_MAX_TOKENS", "12000"))
        self.summary_tokens = int(os.getenv("CONTEXT_SUMMARY_TOKENS", "1200"))
        self.recent_tokens = int(os.getenv("CONTEXT_RECENT_TOKENS", "7000"))
        self.notes_tokens = int(os.getenv("NOTES_MAX_TOKENS", "600"))
        self.summary_update_trigger = int(os.getenv("SUMMARY_UPDATE_TRIGGER_TOKENS", "9000"))

    async def llm(self, messages: List[Dict], temperature: float = 0.2) -> str:
        async with httpx.AsyncClient(timeout=240) as client:
            r = await client.post(
                f"{self.ollama_base}/api/chat",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "options": {"temperature": temperature},
                    "stream": False,
                },
            )
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"]

    def build_recent_history(self, raw_messages: List[Dict], budget_tokens: int) -> List[Dict]:
        """
        Takes the full chat messages (role/content) and keeps as many from the end as fit.
        Keeps system messages out (we provide our own).
        """
        # normalize
        msgs = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in raw_messages]
        msgs = [m for m in msgs if m["role"] in ("user", "assistant") and (m["content"] or "").strip()]

        kept = []
        used = 0
        # take from the end backwards
        for m in reversed(msgs):
            t = approx_tokens(m["content"])
            if used + t > budget_tokens:
                break
            kept.append(m)
            used += t
        kept.reverse()
        return kept

    async def maybe_update_summary(self, summary: str, recent_msgs: List[Dict]) -> str:
        """
        Update summary if recent history is large enough.
        """
        total = sum(approx_tokens(m["content"]) for m in recent_msgs)
        if total < self.summary_update_trigger:
            return summary or ""

        # summarize only the last part to keep it cheap
        tail_text = "\n".join([f'{m["role"].upper()}: {m["content"]}' for m in recent_msgs[-12:]])
        prompt = [
            {"role": "system", "content": "You update a running conversation summary for later retrieval. Be concise but preserve key facts, decisions, entities, constraints, and pending tasks."},
            {"role": "user", "content": f"CURRENT SUMMARY:\n{summary}\n\nNEW DIALOGUE (latest):\n{tail_text}\n\nWrite an UPDATED SUMMARY (no more than ~{self.summary_tokens} tokens)."}
        ]
        new_sum = await self.llm(prompt, temperature=0.1)
        return clamp_tokens(new_sum.strip(), self.summary_tokens)

    def _top_hits_debug(self, hits, n=5):
        lines=[]
        for i,h in enumerate(hits[:n],1):
            md=h.get("metadata") or {}
            p=md.get("original_path") or md.get("file_path") or "unknown"
            s=(h.get("text") or "").replace("\n"," ")[:160]
            lines.append(f"[{i}] {p} — {s} …")
        return "\n".join(lines)

    async def answer(
        self,
        user_text: str,
        raw_messages: List[Dict],
        summary: str,
        notes: str,
    ) -> Tuple[str, str, str]:
        """
        Returns (answer, new_summary, new_notes)
        """
        # AGENTIC LOOP: PLAN → RETRIEVE → RERANK → EVIDENCE → ANSWER/EXTRACT
        plan = await plan_queries(self.llm, user_text)
        mode = plan.get("mode","rag")
        queries = plan["queries"]
        must_include = plan.get("must_include", [])

        final_hits = []
        final_context = ""
        final_sources = []

        for attempt in range(2):  # max 2 iterations
            hits = self.tools.search_multi(queries, top_k_each=8, max_total=30)
            hits = self.tools.filter_must_include(hits, must_include)
            if hits:
                hits = rerank(queries[0], hits, top_n=12)
                context, sources = build_evidence_pack(hits, max_sources=6, max_chars_per_source=1600)
            else:
                context, sources = "", []

            # if good enough: break
            if hits and not _weak_evidence(hits):
                final_hits, final_context, final_sources = hits, context, sources
                break

            # attempt 0 -> refine based on preview and retry
            if attempt == 0:
                preview = _hits_preview(hits, n=5) if hits else "NO HITS"
                plan2 = await refine_queries(self.llm, user_text, preview)
                mode = plan2.get("mode", mode)
                # merge new queries + keep old first
                newq = plan2.get("queries", [])
                merged=[]
                for q in (queries + newq):
                    q=(q or "").strip()
                    if q and q not in merged:
                        merged.append(q)
                queries = merged[:12]
                must_include = plan2.get("must_include", must_include)
                continue

        # if still nothing:
        if not final_hits:
            return ("Nicht in den Dokumenten gefunden.", summary, notes)

        # OPTIONAL: Wenn attempt 2 immer noch weak evidence, dann lieber "nicht gefunden"
        if _weak_evidence(final_hits):
            return ("Nicht in den Dokumenten gefunden.", summary, notes)

        # now answer using final_context
        context = final_context
        hits = final_hits

        # enforce: no "kein zugriff" and no hallucination
        user_task = user_text.strip()
        if mode == "extract":
            user_task += "\n\nGib die Antwort als Tabelle oder strukturierte Liste. Extrahiere nur, was im Kontext steht."

        messages = [
          {"role":"system","content": SYSTEM_PROMPT},
          {"role":"system","content": "RETRIEVED CONTEXT (zitiere als [n]):\n" + context},
          {"role":"user","content":
              f"Aufgabe: {user_task}\n\n"
              f"Regeln:\n"
              f"- Antworte nur aus RETRIEVED CONTEXT.\n"
              f"- Nenne konkrete Pfade/Dateien nur, wenn sie in [n] stehen.\n"
              f"- Jede zentrale Aussage braucht mind. eine Quelle [n].\n"
              f"- Wenn es nicht im Kontext steht: sage exakt 'Nicht in den Dokumenten gefunden.'\n"
          }
        ]
        out = await self.llm(messages, temperature=0.2)
        notes2, answer = strip_notes(out)
        answer = (answer or "").strip()

        # anti-refusal retry
        low = answer.lower()
        if hits and ("kein zugriff" in low or "keinen direkten zugriff" in low or "i don't have access" in low):
            messages.append({"role":"system","content":"Du HAST Zugriff auf RETRIEVED CONTEXT. Entferne jede Zugriff-Ausrede. Antworte nun konkret mit Quellen [n]."})
            out2 = await self.llm(messages, temperature=0.1)
            _, a2 = strip_notes(out2)
            answer = (a2 or "").strip()

        # if still no citations and not "Nicht gefunden", force one retry
        if ("Nicht in den Dokumenten gefunden." not in answer) and ("[" not in answer):
            messages.append({"role":"system","content":"Pflicht: setze Quellen [1], [2] ... bei den relevanten Aussagen. Wiederhole."})
            out3 = await self.llm(messages, temperature=0.1)
            _, a3 = strip_notes(out3)
            answer = (a3 or "").strip()

        # Add clickable file:// links to sources
        if final_sources:
            src_lines = []
            for s in final_sources:
                n = s.get("n")
                display_path = s.get("display_path", s.get("path", ""))
                url = s.get("local_url","")
                if url and display_path:
                    src_lines.append(f"[{n}] [{display_path}]({url})")
                else:
                    src_lines.append(f"[{n}] {display_path}")
            answer = answer.rstrip() + "\n\nQuellen (lokal):\n" + "\n\n".join(src_lines)

        return (answer, summary, notes)
