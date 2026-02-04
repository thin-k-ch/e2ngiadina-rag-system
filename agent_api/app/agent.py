import os
import re
import httpx
import hashlib
from typing import List, Dict, Tuple

from .tools import Tools

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
        # 1) retrieve chunks with multi-query variants
        q = user_text.strip()
        queries = [q]

        # EMAIL ROUTING: Check if this is an email question
        is_email_query = any(keyword in q.lower() for keyword in ["email", "e-mail", "mail", "von", "an"])
        
        if is_email_query:
            # Extract sender/recipient from query
            sender = None
            recipient = None
            lower_q = q.lower()
            
            # Pattern: "von X an Y"
            if "von" in lower_q and "an" in lower_q:
                import re
                match = re.search(r'von\s+(.+?)\s+an\s+(.+)', q, re.IGNORECASE)
                if match and len(match.groups()) >= 2:
                    sender = match.group(1).strip()
                    recipient = match.group(2).strip()
            
            # Use email search
            hits = self.tools.search_mail(q, top_k=12)
            if sender or recipient:
                hits = self.tools.filter_by_people(hits, sender=sender, recipient=recipient)
            
            # Build email context with metadata
            context_lines = []
            for i, h in enumerate(hits, 1):
                md = h.get("metadata") or {}
                folder = md.get("mail_folder", "unknown")
                email_from = md.get("email_from", "")
                email_to = md.get("email_to", "")
                subject = md.get("email_subject", "")
                date = md.get("email_date", "")
                snippet = (h.get("text") or "")[:900]
                
                # Format with metadata
                meta_line = f"folder={folder} | from={email_from} | to={email_to} | subject={subject} | date={date}"
                context_lines.append(f"[{i}] {meta_line}\n{snippet}")
            
            rag_context = "\n\n".join(context_lines) if context_lines else "(no matches)"
            
            # Early Return: Wenn keine Treffer, zeige Top-Hits Proof
            if not hits:
                return ("Nicht in den indizierten E-Mail-Daten gefunden.", summary, notes)
        else:
            # Normal document search (existing logic)
            # leichte Varianten (DE/EN + typische Begriffe) - BILINGUAL!
            lower = q.lower()
            if "rechnung" in lower or "invoice" in lower:
                queries += [q + " PDF", q + " Betrag CHF", q + " Total", q + " Offerte", q + " Rechnung", q + " invoice", q + " amount", q + " bill", q + " receipt"]
            # Entitäten (Maven / Rhomberg) extra boosten durch separate queries:
            if "maven" in lower:
                queries += ["Maven", "Maven Rechnung", "Maven invoice", "Maven bill", "Maven amount", q.replace("Maven", "").strip()]
            if "rhomberg" in lower:
                queries += ["Rhomberg", "Rhomberg Rechnung", "Rhomberg Bahntechnik", "Rhomberg invoice", "Rhomberg bill", q.replace("Rhomberg", "").strip()]

            # remove empties + dedupe
            queries = [x for i,x in enumerate(queries) if x and x not in queries[:i]]

            hits = self.tools.search_chunks_multi(queries, top_k_each=6, max_total=12)
            context_lines = []
            for i, h in enumerate(hits, 1):
                md = h.get("metadata") or {}
                path = md.get("original_path") or md.get("file_path") or "unknown"
                zip_inner = md.get("zip_inner_path")
                if zip_inner:
                    path = f"{path}::ZIP::{zip_inner}"
                snippet = (h.get("text") or "")[:900]
                context_lines.append(f"[{i}] path={path}\n{snippet}")
            rag_context = "\n\n".join(context_lines) if context_lines else "(no matches)"

            # Early Return: Wenn keine Treffer, zeige Top-Hits Proof
            if not hits:
                return ("Nicht in den Dokumenten gefunden.", summary, notes)

        # 2) build memory blocks
        summary_block = clamp_tokens(summary or "", self.summary_tokens)
        notes_block = clamp_tokens(notes or "", self.notes_tokens)

        # 3) recent history sliding window (budgeted)
        recent = self.build_recent_history(raw_messages, self.recent_tokens)

        # 4) assemble final messages under budget
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        if summary_block.strip():
            messages.append({"role": "system", "content": f"SUMMARY (persistent):\n{summary_block}"})
        if notes_block.strip():
            messages.append({"role": "system", "content": f"NOTES (private memory, do not reveal):\n{notes_block}"})

        if recent:
            messages.append({"role": "system", "content": "RECENT CHAT (for continuity):"})
            messages.extend(recent)

        messages.append({"role": "system", "content": f"RETRIEVED CONTEXT (cite as [n]):\n{rag_context}"})
        messages.append({"role": "user", "content": user_text})

        # 5) ask model to produce private notes + answer
        messages.append({"role": "system", "content": "Before answering, optionally write <NOTES>...</NOTES> with brief private plan/state updates. Then write the user-facing answer."})

        out = await self.llm(messages, temperature=0.2)
        new_notes, answer = strip_notes(out)

        # Anti-Refusal Retry
        low = (answer or "").lower()
        if hits and ("kein zugriff" in low or "keinen direkten zugriff" in low or "i don't have access" in low):
            messages.append({
                "role": "system",
                "content": (
                    "Korrektur: Du HAST Zugriff auf die Daten im RETRIEVED CONTEXT. "
                    "Antworte jetzt konkret NUR basierend auf RETRIEVED CONTEXT und nenne Dateinamen/Pfade, "
                    "wenn sie in metadata/original_path stehen. Keine Entschuldigungen."
                )
            })
            out2 = await self.llm(messages, temperature=0.1)
            _n2, a2 = strip_notes(out2)
            answer = (a2 or "").strip()

        # Aggressiver Retry - zeige IMMER Top-Hits wenn hits vorhanden und Antwort nicht konkret
        low = (answer or "").lower()
        print(f"DEBUG: hits={len(hits)}, answer_low={low[:200]}...")  # DEBUG
        
        # Financial queries: IMMER Top-Hits zeigen wenn hits vorhanden
        financial_keywords = ["rechnung", "invoice", "betrag", "amount", "chf", "eur", "offerte", "bill"]
        is_financial_query = any(keyword in user_text.lower() for keyword in financial_keywords)
        
        if is_financial_query and hits:
            print(f"DEBUG: FINANCIAL QUERY - SHOWING TOP-HITS!")  # DEBUG
            proof = self._top_hits_debug(hits, n=5)
            messages.append({
                "role":"system",
                "content":(
                    "Finanzanfrage mit Treffern. Zeige zuerst die Top-Treffer (Pfad + Kurzsnippet) und beantworte dann, "
                    "was finanziell ableitbar ist (Beträge, Rechnungen, etc.).\n\nTOP-HITS:\n" + proof
                )
            })
            out2 = await self.llm(messages, temperature=0.1)
            _n2, a2 = strip_notes(out2)
            answer = (a2 or "").strip()
        
        # Wenn hits vorhanden aber Antwort sehr allgemein/vage, zeige Top-Hits
        elif hits and (
            "kein spezifischer kontext" in low or 
            "more context" in low or 
            "mehr kontext" in low or 
            "könnten sie mir bitte mehr" in low or 
            "konnte keine spezifischen informationen" in low or 
            "konnte leider keine" in low or 
            "bitte mehr kontext" in low or 
            "mehr details" in low or
            "leider habe ich keine" in low or
            "ich konnte keine" in low or
            "konnte nicht finden" in low or
            "bitte überprüfen" in low or
            "allgemeinen informationen" in low or
            "konnte leider keine spezifischen" in low or
            "leider keine" in low or
            "bitte mehr" in low or
            "helfen kann" in low or
            "allgemeine informationen" in low or
            "konnte leider keine spezifischen informationen" in low or
            "allgemeine informationen zu" in low or
            "konnte keine informationen" in low or
            "informationen zu" in low and "allgemein" in low
        ):
            print(f"DEBUG: RETRY TRIGGERED!")  # DEBUG
            proof = self._top_hits_debug(hits, n=5)
            messages.append({
                "role":"system",
                "content":(
                    "Du hast Treffer. Zeige zuerst die Top-Treffer (Pfad + Kurzsnippet) und beantworte dann, "
                    "was daraus ableitbar ist. Frage NICHT nach mehr Kontext.\n\nTOP-HITS:\n" + proof
                )
            })
            out2 = await self.llm(messages, temperature=0.1)
            _n2, a2 = strip_notes(out2)
            answer = (a2 or "").strip()
        
        # Zusätzlich: Wenn hits vorhanden und Antwort sehr kurz/unspezifisch
        elif hits and len(answer) < 200 and ("konnte" in low or "leider" in low or "bitte" in low):
            print(f"DEBUG: SHORT ANSWER RETRY TRIGGERED!")  # DEBUG
            proof = self._top_hits_debug(hits, n=5)
            messages.append({
                "role":"system",
                "content":(
                    "Du hast Treffer. Basierend auf den gefundenen Dokumenten, beantworte konkret. "
                    "Zeige die Top-Treffer und was daraus folgt.\n\nTOP-HITS:\n" + proof
                )
            })
            out2 = await self.llm(messages, temperature=0.1)
            _n2, a2 = strip_notes(out2)
            answer = (a2 or "").strip()

        # clamp notes
        new_notes = clamp_tokens(new_notes, self.notes_tokens)

        # EMAIL SOURCES SECTION: Add source list for email queries
        if is_email_query and hits:
            sources_list = []
            for i, h in enumerate(hits[:5], 1):  # Top 5 sources
                md = h.get("metadata") or {}
                folder = md.get("mail_folder", "unknown")
                email_from = md.get("email_from", "")
                email_to = md.get("email_to", "")
                subject = md.get("email_subject", "")
                date = md.get("email_date", "")
                
                source_line = f"[{i}] {folder} | {email_from} -> {email_to} | {subject} | {date}"
                sources_list.append(source_line)
            
            sources_section = "\n\nGefundene E-Mail-Quellen:\n" + "\n".join(sources_list)
            answer += sources_section

        # 6) maybe update summary (based on recent size)
        new_summary = await self.maybe_update_summary(summary_block, recent + [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}])
        
        # Harter Guard - Entferne常见的 Ablehnungssätze
        if hits:
            answer = answer.replace("Entschuldigung, aber ich habe keine direkten Zugriff auf Ihre Daten", "")
            answer = answer.replace("Entschuldigung, aber ich habe keinen direkten Zugriff auf Ihre Daten", "")
        
        return answer, new_summary, new_notes
