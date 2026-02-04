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
        # 1) retrieve chunks
        hits = self.tools.search_chunks(user_text, top_k=10)
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

        # clamp notes
        new_notes = clamp_tokens(new_notes, self.notes_tokens)

        # 6) maybe update summary (based on recent size)
        new_summary = await self.maybe_update_summary(summary_block, recent + [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}])
        return answer, new_summary, new_notes
