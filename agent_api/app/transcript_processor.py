"""
Transcript Processor - Detects transcript/protocol requests and provides
specialized processing that bypasses RAG search entirely.

Use cases:
- Whisper transcripts → structured meeting minutes
- Pasted text → protocol with action items
- File reference → load + process

The key insight: RAG search is counterproductive for transcript processing.
The full text must go directly to the LLM with a specialized prompt.
"""

import re
import os
from typing import Optional, Tuple


def detect_transcript_mode(user_text: str) -> Optional[str]:
    """
    Detect if the user wants transcript/protocol processing.
    
    Returns a mode string if detected:
    - "protocol" → Generate structured meeting minutes
    - "summary" → Generate executive summary
    - None → Not a transcript request
    """
    text_lower = user_text.lower()
    
    # Strong signals: explicit protocol/transcript keywords
    protocol_patterns = [
        r'(?:erstell|schreib|mach|generier|formulier)\w*\s+.*?(?:protokoll|niederschrift|mitschrift)',
        r'(?:protokoll|niederschrift|mitschrift)\s+.*?(?:erstell|schreib|mach|generier|formulier)',
        r'transkript\w*\s+.*?(?:protokoll|aufbereite|verarbeit|überführ|umwandel|ausarbeit|formulier)',
        r'(?:protokoll|aufbereite|verarbeit|überführ|umwandel|ausarbeit|formulier)\w*\s+.*?transkript',
        r'whisper\s*[-‑]?\s*(?:transkript|aufnahme|text|output)',
        r'(?:sitzung|meeting|besprechung|call)\s*[-‑]?\s*protokoll',
        r'(?:aus|von)\s+(?:diesem|der|dem)\s+(?:transkript|aufnahme|mitschnitt|aufzeichnung)',
        r'(?:pendenzen|aktionspunkte|action\s*items|todos?)\s+.*?(?:aus|extrahier|erstell)',
        r'(?:dieses?|folgendes?)\s+(?:transkript|aufnahme|gespräch|meeting)',
    ]
    
    for pattern in protocol_patterns:
        if re.search(pattern, text_lower):
            return "protocol"
    
    # Medium signal: long text (>1000 chars) + protocol-like keywords
    if len(user_text) > 1000:
        medium_signals = [
            'protokoll', 'transkript', 'pendenzen', 'zusammenfass',
            'aufbereite', 'strukturier', 'sitzung', 'meeting',
            'besprechung', 'niederschrift', 'action items',
        ]
        if any(s in text_lower for s in medium_signals):
            return "protocol"
    
    # Weak signal: very long pasted text (>3000 chars) that looks like a transcript
    # (contains speaker patterns, timestamps, or conversational patterns)
    if len(user_text) > 3000:
        transcript_markers = [
            r'\d{1,2}:\d{2}',           # timestamps
            r'(?:sprecher|speaker)\s*\d', # speaker labels
            r'^[A-ZÄÖÜ][a-zäöü]+:\s',    # "Name: text" pattern (multiline)
            r'\b(?:okay|also|genau|ja|nein|ähm|mhm)\b',  # filler words
        ]
        marker_count = sum(1 for p in transcript_markers if re.search(p, user_text, re.MULTILINE))
        if marker_count >= 2:
            return "protocol"
    
    return None


def extract_file_reference(user_text: str) -> Optional[str]:
    """
    Extract a file path from the user text if they reference a specific file.
    
    Examples:
    - "verarbeite /data/MailsFEA/transcript.txt"
    - "erstelle ein Protokoll aus der Datei meeting_2025.txt"
    """
    # Explicit path patterns
    path_patterns = [
        r'(?:datei|file|pfad|path)\s*[:\s]+([/\w\-\.\s]+\.(?:txt|md|rst))',
        r'(/(?:data|media)[/\w\-\.\s]+\.(?:txt|md|rst))',
        r'(?:aus|von|für)\s+([/\w\-\.\s]+\.(?:txt|md|rst))',
    ]
    
    for pattern in path_patterns:
        m = re.search(pattern, user_text, re.IGNORECASE)
        if m:
            path = m.group(1).strip()
            return path
    
    return None


def separate_instruction_and_transcript(user_text: str) -> Tuple[str, str]:
    """
    Separate the user's instruction from the pasted transcript text.
    
    Heuristic: The instruction is usually the first 1-3 lines,
    the transcript follows after a blank line or separator.
    """
    lines = user_text.split('\n')
    
    # Look for a clear separator
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i > 0 and i < 10:
            # Blank line after short instruction
            if not stripped and i >= 1:
                instruction_part = '\n'.join(lines[:i]).strip()
                transcript_part = '\n'.join(lines[i+1:]).strip()
                if len(instruction_part) < 500 and len(transcript_part) > 200:
                    return instruction_part, transcript_part
            # Explicit separator
            if stripped in ('---', '===', '***', '---', 'Transkript:', 'Text:', 'Inhalt:'):
                instruction_part = '\n'.join(lines[:i]).strip()
                transcript_part = '\n'.join(lines[i+1:]).strip()
                if transcript_part:
                    return instruction_part, transcript_part
    
    # No clear separator found - check if the message starts with a short instruction
    # followed by much longer text
    if len(lines) > 5:
        first_line = lines[0].strip()
        rest = '\n'.join(lines[1:]).strip()
        if len(first_line) < 300 and len(rest) > len(first_line) * 3:
            return first_line, rest
    
    # Fallback: treat everything as transcript, use default instruction
    return "", user_text


PROTOCOL_SYSTEM_PROMPT = """DU BIST EIN PROFESSIONELLER PROTOKOLL-ERSTELLER.

Deine Aufgabe ist es, aus dem bereitgestellten Text (Transkript, Mitschrift, Gesprächsaufzeichnung) 
ein professionelles, vollständiges Sitzungsprotokoll zu erstellen.

STRUKTUR DES PROTOKOLLS:

# Sitzungsprotokoll

## Angaben zur Sitzung
- **Datum:** (falls erkennbar, sonst "Nicht angegeben")
- **Teilnehmende:** (alle genannten Personen mit Rolle/Funktion falls erkennbar)
- **Thema/Anlass:** (Hauptthema der Besprechung)

## Traktanden / Besprochene Themen
Nummerierte Liste aller besprochenen Themen als Überschriften.

### 1. [Thema]
- Zusammenfassung der Diskussion
- Wichtige Aussagen und Positionen der Teilnehmenden
- Getroffene Entscheidungen (fett markiert)

### 2. [Nächstes Thema]
...

## Beschlüsse und Entscheidungen
Nummerierte Liste aller explizit getroffenen Entscheidungen.

## Pendenzenliste / Offene Punkte
| Nr | Pendenz | Verantwortlich | Termin | Status |
|----|---------|----------------|--------|--------|
| 1  | ...     | ...            | ...    | offen  |

## Nächste Schritte
- Konkrete nächste Aktionen mit Verantwortlichkeiten

---

REGELN:
1. Antworte IMMER auf Deutsch (ausser der Originaltext ist in einer anderen Sprache)
2. Sei VOLLSTÄNDIG - erfasse JEDEN besprochenen Punkt, nicht nur die Hauptthemen
3. Unterscheide klar zwischen Fakten, Meinungen und Entscheidungen
4. Behalte die Detailtiefe - wichtige Zahlen, Daten, Namen MÜSSEN enthalten sein
5. Bei Whisper-Transkripten: Erkennungsfehler intelligent korrigieren
6. Pendenzenliste ist PFLICHT - extrahiere alle Aufgaben, auch implizite
7. Wenn Personen sprechen, ordne Aussagen den Personen zu
8. Formatiere professionell mit Markdown
9. Kürze NICHT ab - das Protokoll soll umfassend sein
10. Fachbegriffe und Abkürzungen bei erster Nennung erklären falls möglich
"""

PROTOCOL_USER_TEMPLATE = """Erstelle ein vollständiges, professionelles Sitzungsprotokoll aus dem folgenden Transkript.

{instruction}

--- TRANSKRIPT BEGINN ---
{transcript}
--- TRANSKRIPT ENDE ---

Erstelle nun das vollständige Protokoll mit allen Traktanden, Beschlüssen und einer detaillierten Pendenzenliste."""


async def load_transcript_file(file_path: str) -> Optional[str]:
    """
    Load a transcript file from the filesystem.
    Tries multiple base paths.
    """
    candidates = [
        file_path,
        os.path.join("/media/felix/RAG/1", file_path.lstrip("/")),
        os.path.join("/data", file_path.lstrip("/")),
    ]
    
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                print(f"⚠️ Failed to read {path}: {e}")
    
    # Try via PyRunner for containerized access
    try:
        from .code_executor import execute_code
        code = f"""
import os
path = os.path.join(DATA_ROOT, {repr(file_path.lstrip('/'))})
if os.path.isfile(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        result = f.read()
else:
    result = None
"""
        resp = await execute_code(code)
        if resp.get("ok") and resp.get("result"):
            return resp["result"]
    except Exception as e:
        print(f"⚠️ PyRunner file load failed: {e}")
    
    return None
