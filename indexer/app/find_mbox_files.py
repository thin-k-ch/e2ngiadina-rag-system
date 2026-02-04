import os

def find_mbox_files(mail_root: str) -> list[str]:
    """
    Thunderbird/ExQuilla-Ordner nach *lokalen* MBOX-Dateien durchsuchen.
    - nimmt Inbox/Sent/Archiv/... + Unterordner (*.sbd) automatisch mit
    - ignoriert .msf, .sqlite, .dat, json, attachment dirs
    - prüft per Content-Heuristik, ob es wirklich MBOX ist (From_-Separator)
    """
    EXCLUDE_EXT = {
        ".msf", ".sqlite", ".sqlite3", ".dat", ".json", ".log", ".ini", ".bak",
        ".db", ".pem", ".crt", ".key"
    }
    EXCLUDE_DIR_NAMES = {
        "mailbox-attachments", "cache", "Cache", "uploads", "tmp", "Temp"
    }

    def looks_like_mbox(path: str) -> bool:
        try:
            # MBOX hat i.d.R. "From " Separatorzeilen
            with open(path, "rb") as f:
                head = f.read(8192)
            if not head:
                return False
            # klassisch: beginnt mit "From " oder enthält "\nFrom "
            if head.startswith(b"From "):
                return True
            if b"\nFrom " in head:
                return True
            # manche mbox starten mit Headern, haben aber sehr früh "From "
            # (wir prüfen eine zweite Portion, aber klein halten)
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size > 65536:
                    f.seek(65536)
                    chunk = f.read(8192)
                    if b"\nFrom " in chunk:
                        return True
            return False
        except Exception:
            return False

    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(mail_root):
        # prune excluded dirs
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIR_NAMES and not d.startswith(".")
        ]

        for fn in filenames:
            # ignore hidden
            if fn.startswith("."):
                continue

            p = os.path.join(dirpath, fn)

            # ignore if in excluded dir path
            if any(f"/{d}/" in p.replace("\\", "/") for d in EXCLUDE_DIR_NAMES):
                continue

            # drop known non-mbox extensions
            _, ext = os.path.splitext(fn)
            ext = ext.lower()
            if ext in EXCLUDE_EXT:
                continue

            # Thunderbird mbox files are typically files WITHOUT extension.
            # But we accept "no ext" only; if there is an extension we skip.
            if ext != "":
                continue

            # avoid tiny files (empty/placeholder)
            try:
                if os.path.getsize(p) < 256:
                    continue
            except Exception:
                continue

            # content heuristic: only take real mbox
            if looks_like_mbox(p):
                out.append(p)

    out.sort()
    return out
