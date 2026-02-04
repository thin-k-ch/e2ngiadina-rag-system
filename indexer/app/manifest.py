import sqlite3
from dataclasses import dataclass

@dataclass
class ManifestRow:
    path: str
    sha1: str
    mtime: int
    size: int

class Manifest:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init()

    def _init(self):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
          path TEXT PRIMARY KEY,
          sha1 TEXT NOT NULL,
          mtime INTEGER NOT NULL,
          size INTEGER NOT NULL
        )""")
        con.commit()
        con.close()

    def get(self, path: str):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("SELECT path, sha1, mtime, size FROM files WHERE path=?", (path,))
        r = cur.fetchone()
        con.close()
        return ManifestRow(*r) if r else None

    def upsert(self, row: ManifestRow):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("""
        INSERT INTO files(path,sha1,mtime,size) VALUES (?,?,?,?)
        ON CONFLICT(path) DO UPDATE SET sha1=excluded.sha1, mtime=excluded.mtime, size=excluded.size
        """, (row.path, row.sha1, row.mtime, row.size))
        con.commit()
        con.close()
