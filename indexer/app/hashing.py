import hashlib
import os

def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def file_stat(path: str) -> tuple[int, int]:
    st = os.stat(path)
    return (int(st.st_mtime), int(st.st_size))
