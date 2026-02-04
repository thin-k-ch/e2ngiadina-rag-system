import os
from urllib.parse import quote

def to_file_url(path: str, file_base: str | None = None) -> str:
    if not path:
        return ""
    p = path.strip()

    # If path is relative, join with base
    if file_base and not p.startswith("/"):
        p = os.path.join(file_base, p)

    # If base is provided and path is under base, keep absolute
    # We still encode full absolute path
    return "file://" + quote(p)

def make_clickable_path(path: str, file_base: str | None = None, use_http_proxy: bool = True) -> tuple[str, str]:
    """
    Returns (display_path, url) where display_path is relative to file_base
    and url is either file:// or HTTP proxy URL
    """
    if not path:
        return "", ""
    
    p = path.strip()
    display_path = p
    
    # Make display path relative to base
    if file_base and p.startswith(file_base):
        display_path = p[len(file_base):].lstrip("/")
    
    # Create URL - ALWAYS use full path for HTTP proxy
    if use_http_proxy:
        # Use HTTP proxy (works in browsers) - ALWAYS use full path
        # If path is relative, combine with base
        full_path = p
        if file_base and not p.startswith("/"):
            full_path = os.path.join(file_base, p)
        url = f"http://localhost:11436/open?path={quote(full_path)}"
    else:
        # Use file:// URL (blocked by browsers)
        url = to_file_url(p, file_base)
    
    return display_path, url
