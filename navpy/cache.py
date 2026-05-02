"""
Local disk cache for NAV data.

Cache lives at ~/.navpy/cache/
Files are named by scheme code: <code>.json
TTL defaults to 24 hours — re-fetches if stale.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Optional

CACHE_DIR = Path.home() / ".navpy" / "cache"
DEFAULT_TTL = 60 * 60 * 24   # 24 hours in seconds


def _cache_path(scheme_code: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{scheme_code}.json"


def get(scheme_code: str, ttl: int = DEFAULT_TTL) -> Optional[dict]:
    """
    Return cached data for a scheme code, or None if missing / stale.

    Parameters
    ----------
    scheme_code : str
    ttl         : int  seconds before cache is considered stale (default 24h)

    Returns
    -------
    dict with keys 'meta' and 'data', or None
    """
    path = _cache_path(scheme_code)
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > ttl:
            return None
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def set(scheme_code: str, payload: dict) -> None:
    """
    Write payload to cache for a scheme code.

    Parameters
    ----------
    scheme_code : str
    payload     : dict  — the raw mfapi.in response (keys: meta, data)
    """
    path = _cache_path(scheme_code)
    try:
        with open(path, 'w') as f:
            json.dump(payload, f)
    except OSError:
        pass   # cache write failures are non-fatal


def invalidate(scheme_code: str) -> bool:
    """
    Delete the cache entry for a scheme code.

    Returns True if a file was deleted, False if nothing existed.
    """
    path = _cache_path(scheme_code)
    if path.exists():
        path.unlink()
        return True
    return False


def clear_all() -> int:
    """
    Delete all cached files.

    Returns
    -------
    int : number of files deleted
    """
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for p in CACHE_DIR.glob("*.json"):
        try:
            p.unlink()
            count += 1
        except OSError:
            pass
    return count


def cache_info() -> dict:
    """
    Return metadata about the current cache state.

    Returns
    -------
    dict with keys: cache_dir, file_count, total_size_kb, oldest_file, newest_file
    """
    if not CACHE_DIR.exists():
        return {
            "cache_dir": str(CACHE_DIR),
            "file_count": 0,
            "total_size_kb": 0,
            "oldest_file": None,
            "newest_file": None,
        }
    files = list(CACHE_DIR.glob("*.json"))
    if not files:
        return {
            "cache_dir": str(CACHE_DIR),
            "file_count": 0,
            "total_size_kb": 0,
            "oldest_file": None,
            "newest_file": None,
        }
    mtimes = [f.stat().st_mtime for f in files]
    sizes = [f.stat().st_size for f in files]
    return {
        "cache_dir": str(CACHE_DIR),
        "file_count": len(files),
        "total_size_kb": round(sum(sizes) / 1024, 1),
        "oldest_file": files[mtimes.index(min(mtimes))].stem,
        "newest_file": files[mtimes.index(max(mtimes))].stem,
    }
