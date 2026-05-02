"""
Low-level NAV fetching from api.mfapi.in.
"""

from __future__ import annotations
from typing import Optional, List, Dict
import time
import requests
from . import cache as _cache
from .exceptions import FetchError, SchemeNotFoundError

BASE_URL = "https://api.mfapi.in/mf"
SEARCH_URL = f"{BASE_URL}/search"
TIMEOUT = 30


def _get(url: str) -> dict:
    """Raw HTTP GET with 3-attempt exponential backoff."""
    last_exc = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout as e:
            last_exc = e
            time.sleep(2 ** attempt)
        except requests.exceptions.HTTPError as e:
            raise FetchError(url, str(e))
        except requests.exceptions.ConnectionError as e:
            last_exc = e
            time.sleep(2 ** attempt)
        except ValueError as e:
            raise FetchError(url, f"Invalid JSON response: {e}")
    raise FetchError(url, f"Request failed after 3 attempts: {last_exc}")


def fetch_scheme(
    scheme_code: str,
    force_refresh: bool = False,
    ttl: int = _cache.DEFAULT_TTL,
) -> dict:
    """
    Fetch full NAV history for a scheme code from mfapi.in.

    Uses local disk cache (24h TTL) unless force_refresh=True.
    """
    code = str(scheme_code).strip()

    if not force_refresh:
        cached = _cache.get(code, ttl=ttl)
        if cached is not None:
            return cached

    data = _get(f"{BASE_URL}/{code}")

    if not isinstance(data, dict):
        raise FetchError(code, "Unexpected response format")

    if data.get("status") == "ERROR":
        raise SchemeNotFoundError(code)

    if "data" not in data:
        raise FetchError(code, "Response missing 'data' key")

    _cache.set(code, data)
    return data


def search_schemes(query: str, max_results: int = 20) -> List[Dict]:
    """
    Search for schemes by name on mfapi.in.
    Returns list of dicts with keys: schemeCode (str), schemeName (str).
    """
    if not query or not query.strip():
        return []

    try:
        results = _get(f"{SEARCH_URL}?q={requests.utils.quote(query.strip())}")
    except FetchError:
        return []

    if not isinstance(results, list):
        return []

    normalised = []
    for item in results[:max_results]:
        if isinstance(item, dict) and "schemeCode" in item and "schemeName" in item:
            normalised.append({
                "schemeCode": str(item["schemeCode"]),
                "schemeName": str(item["schemeName"]),
            })

    return normalised


def fetch_meta(scheme_code: str) -> Optional[dict]:
    """Return just the 'meta' block for a scheme, or None on failure."""
    try:
        payload = fetch_scheme(scheme_code)
        return payload.get("meta", {})
    except (FetchError, SchemeNotFoundError):
        return None
