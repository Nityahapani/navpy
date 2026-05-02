"""
Fund name / code search and resolution.
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import difflib
from . import fetch as _fetch
from .exceptions import SchemeNotFoundError, AmbiguousSchemeError
from .models import SchemeInfo


def is_scheme_code(query: str) -> bool:
    """Return True if the query looks like a numeric AMFI scheme code."""
    return query.strip().isdigit()


def search(query: str, max_results: int = 20) -> List[SchemeInfo]:
    """
    Search for mutual fund schemes by name or keyword.

    Parameters
    ----------
    query       : str
    max_results : int

    Returns
    -------
    list of SchemeInfo objects ranked by relevance
    """
    raw = _fetch.search_schemes(query, max_results=max_results)
    if not raw:
        return []
    return [
        SchemeInfo(
            scheme_code=r["schemeCode"],
            scheme_name=r["schemeName"],
        )
        for r in raw
    ]


def resolve(
    query: str,
    prefer: str = "growth",
    plan: str = "splice",
    interactive: bool = True,
) -> SchemeInfo:
    """
    Resolve a user query (name or code) to a single SchemeInfo.

    Parameters
    ----------
    query       : str
    prefer      : str  keyword to prefer in scheme name (default: 'growth')
    plan        : str  'direct', 'regular', or 'splice'
    interactive : bool whether to prompt when multiple matches remain

    Raises
    ------
    SchemeNotFoundError
    AmbiguousSchemeError  (if interactive=False and ambiguous)
    """
    query = query.strip()

    if is_scheme_code(query):
        meta = _fetch.fetch_meta(query)
        if meta is None:
            raise SchemeNotFoundError(query)
        return SchemeInfo(
            scheme_code=query,
            scheme_name=meta.get("scheme_name", query),
            fund_house=meta.get("fund_house", ""),
            scheme_type=meta.get("scheme_type", ""),
            scheme_category=meta.get("scheme_category", ""),
        )

    raw = _fetch.search_schemes(query, max_results=30)
    if not raw:
        raise SchemeNotFoundError(query)

    candidates = raw.copy()

    plan_keyword = {
        "direct": "direct",
        "regular": "regular",
        "splice": None,
    }.get(plan.lower())

    if plan_keyword:
        plan_filtered = [r for r in candidates if plan_keyword in r["schemeName"].lower()]
        if plan_filtered:
            candidates = plan_filtered

    if prefer:
        pref_filtered = [r for r in candidates if prefer.lower() in r["schemeName"].lower()]
        if pref_filtered:
            candidates = pref_filtered

    if len(candidates) == 1:
        r = candidates[0]
        meta = _fetch.fetch_meta(r["schemeCode"]) or {}
        return SchemeInfo(
            scheme_code=r["schemeCode"],
            scheme_name=r["schemeName"],
            fund_house=meta.get("fund_house", ""),
            scheme_type=meta.get("scheme_type", ""),
            scheme_category=meta.get("scheme_category", ""),
        )

    def similarity(name: str) -> float:
        return difflib.SequenceMatcher(None, query.lower(), name.lower()).ratio()

    candidates_scored = sorted(candidates, key=lambda r: similarity(r["schemeName"]), reverse=True)

    if len(candidates_scored) >= 2:
        top = similarity(candidates_scored[0]["schemeName"])
        second = similarity(candidates_scored[1]["schemeName"])
        if top - second > 0.15:
            r = candidates_scored[0]
            meta = _fetch.fetch_meta(r["schemeCode"]) or {}
            return SchemeInfo(
                scheme_code=r["schemeCode"],
                scheme_name=r["schemeName"],
                fund_house=meta.get("fund_house", ""),
                scheme_type=meta.get("scheme_type", ""),
                scheme_category=meta.get("scheme_category", ""),
            )

    if not interactive:
        raise AmbiguousSchemeError(query, candidates_scored)

    display = candidates_scored[:10]
    print(f"\nFound {len(candidates_scored)} matches for '{query}':")
    print("-" * 70)
    for i, r in enumerate(display, 1):
        print(f"  [{i:2d}]  {r['schemeCode']:>8}  {r['schemeName']}")
    print("-" * 70)

    while True:
        try:
            choice = input(f"Enter number (1-{len(display)}), or 0 to cancel: ").strip()
            if choice == "0":
                raise SchemeNotFoundError(query)
            idx = int(choice) - 1
            if 0 <= idx < len(display):
                r = display[idx]
                meta = _fetch.fetch_meta(r["schemeCode"]) or {}
                return SchemeInfo(
                    scheme_code=r["schemeCode"],
                    scheme_name=r["schemeName"],
                    fund_house=meta.get("fund_house", ""),
                    scheme_type=meta.get("scheme_type", ""),
                    scheme_category=meta.get("scheme_category", ""),
                )
            print(f"Please enter a number between 1 and {len(display)}.")
        except ValueError:
            print("Please enter a valid number.")
        except (EOFError, KeyboardInterrupt):
            raise SchemeNotFoundError(query)


def resolve_pair(
    query: str,
    interactive: bool = True,
) -> Tuple[Optional[SchemeInfo], Optional[SchemeInfo]]:
    """
    Resolve a fund name to both its Regular and Direct plan variants.

    Returns
    -------
    (regular_info, direct_info) — either may be None if not found
    """
    raw = _fetch.search_schemes(query, max_results=50)
    if not raw:
        return None, None

    def score(name: str) -> float:
        return difflib.SequenceMatcher(None, query.lower(), name.lower()).ratio()

    regular_candidates = [
        r for r in raw
        if "regular" in r["schemeName"].lower()
        and "growth" in r["schemeName"].lower()
        and "idcw" not in r["schemeName"].lower()
        and "dividend" not in r["schemeName"].lower()
    ]
    direct_candidates = [
        r for r in raw
        if "direct" in r["schemeName"].lower()
        and "growth" in r["schemeName"].lower()
        and "idcw" not in r["schemeName"].lower()
        and "dividend" not in r["schemeName"].lower()
    ]

    def best(candidates: list) -> Optional[SchemeInfo]:
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda r: score(r["schemeName"]), reverse=True)
        r = ranked[0]
        meta = _fetch.fetch_meta(r["schemeCode"]) or {}
        return SchemeInfo(
            scheme_code=r["schemeCode"],
            scheme_name=r["schemeName"],
            fund_house=meta.get("fund_house", ""),
        )

    return best(regular_candidates), best(direct_candidates)
