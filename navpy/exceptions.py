"""Custom exceptions for navpy."""


class NavpyError(Exception):
    """Base exception for all navpy errors."""
    pass


class SchemeNotFoundError(NavpyError):
    """Raised when no scheme matches the given query."""

    def __init__(self, query: str) -> None:
        self.query = query
        super().__init__(
            f"No scheme found matching '{query}'. "
            "Try a shorter search term or use search(query) to browse."
        )


class AmbiguousSchemeError(NavpyError):
    """Raised when multiple schemes match and auto-resolution is disabled."""

    def __init__(self, query: str, matches: list) -> None:
        self.query = query
        self.matches = matches
        lines = "\n".join(
            f"  [{i + 1}] {m['schemeCode']:>8}  {m['schemeName']}"
            for i, m in enumerate(matches[:10])
        )
        super().__init__(
            f"Multiple schemes found for '{query}':\n{lines}\n"
            "Pass scheme_code=<code> directly, or use search(query) to browse."
        )


class FetchError(NavpyError):
    """Raised when NAV data cannot be fetched from the API."""

    def __init__(self, code: str, reason: str = "") -> None:
        self.code = code
        super().__init__(
            f"Failed to fetch NAV data for scheme code '{code}'"
            + (f": {reason}" if reason else ".")
        )


class NoDataError(NavpyError):
    """Raised when a scheme exists but has no NAV data in the requested range."""

    def __init__(self, code: str, start: str = None, end: str = None) -> None:
        self.code = code
        period = ""
        if start or end:
            period = f" between {start} and {end}"
        super().__init__(
            f"No NAV data available for scheme '{code}'{period}. "
            "The fund may not have existed in this period, "
            "or the data source has no records."
        )


class InvalidDateError(NavpyError):
    """Raised when a date string cannot be parsed."""

    def __init__(self, value: str) -> None:
        super().__init__(
            f"Cannot parse date '{value}'. "
            "Use YYYY-MM-DD format, or a shorthand like "
            "'1y', '3y', '5y', '6m', 'ytd', 'max'."
        )
