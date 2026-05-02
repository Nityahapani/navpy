"""
navpy command-line interface.

Usage:
    navpy "Mirae Asset Large Cap" --start 3y
    navpy "107578" --start 2018-01-01 --end 2023-12-31
    navpy "DSP Flexi Cap" --plan direct --output nav.csv
    navpy search "mirae"
    navpy cache info
    navpy cache clear
"""

from __future__ import annotations
import argparse
import sys


def _print_table(data, scheme_name, plan, max_rows=30):
    """Print a formatted NAV table to terminal."""
    rows = data.head(max_rows)
    total = len(data)
    truncated = total > max_rows

    col_w = max(len(scheme_name), 40)
    sep = "─" * (col_w + 28)

    print(f"\n{'DATE':12s}  {'NAV':>14s}  {'DAILY RET':>10s}")
    print(sep)

    prev_nav = None
    for _, row in rows.iterrows():
        nav = row['nav']
        ret_str = ""
        if prev_nav is not None and prev_nav > 0:
            ret = (nav / prev_nav - 1) * 100
            ret_str = f"{ret:>+9.3f}%"
        date_str = str(row['date'].date()) if hasattr(row['date'], 'date') else str(row['date'])
        print(f"{date_str:12s}  {nav:>14.4f}  {ret_str:>10s}")
        prev_nav = nav

    print(sep)
    if truncated:
        print(f"  Showing {max_rows} of {total} records. Use --output to save all.")
    else:
        print(f"  {total} records  |  Plan: {plan}")
    print()


def cmd_get(args):
    """Execute the get command."""
    from .core import get
    from .exceptions import NavpyError

    try:
        result = get(
            query=args.query,
            start=args.start or None,
            end=args.end or None,
            plan=args.plan,
            option=args.option,
            force_refresh=args.force_refresh,
            interactive=True,
        )
    except NavpyError as e:
        print(f"\nError: {e}\n", file=sys.stderr)
        sys.exit(1)

    # Output routing
    if args.output:
        if args.output.endswith('.json'):
            result.to_json(args.output)
        else:
            result.to_csv(args.output)
        result.print_summary()
    else:
        result.print_summary()
        _print_table(result.data, result.scheme_name, result.plan, max_rows=args.rows)


def cmd_search(args):
    """Execute the search command."""
    from .search import search
    from .exceptions import NavpyError

    try:
        results = search(args.query, max_results=args.limit)
    except NavpyError as e:
        print(f"\nError: {e}\n", file=sys.stderr)
        sys.exit(1)

    if not results:
        print(f"\nNo results found for '{args.query}'.\n")
        return

    print(f"\nSearch results for '{args.query}'  ({len(results)} found):")
    print("─" * 72)
    print(f"  {'CODE':>8}  {'SCHEME NAME'}")
    print("─" * 72)
    for r in results:
        print(f"  {r.scheme_code:>8}  {r.scheme_name}")
    print("─" * 72 + "\n")


def cmd_cache(args):
    """Execute cache subcommands."""
    from . import cache as _cache

    if args.cache_cmd == "info":
        info = _cache.cache_info()
        print(f"\nCache directory : {info['cache_dir']}")
        print(f"Cached schemes  : {info['file_count']}")
        print(f"Total size      : {info['total_size_kb']} KB")
        if info['oldest_file']:
            print(f"Oldest entry    : scheme {info['oldest_file']}")
            print(f"Newest entry    : scheme {info['newest_file']}")
        print()

    elif args.cache_cmd == "clear":
        n = _cache.clear_all()
        print(f"\nCleared {n} cached entries.\n")

    elif args.cache_cmd == "invalidate":
        deleted = _cache.invalidate(args.code)
        if deleted:
            print(f"\nCache entry for '{args.code}' deleted.\n")
        else:
            print(f"\nNo cache entry found for '{args.code}'.\n")


def main():
    parser = argparse.ArgumentParser(
        prog='navpy',
        description='Fetch day-wise NAV data for Indian mutual funds.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  navpy "Mirae Asset Large Cap" --start 3y
  navpy "107578" --start 2015-01-01 --end 2020-12-31
  navpy "DSP Flexi Cap" --plan direct --output nav.csv
  navpy "HDFC Flexi Cap" --start 5y --output hdfc.json
  navpy search "mirae asset"
  navpy cache info
  navpy cache clear
        """,
    )

    sub = parser.add_subparsers(dest='command')

    # ── get (default command) ──────────────────────────────────────────────
    get_p = sub.add_parser('get', help='Fetch NAV data for a fund')
    get_p.add_argument('query',
                       help='Fund name, keyword, or AMFI scheme code')
    get_p.add_argument('--start', default=None,
                       help="Start date: YYYY-MM-DD, '1y', '3y', '5y', '6m', 'ytd', 'max'")
    get_p.add_argument('--end', default=None,
                       help="End date: YYYY-MM-DD or shorthand (default: today)")
    get_p.add_argument('--plan', default='splice',
                       choices=['splice', 'direct', 'regular'],
                       help="Which plan to use (default: splice)")
    get_p.add_argument('--option', default='growth',
                       choices=['growth', 'idcw'],
                       help="Growth or IDCW option (default: growth)")
    get_p.add_argument('--output', '-o', default=None,
                       help="Save to file: path.csv or path.json")
    get_p.add_argument('--force-refresh', '-f', action='store_true',
                       help="Bypass cache and fetch fresh data")
    get_p.add_argument('--rows', type=int, default=30,
                       help="Max rows to display in terminal (default: 30)")

    # ── search ────────────────────────────────────────────────────────────
    search_p = sub.add_parser('search', help='Search for fund schemes by name')
    search_p.add_argument('query', help='Fund name or keyword to search')
    search_p.add_argument('--limit', type=int, default=20,
                          help='Maximum results to show (default: 20)')

    # ── cache ─────────────────────────────────────────────────────────────
    cache_p = sub.add_parser('cache', help='Manage local NAV cache')
    cache_sub = cache_p.add_subparsers(dest='cache_cmd')
    cache_sub.add_parser('info', help='Show cache statistics')
    cache_sub.add_parser('clear', help='Delete all cached entries')
    inv_p = cache_sub.add_parser('invalidate', help='Delete cache for one scheme')
    inv_p.add_argument('code', help='AMFI scheme code to invalidate')

    # ── Handle bare query (no subcommand) ─────────────────────────────────
    # Allow: navpy "Fund Name" --start 3y   (without 'get' keyword)
    args, remaining = parser.parse_known_args()

    if args.command is None:
        # Re-parse treating first positional as a 'get' command
        sys.argv.insert(1, 'get')
        args = parser.parse_args()

    if args.command == 'get' or args.command is None:
        cmd_get(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'cache':
        if not args.cache_cmd:
            cache_p.print_help()
        else:
            cmd_cache(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
