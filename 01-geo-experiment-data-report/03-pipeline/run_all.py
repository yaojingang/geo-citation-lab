"""
Three-platform sequential extractor with shared fetch pool.

Scans chatgpt / Google / perplexity HTML dirs, deduplicates citation
URLs across platforms, fetches them once through a single rate-limiter,
then writes per-platform output + a unified summary.

Fetch results are cached to disk immediately so the job can resume
after a crash without re-fetching already-completed URLs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests

from fetch_utils import (
    USER_AGENT,
    DomainRateLimiter,
    NOT_FETCHED_PLACEHOLDER,
    _make_browser_fetch_fn,
    citation_cache_path,
    fetch_url_content,
    write_failures,
    write_json,
)

from chatgpt_extract import parse_chatgpt_html
from google_extract import parse_google_html
from perplexity_extract import parse_perplexity_html, _is_answer_file


PLATFORMS: list[dict[str, Any]] = [
    {
        "key": "chatgpt",
        "label": "ChatGPT",
        "parse_fn": parse_chatgpt_html,
        "subdir": "chatgpt",
        "file_filter": None,
    },
    {
        "key": "google",
        "label": "Google",
        "parse_fn": parse_google_html,
        "subdir": "Google",
        "file_filter": None,
    },
    {
        "key": "perplexity",
        "label": "Perplexity",
        "parse_fn": parse_perplexity_html,
        "subdir": "perplexity",
        "file_filter": _is_answer_file,
    },
]


# ---------------------------------------------------------------------------
# FetchCache: disk-backed per-URL result cache
# ---------------------------------------------------------------------------

class FetchCache:
    """
    Stores one JSON file per URL keyed by SHA-256 hash.
    Thread-safe for concurrent writes.
    """

    def __init__(self, cache_dir: Path):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self._dir / f"{digest}.json"

    def has_success(self, url: str) -> bool:
        p = self._path(url)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return not data.get("fetch_error")
        except (json.JSONDecodeError, OSError):
            return False

    def get(self, url: str) -> dict[str, Any] | None:
        p = self._path(url)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, url: str, result: dict[str, Any]) -> None:
        p = self._path(url)
        payload = json.dumps(result, ensure_ascii=False)
        with self._lock:
            p.write_text(payload, encoding="utf-8")

    def count_cached(self, urls: set[str]) -> tuple[int, int]:
        """Returns (cached_ok, cached_fail) among the given URLs."""
        ok = fail = 0
        for url in urls:
            r = self.get(url)
            if r is None:
                continue
            if r.get("fetch_error"):
                fail += 1
            else:
                ok += 1
        return ok, fail


class _CacheLookup:
    """Dict-like adapter so enrich_record_with_fetch can read from FetchCache."""

    def __init__(self, cache: FetchCache):
        self._cache = cache

    def get(self, url: str, default: Any = None) -> Any:
        r = self._cache.get(url)
        return r if r is not None else default


# ---------------------------------------------------------------------------
# Phase 1: scan & parse
# ---------------------------------------------------------------------------

def scan_platform(
    plat: dict[str, Any],
    base_dir: Path,
    output_dir: Path,
    skip_existing: bool,
) -> tuple[list[tuple[Path, dict[str, Any], Path]], set[str]]:
    input_dir = (base_dir / plat["subdir"]).resolve()
    plat_output = output_dir / plat["key"]
    label = plat["label"]

    if not input_dir.exists():
        print(f"    {label}: directory not found, skipping", flush=True)
        return [], set()

    html_files = sorted(input_dir.rglob("*.html"))
    ff: Callable[[Path], bool] | None = plat["file_filter"]
    if ff:
        html_files = [f for f in html_files if ff(f)]

    total = len(html_files)
    print(f"    {label}: found {total} HTML files, parsing...", end="", flush=True)

    records: list[tuple[Path, dict[str, Any], Path]] = []
    plat_urls: set[str] = set()
    skipped = 0
    t0 = time.monotonic()

    for i, fp in enumerate(html_files):
        rel = fp.relative_to(input_dir)
        out = (plat_output / rel).with_suffix(".json")
        if skip_existing and out.exists():
            skipped += 1
            continue
        rec = plat["parse_fn"](fp)
        rec["platform"] = plat["key"]
        records.append((fp, rec, out))
        for c in rec.get("citations", []):
            if c.get("url"):
                plat_urls.add(c["url"])

        done = i + 1
        if done % 50 == 0 or done == total:
            elapsed = time.monotonic() - t0
            print(
                f"\r    {label}: parsed {done}/{total} ({elapsed:.0f}s)        ",
                end="", flush=True,
            )

    elapsed = time.monotonic() - t0
    skip_info = f", skipped {skipped}" if skipped else ""
    print(
        f"\r    {label}: {len(records)} files parsed,"
        f" {len(plat_urls)} URLs ({elapsed:.0f}s{skip_info})        ",
        flush=True,
    )

    return records, plat_urls


# ---------------------------------------------------------------------------
# Phase 2: unified fetch with disk cache
# ---------------------------------------------------------------------------

def fetch_all_global(
    urls: list[str],
    cache: FetchCache,
    timeout: int,
    delay: float,
    domain_delay: float,
    max_concurrent: int,
    retries: int,
    browser_fallback: bool,
) -> tuple[int, int]:
    """
    Fetch URLs not yet cached. Results go straight to disk via cache.put().
    Returns (total_ok, total_fail) across cached + newly fetched.
    """
    to_fetch = [u for u in urls if not cache.has_success(u)]
    cached_ok = len(urls) - len(to_fetch)

    if cached_ok:
        print(f"  已缓存成功: {cached_ok}  待抓取: {len(to_fetch)}", flush=True)

    if not to_fetch:
        print("  全部已缓存，跳过抓取", flush=True)
        fail_count = sum(1 for u in urls if not cache.has_success(u))
        return cached_ok, fail_count

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    limiter = DomainRateLimiter(domain_delay=domain_delay, global_delay=delay)
    browser_fn = _make_browser_fetch_fn() if browser_fallback else None

    total_target = len(urls)
    fetch_total = len(to_fetch)
    ok = cached_ok
    fail = 0
    start = time.monotonic()
    mc = max(1, max_concurrent)
    retries = max(1, retries)
    width = len(str(total_target))

    def _process(url: str, fetch_idx: int) -> None:
        nonlocal ok, fail
        r = fetch_url_content(session, url, timeout, retries, limiter, browser_fn)
        cache.put(url, r)
        st = r.get("error_class") or "ok"
        if r.get("fetch_error"):
            fail += 1
        else:
            ok += 1
        done_global = ok + fail
        elapsed = time.monotonic() - start
        pct = done_global / total_target * 100
        print(
            f"  [{done_global:>{width}}/{total_target}]"
            f" {pct:5.1f}%"
            f"  ok={ok} fail={fail}"
            f"  {elapsed:6.0f}s"
            f"  {st:12s}"
            f"  {url[:58]}",
            flush=True,
        )

    if mc <= 1:
        for idx, url in enumerate(to_fetch):
            _process(url, idx)
    else:
        with ThreadPoolExecutor(max_workers=mc) as pool:
            fmap = {
                pool.submit(_process, url, idx): url
                for idx, url in enumerate(to_fetch)
            }
            for future in as_completed(fmap):
                future.result()

    elapsed = time.monotonic() - start
    print(
        f"\n  抓取完成: 目标 {total_target}, 成功 {ok},"
        f" 失败 {fail}, 耗时 {elapsed:.1f}s\n",
    )
    return ok, fail


# ---------------------------------------------------------------------------
# Phase 3: write per-platform results (reads from cache lazily)
# ---------------------------------------------------------------------------

def _enrich_record(
    record: dict[str, Any],
    cache_lookup: _CacheLookup,
) -> dict[str, Any]:
    enriched = dict(record)
    enriched_citations: list[dict[str, Any]] = []
    for citation in record.get("citations", []):
        url = citation.get("url", "")
        fetch_info = cache_lookup.get(url, {**NOT_FETCHED_PLACEHOLDER, "url": url})
        ec = dict(citation)
        ec.update({
            "final_url": fetch_info.get("final_url", ""),
            "status_code": fetch_info.get("status_code"),
            "title": fetch_info.get("title", ""),
            "fetched_html": fetch_info.get("fetched_html", ""),
            "fetch_error": fetch_info.get("fetch_error"),
            "error_class": fetch_info.get("error_class", ""),
            "body_snippet": fetch_info.get("body_snippet", ""),
            "attempts": fetch_info.get("attempts", 0),
        })
        enriched_citations.append(ec)
    enriched["citations"] = enriched_citations
    return enriched


def write_platform_results(
    plat: dict[str, Any],
    records: list[tuple[Path, dict[str, Any], Path]],
    cache: FetchCache,
    output_dir: Path,
) -> dict[str, Any]:
    if not records:
        return {"files": 0, "ok": 0, "fail": 0}

    plat_output = output_dir / plat["key"]
    html_cache_dir = plat_output / "_citation_cache"
    html_cache_dir.mkdir(parents=True, exist_ok=True)
    lookup = _CacheLookup(cache)

    url_to_sources: dict[str, list[str]] = {}
    ok_count = 0
    fail_count = 0

    index_path = plat_output / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as idx_f:
        for _src, rec, out_path in records:
            enriched = _enrich_record(rec, lookup)
            write_json(out_path, enriched)

            for c in rec.get("citations", []):
                url = c.get("url", "")
                if url:
                    url_to_sources.setdefault(url, []).append(str(out_path))

            idx_rec = dict(enriched)
            idx_cits = []
            for cit in enriched.get("citations", []):
                c = dict(cit)
                fhtml = c.pop("fetched_html", "")
                if fhtml:
                    cp = citation_cache_path(html_cache_dir, c["url"])
                    cp.write_text(fhtml, encoding="utf-8")
                    c["fetched_html_path"] = str(cp)
                else:
                    c["fetched_html_path"] = ""
                idx_cits.append(c)
            idx_rec["citations"] = idx_cits
            idx_f.write(json.dumps(idx_rec, ensure_ascii=False) + "\n")

    # Collect per-platform failures
    failures: list[dict[str, Any]] = []
    for url, sources in url_to_sources.items():
        info = cache.get(url)
        if info and not info.get("fetch_error"):
            ok_count += 1
        elif info and info.get("fetch_error"):
            fail_count += 1
            failures.append({
                "url": url,
                "error_class": info.get("error_class", ""),
                "fetch_error": info.get("fetch_error", ""),
                "status_code": info.get("status_code"),
                "body_snippet": info.get("body_snippet", "")[:512],
                "attempts": info.get("attempts", 0),
                "source_files": sources,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    write_failures(plat_output, failures)
    return {"files": len(records), "ok": ok_count, "fail": fail_count}


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ChatGPT + Google + Perplexity extraction with shared fetch pool.",
    )
    p.add_argument(
        "--base-dir", type=str, default=".",
        help="Base dir containing chatgpt/, Google/, perplexity/ sub-dirs.",
    )
    p.add_argument("--output", type=str, default="output_all")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--domain-delay", type=float, default=3.0)
    p.add_argument("--max-concurrent", type=int, default=5)
    p.add_argument("--timeout", type=int, default=15)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--browser-fallback", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared fetch cache (survives crashes)
    cache = FetchCache(output_dir / "_fetch_cache")

    # ── Phase 1: Scan ──
    sep = "=" * 64
    print(f"\n{sep}")
    print("  三平台串行抓取调度器")
    print(sep)
    print()
    print("  [Phase 1] 扫描 & 解析 HTML", flush=True)
    print()

    all_records: dict[str, list[tuple[Path, dict[str, Any], Path]]] = {}
    per_plat_urls: dict[str, set[str]] = {}
    global_urls: set[str] = set()
    scan_start = time.monotonic()

    for plat in PLATFORMS:
        records, urls = scan_platform(plat, base_dir, output_dir, args.skip_existing)
        all_records[plat["key"]] = records
        per_plat_urls[plat["key"]] = urls
        global_urls |= urls

    scan_elapsed = time.monotonic() - scan_start

    raw_total = sum(len(u) for u in per_plat_urls.values())
    dedup_total = len(global_urls)
    saved = raw_total - dedup_total

    print()
    print(f"  {'─' * 48}")
    for plat in PLATFORMS:
        key = plat["key"]
        print(
            f"    {plat['label']:12s}  {len(all_records[key]):3d} files"
            f"  {len(per_plat_urls[key]):5d} URLs"
        )
    print(f"  {'─' * 48}")
    print(f"    合计(去重前): {raw_total:5d}    去重后: {dedup_total:5d}    节省: {saved}")
    print(f"    扫描耗时: {scan_elapsed:.0f}s")
    print(sep)

    # ── Phase 2: Fetch ──
    total_ok = 0
    total_fail = 0

    if not args.no_fetch and global_urls:
        print(
            f"\n  [Phase 2] 抓取 {dedup_total} 个 URL"
            f"  (并发={args.max_concurrent}, 域名间隔={args.domain_delay}s)\n"
        )
        total_ok, total_fail = fetch_all_global(
            urls=list(global_urls),
            cache=cache,
            timeout=args.timeout,
            delay=args.delay,
            domain_delay=args.domain_delay,
            max_concurrent=args.max_concurrent,
            retries=args.retries,
            browser_fallback=args.browser_fallback,
        )

    # ── Phase 3: Write ──
    print(f"{sep}")
    print("  [Phase 3] 写入结果")
    print(sep)

    plat_stats: dict[str, dict[str, Any]] = {}
    for plat in PLATFORMS:
        key = plat["key"]
        stats = write_platform_results(plat, all_records[key], cache, output_dir)
        plat_stats[key] = stats
        print(
            f"  {plat['label']:12s}  files={stats['files']}"
            f"  ok={stats['ok']}  fail={stats['fail']}"
            f"  → {output_dir / key}"
        )

    # Global failure log
    global_failures: list[dict[str, Any]] = []
    global_url_to_sources: dict[str, list[str]] = {}
    for plat in PLATFORMS:
        for _, rec, out_path in all_records[plat["key"]]:
            for c in rec.get("citations", []):
                url = c.get("url", "")
                if url:
                    global_url_to_sources.setdefault(url, []).append(str(out_path))

    for url, sources in global_url_to_sources.items():
        info = cache.get(url)
        if info and info.get("fetch_error"):
            global_failures.append({
                "url": url,
                "error_class": info.get("error_class", ""),
                "fetch_error": info.get("fetch_error", ""),
                "status_code": info.get("status_code"),
                "body_snippet": info.get("body_snippet", "")[:512],
                "attempts": info.get("attempts", 0),
                "source_files": sources,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    gf_path = write_failures(output_dir, global_failures)

    # Summary JSON
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platforms": {
            plat["key"]: {
                "html_files": plat_stats[plat["key"]]["files"],
                "unique_urls": len(per_plat_urls[plat["key"]]),
                "fetch_ok": plat_stats[plat["key"]]["ok"],
                "fetch_fail": plat_stats[plat["key"]]["fail"],
            }
            for plat in PLATFORMS
        },
        "global": {
            "total_unique_urls": dedup_total,
            "dedup_saved": saved,
            "fetch_ok": total_ok,
            "fetch_fail": total_fail,
        },
    }
    write_json(output_dir / "summary.json", summary)

    # ── Final Report ──
    if global_failures:
        by_class: dict[str, int] = {}
        for f in global_failures:
            cls = f.get("error_class", "unknown")
            by_class[cls] = by_class.get(cls, 0) + 1
        print(f"\n  失败分类:")
        for cls, cnt in sorted(by_class.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {cnt}")

    print(f"\n{sep}")
    print("  全部完成!")
    print(f"  总 URL: {dedup_total}  成功: {total_ok}  失败: {total_fail}")
    if saved:
        print(f"  跨平台去重节省: {saved} 次重复抓取")
    print(f"  输出目录: {output_dir}")
    print(f"  汇总文件: {output_dir / 'summary.json'}")
    print(f"  抓取缓存: {output_dir / '_fetch_cache'}")
    if global_failures:
        print(f"  失败日志: {gf_path}")
    print(sep)


if __name__ == "__main__":
    main()
