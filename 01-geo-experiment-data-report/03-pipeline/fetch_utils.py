from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

warnings.filterwarnings(
    "ignore",
    message=r".*doesn't match a supported version!",
    category=Warning,
)

import requests
from bs4 import BeautifulSoup
from readability import Document

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

JS_CHALLENGE_SIGNATURES = [
    "just a moment",
    "checking your browser",
    "enable javascript",
    "cf-browser-verification",
    "challenge-platform",
    "ray id",
    "_cf_chl",
    "managed by cloudflare",
    "attention required",
    "access denied",
    "please verify you are a human",
]

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def infer_name_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "unknown"


def get_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def strip_badges(name: str) -> str:
    cleaned = normalize_text(name)
    cleaned = re.sub(r"\+\d+$", "", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Classify fetch errors
# ---------------------------------------------------------------------------

def classify_error(
    exc: Exception | None,
    status_code: int | None,
    body_snippet: str,
) -> str:
    if exc:
        exc_str = type(exc).__name__.lower() + " " + str(exc).lower()
        if "timeout" in exc_str or "timed out" in exc_str:
            return "timeout"
        if "connection" in exc_str or "refused" in exc_str or "unreachable" in exc_str:
            return "connection"
        if "ssl" in exc_str or "certificate" in exc_str:
            return "ssl"
        if "dns" in exc_str or "name or service" in exc_str or "getaddrinfo" in exc_str:
            return "dns"

    if status_code is not None:
        if status_code == 403:
            return "forbidden"
        if status_code == 404:
            return "not_found"
        if status_code == 429:
            return "rate_limited"
        if status_code >= 500:
            return "server_error"

    snippet_lower = body_snippet.lower()
    for sig in JS_CHALLENGE_SIGNATURES:
        if sig in snippet_lower:
            return "js_challenge"

    if exc:
        return "unknown"
    return ""


# ---------------------------------------------------------------------------
# Domain-level rate limiter (thread-safe)
# ---------------------------------------------------------------------------

class DomainRateLimiter:
    def __init__(self, domain_delay: float, global_delay: float):
        self._domain_delay = domain_delay
        self._global_delay = global_delay
        self._lock = threading.Lock()
        self._last_domain: dict[str, float] = {}
        self._last_global: float = 0.0

    def wait(self, url: str) -> None:
        domain = get_domain(url)
        with self._lock:
            now = time.monotonic()
            wait_domain = max(0.0, self._last_domain.get(domain, 0.0) - now)
            wait_global = max(0.0, self._last_global - now)
            wait_time = max(wait_domain, wait_global)
            next_ts = now + wait_time
            self._last_domain[domain] = next_ts + self._domain_delay
            self._last_global = next_ts + self._global_delay
        if wait_time > 0:
            time.sleep(wait_time)


# ---------------------------------------------------------------------------
# Fetch with retry + backoff + jitter + error classification
# ---------------------------------------------------------------------------

def _build_result(
    url: str,
    final_url: str = "",
    status_code: int | None = None,
    title: str = "",
    fetched_html: str = "",
    fetch_error: str | None = None,
    error_class: str = "",
    body_snippet: str = "",
    attempts: int = 0,
) -> dict[str, Any]:
    return {
        "url": url,
        "final_url": final_url,
        "status_code": status_code,
        "title": title,
        "fetched_html": fetched_html,
        "fetch_error": fetch_error,
        "error_class": error_class,
        "body_snippet": body_snippet,
        "attempts": attempts,
    }


def _detect_js_challenge(body: str) -> bool:
    lower = body[:4096].lower()
    return any(sig in lower for sig in JS_CHALLENGE_SIGNATURES)


def _backoff_sleep(attempt: int) -> None:
    base = min(2 ** attempt, 30)
    jitter = random.uniform(0, base * 0.5)
    time.sleep(base + jitter)


def fetch_url_content(
    session: requests.Session,
    url: str,
    timeout: int,
    retries: int,
    limiter: DomainRateLimiter | None,
    browser_fn: Any | None = None,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    last_status: int | None = None
    last_snippet: str = ""

    for attempt in range(1, retries + 1):
        if limiter:
            limiter.wait(url)

        try:
            resp = session.get(url, timeout=timeout, allow_redirects=True)
            last_status = resp.status_code
            last_snippet = resp.text[:2048]

            if resp.status_code in RETRYABLE_STATUS_CODES:
                last_exc = requests.HTTPError(
                    f"HTTP {resp.status_code}", response=resp
                )
                if attempt < retries:
                    _backoff_sleep(attempt)
                continue

            resp.raise_for_status()

            if _detect_js_challenge(resp.text):
                if browser_fn:
                    return _browser_fetch(browser_fn, url, attempt)
                return _build_result(
                    url=url,
                    final_url=resp.url,
                    status_code=resp.status_code,
                    fetch_error="js_challenge detected",
                    error_class="js_challenge",
                    body_snippet=resp.text[:2048],
                    attempts=attempt,
                )

            try:
                doc = Document(resp.text)
                main_html = doc.summary(html_partial=False)
                page_title = normalize_text(doc.short_title() or "")
            except Exception:
                main_html = resp.text[:50000] if resp.text else ""
                page_title = ""

            return _build_result(
                url=url,
                final_url=resp.url,
                status_code=resp.status_code,
                title=page_title,
                fetched_html=main_html,
                attempts=attempt,
            )

        except requests.exceptions.RequestException as exc:
            last_exc = exc
            last_status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt < retries:
                _backoff_sleep(attempt)
            continue

    if browser_fn and classify_error(last_exc, last_status, last_snippet) in (
        "js_challenge", "forbidden", "rate_limited", "timeout",
    ):
        return _browser_fetch(browser_fn, url, retries)

    err_class = classify_error(last_exc, last_status, last_snippet)
    return _build_result(
        url=url,
        status_code=last_status,
        fetch_error=str(last_exc) if last_exc else f"HTTP {last_status}",
        error_class=err_class,
        body_snippet=last_snippet[:2048],
        attempts=retries,
    )


# ---------------------------------------------------------------------------
# Playwright browser fallback
# ---------------------------------------------------------------------------

def _make_browser_fetch_fn() -> Any:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F811
    except ImportError:
        print(
            "[warn] playwright not installed; browser fallback disabled. "
            "Install with: pip install playwright && python -m playwright install chromium"
        )
        return None

    _pw_holder: dict[str, Any] = {}

    def browser_fetch(url: str) -> dict[str, Any]:
        if "pw" not in _pw_holder:
            pw = sync_playwright().start()
            browser = pw.chromium.launch(headless=True)
            _pw_holder["pw"] = pw
            _pw_holder["browser"] = browser
        browser = _pw_holder["browser"]
        page = browser.new_page(user_agent=USER_AGENT)
        try:
            page.goto(url, timeout=30_000, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)
            html = page.content()
            try:
                doc = Document(html)
                fetched = doc.summary(html_partial=False)
                title = normalize_text(doc.short_title() or "")
            except Exception:
                fetched = html[:50000] if html else ""
                title = ""
            return {
                "fetched_html": fetched,
                "title": title,
                "final_url": page.url,
            }
        finally:
            page.close()

    return browser_fetch


def _browser_fetch(browser_fn: Any, url: str, prev_attempts: int) -> dict[str, Any]:
    try:
        result = browser_fn(url)
        return _build_result(
            url=url,
            final_url=result.get("final_url", url),
            status_code=200,
            title=result.get("title", ""),
            fetched_html=result.get("fetched_html", ""),
            fetch_error=None,
            error_class="",
            attempts=prev_attempts + 1,
        )
    except Exception as exc:  # noqa: BLE001
        return _build_result(
            url=url,
            fetch_error=f"browser_fallback failed: {exc}",
            error_class="browser_error",
            attempts=prev_attempts + 1,
        )


# ---------------------------------------------------------------------------
# Orchestrator: fetch all URLs with domain rate-limiting
# ---------------------------------------------------------------------------

def fetch_all_urls(
    urls: list[str],
    timeout: int,
    delay: float,
    domain_delay: float,
    max_concurrent: int,
    retries: int,
    browser_fallback: bool,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not urls:
        return results

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    limiter = DomainRateLimiter(domain_delay=domain_delay, global_delay=delay)

    browser_fn = _make_browser_fetch_fn() if browser_fallback else None

    total = len(urls)

    if max_concurrent <= 1:
        for idx, url in enumerate(urls, 1):
            print(f"  [{idx}/{total}] {url[:90]}...", flush=True)
            results[url] = fetch_url_content(
                session, url, timeout, retries, limiter, browser_fn
            )
            status = results[url].get("error_class") or "ok"
            print(f"    -> {status}", flush=True)
        return results

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        future_map = {
            pool.submit(
                fetch_url_content, session, url, timeout, retries, limiter, browser_fn
            ): url
            for url in urls
        }
        done_count = 0
        for future in as_completed(future_map):
            url = future_map[future]
            done_count += 1
            results[url] = future.result()
            status = results[url].get("error_class") or "ok"
            print(f"  [{done_count}/{total}] {status} {url[:80]}", flush=True)
    return results


# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def citation_cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.html"


# ---------------------------------------------------------------------------
# Enrich parsed records with fetch results
# ---------------------------------------------------------------------------

NOT_FETCHED_PLACEHOLDER: dict[str, Any] = {
    "final_url": "",
    "status_code": None,
    "title": "",
    "fetched_html": "",
    "fetch_error": "not fetched",
    "error_class": "not_fetched",
    "body_snippet": "",
    "attempts": 0,
}


def enrich_record_with_fetch(
    record: dict[str, Any],
    fetched_by_url: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    enriched = dict(record)
    enriched_citations: list[dict[str, Any]] = []
    for citation in record.get("citations", []):
        url = citation["url"]
        fetch_info = fetched_by_url.get(url, {**NOT_FETCHED_PLACEHOLDER, "url": url})
        enriched_citation = dict(citation)
        enriched_citation.update(
            {
                "final_url": fetch_info.get("final_url", ""),
                "status_code": fetch_info.get("status_code"),
                "title": fetch_info.get("title", ""),
                "fetched_html": fetch_info.get("fetched_html", ""),
                "fetch_error": fetch_info.get("fetch_error"),
                "error_class": fetch_info.get("error_class", ""),
                "body_snippet": fetch_info.get("body_snippet", ""),
                "attempts": fetch_info.get("attempts", 0),
            }
        )
        enriched_citations.append(enriched_citation)
    enriched["citations"] = enriched_citations
    return enriched


# ---------------------------------------------------------------------------
# Failure log: write + read + backfill
# ---------------------------------------------------------------------------

def collect_failures(
    fetched_by_url: dict[str, dict[str, Any]],
    url_to_source_files: dict[str, list[str]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for url, info in fetched_by_url.items():
        if info.get("fetch_error"):
            failures.append({
                "url": url,
                "error_class": info.get("error_class", ""),
                "fetch_error": info.get("fetch_error", ""),
                "status_code": info.get("status_code"),
                "body_snippet": info.get("body_snippet", "")[:512],
                "attempts": info.get("attempts", 0),
                "source_files": url_to_source_files.get(url, []),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    return failures


def write_failures(output_dir: Path, failures: list[dict[str, Any]]) -> Path:
    path = output_dir / "_failures.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for record in failures:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_failures(failures_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with failures_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def backfill_json_file(json_path: Path, url: str, fetch_info: dict[str, Any]) -> bool:
    if not json_path.exists():
        return False
    data = json.loads(json_path.read_text(encoding="utf-8"))
    changed = False
    for citation in data.get("citations", []):
        if citation.get("url") == url and citation.get("fetch_error"):
            citation["final_url"] = fetch_info.get("final_url", "")
            citation["status_code"] = fetch_info.get("status_code")
            citation["title"] = fetch_info.get("title", "")
            citation["fetched_html"] = fetch_info.get("fetched_html", "")
            citation["fetch_error"] = fetch_info.get("fetch_error")
            citation["error_class"] = fetch_info.get("error_class", "")
            citation["body_snippet"] = fetch_info.get("body_snippet", "")
            citation["attempts"] = fetch_info.get("attempts", 0)
            changed = True
    if changed:
        write_json(json_path, data)
    return changed


# ---------------------------------------------------------------------------
# Shared CLI argument builder
# ---------------------------------------------------------------------------

def add_fetch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output", type=str,
        help="Output path. For --file: json path. For --dir: output root dir.",
    )
    parser.add_argument(
        "--no-fetch", action="store_true",
        help="Only extract citation URLs, skip fetching.",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Min delay (seconds) between any two fetches.",
    )
    parser.add_argument(
        "--domain-delay", type=float, default=3.0,
        help="Min delay (seconds) between fetches to the same domain.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=1,
        help="Max concurrent fetch workers (1 = sequential).",
    )
    parser.add_argument(
        "--timeout", type=int, default=15,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="Max retry attempts per URL (with exponential backoff).",
    )
    parser.add_argument(
        "--browser-fallback", action="store_true",
        help="Use Playwright browser as fallback for JS-challenge pages.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Batch mode: skip output json that already exists.",
    )


def make_fetch_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "timeout": args.timeout,
        "delay": args.delay,
        "domain_delay": args.domain_delay,
        "max_concurrent": max(1, args.max_concurrent),
        "retries": max(1, args.retries),
        "browser_fallback": args.browser_fallback,
    }


# ---------------------------------------------------------------------------
# Generic run modes (shared by all extractors)
# ---------------------------------------------------------------------------

def run_single_file(
    args: argparse.Namespace,
    parse_fn: Callable[[Path], dict[str, Any]],
    platform: str,
) -> None:
    input_file = Path(args.file).resolve()
    output_file = (
        Path(args.output).resolve()
        if args.output
        else input_file.with_suffix(".json")
    )

    record = parse_fn(input_file)
    record["platform"] = platform
    urls = [c["url"] for c in record["citations"]]
    fetched_by_url: dict[str, dict[str, Any]] = {}

    if not args.no_fetch and urls:
        print(f"[single] fetching {len(urls)} citation(s)...")
        fetched_by_url = fetch_all_urls(urls=urls, **make_fetch_kwargs(args))

    enriched = enrich_record_with_fetch(record, fetched_by_url)
    write_json(output_file, enriched)

    fail_count = sum(1 for c in enriched["citations"] if c.get("fetch_error"))
    print(f"[single] output: {output_file}")
    if fail_count:
        print(f"[single] failures: {fail_count}/{len(enriched['citations'])}")


def run_batch(
    args: argparse.Namespace,
    parse_fn: Callable[[Path], dict[str, Any]],
    platform: str,
    default_output_name: str = "json_output",
    file_filter: Callable[[Path], bool] | None = None,
) -> None:
    input_dir = Path(args.dir).resolve()
    output_dir = (
        Path(args.output).resolve()
        if args.output
        else Path.cwd() / default_output_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(input_dir.rglob("*.html"))
    if file_filter:
        html_files = [f for f in html_files if file_filter(f)]
    if not html_files:
        print(f"[batch] no html found under: {input_dir}")
        return

    print(f"[batch] scanning {len(html_files)} html files...")
    parsed_records: list[tuple[Path, dict[str, Any], Path]] = []
    unique_urls: dict[str, None] = {}
    url_to_source_files: dict[str, list[str]] = {}

    for file_path in html_files:
        rel_path = file_path.relative_to(input_dir)
        out_path = (output_dir / rel_path).with_suffix(".json")
        if args.skip_existing and out_path.exists():
            continue
        record = parse_fn(file_path)
        record["platform"] = platform
        parsed_records.append((file_path, record, out_path))
        for citation in record.get("citations", []):
            curl = citation["url"]
            unique_urls[curl] = None
            url_to_source_files.setdefault(curl, []).append(str(out_path))

    print(f"[batch] parsed: {len(parsed_records)}, unique URLs: {len(unique_urls)}")

    fetched_by_url: dict[str, dict[str, Any]] = {}
    if not args.no_fetch and unique_urls:
        print(f"[batch] fetching {len(unique_urls)} unique citation URL(s)...")
        fetched_by_url = fetch_all_urls(
            urls=list(unique_urls.keys()), **make_fetch_kwargs(args)
        )

    cache_dir = output_dir / "_citation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as index_file:
        for _source, record, out_path in parsed_records:
            enriched = enrich_record_with_fetch(record, fetched_by_url)
            write_json(out_path, enriched)

            index_record = dict(enriched)
            index_citations = []
            for citation in enriched.get("citations", []):
                c = dict(citation)
                fetched_html = c.pop("fetched_html", "")
                if fetched_html:
                    cp = citation_cache_path(cache_dir, c["url"])
                    cp.write_text(fetched_html, encoding="utf-8")
                    c["fetched_html_path"] = str(cp)
                else:
                    c["fetched_html_path"] = ""
                index_citations.append(c)
            index_record["citations"] = index_citations
            index_file.write(json.dumps(index_record, ensure_ascii=False) + "\n")

    failures = collect_failures(fetched_by_url, url_to_source_files)
    fail_path = write_failures(output_dir, failures)

    ok_count = sum(1 for v in fetched_by_url.values() if not v.get("fetch_error"))
    print(f"[batch] html files total: {len(html_files)}")
    print(f"[batch] processed files: {len(parsed_records)}")
    print(f"[batch] fetch ok: {ok_count}, fetch fail: {len(failures)}")
    print(f"[batch] output dir: {output_dir}")
    print(f"[batch] index: {index_path}")
    if failures:
        print(f"[batch] failures log: {fail_path}")
        by_class: dict[str, int] = {}
        for f in failures:
            cls = f.get("error_class", "unknown")
            by_class[cls] = by_class.get(cls, 0) + 1
        for cls, cnt in sorted(by_class.items(), key=lambda x: -x[1]):
            print(f"  {cls}: {cnt}")


def run_retry_failures(args: argparse.Namespace) -> None:
    failures_path = Path(args.retry_failures).resolve()
    if not failures_path.exists():
        print(f"[retry] failures file not found: {failures_path}")
        return

    output_dir = failures_path.parent
    failures = read_failures(failures_path)
    if not failures:
        print("[retry] no failures to retry")
        return

    urls = list({f["url"] for f in failures})
    print(f"[retry] retrying {len(urls)} failed URL(s)...")

    fetched_by_url = fetch_all_urls(urls=urls, **make_fetch_kwargs(args))

    cache_dir = output_dir / "_citation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    backfilled = 0
    still_failed: list[dict[str, Any]] = []

    for failure_record in failures:
        url = failure_record["url"]
        info = fetched_by_url.get(url, {})

        if info.get("fetch_error"):
            still_failed.append({
                "url": url,
                "error_class": info.get("error_class", ""),
                "fetch_error": info.get("fetch_error", ""),
                "status_code": info.get("status_code"),
                "body_snippet": info.get("body_snippet", "")[:512],
                "attempts": info.get("attempts", 0),
                "source_files": failure_record.get("source_files", []),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prev_timestamp": failure_record.get("timestamp", ""),
            })
            continue

        if info.get("fetched_html"):
            cp = citation_cache_path(cache_dir, url)
            cp.write_text(info["fetched_html"], encoding="utf-8")

        for json_path_str in failure_record.get("source_files", []):
            json_path = Path(json_path_str)
            if backfill_json_file(json_path, url, info):
                backfilled += 1

    write_failures(output_dir, still_failed)
    print(f"[retry] success: {len(urls) - len(still_failed)}, still failed: {len(still_failed)}")
    print(f"[retry] json files backfilled: {backfilled}")
    if still_failed:
        print(f"[retry] updated failures log: {output_dir / '_failures.jsonl'}")
