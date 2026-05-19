from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from fetch_utils import (
    add_fetch_args,
    infer_name_from_url,
    normalize_text,
    run_batch,
    run_retry_failures,
    run_single_file,
)

PLATFORM = "perplexity"

PERPLEXITY_FILTER_DOMAINS = {
    "perplexity.ai", "www.perplexity.ai",
}


def _is_filtered_link(url: str) -> bool:
    netloc = urlparse(url).netloc.lower()
    if any(netloc == d or netloc.endswith("." + d) for d in PERPLEXITY_FILTER_DOMAINS):
        return True
    if "google.com/s2/favicons" in url:
        return True
    return False


# ---------------------------------------------------------------------------
# Dual-file pairing: _0_ (answer) + _1_ (links)
# ---------------------------------------------------------------------------

_FILE_INDEX_RE = re.compile(r"_([01])_\d{3}_")


def _get_file_index(path: Path) -> str | None:
    m = _FILE_INDEX_RE.search(path.name)
    return m.group(1) if m else None


def find_pair_file(answer_path: Path) -> Path | None:
    """Given a _0_ answer file, find the corresponding _1_ links file."""
    idx = _get_file_index(answer_path)
    if idx != "0":
        return None
    parent = answer_path.parent
    for sibling in parent.iterdir():
        if sibling == answer_path:
            continue
        if sibling.suffix.lower() != ".html":
            continue
        if _get_file_index(sibling) == "1":
            return sibling
    return None


# ---------------------------------------------------------------------------
# Perplexity HTML parsing
# ---------------------------------------------------------------------------

def extract_question(soup: BeautifulSoup) -> str:
    title_tag = soup.title
    if title_tag and title_tag.string:
        return normalize_text(title_tag.string)

    h1 = soup.select_one("h1.group\\/query")
    if not h1:
        h1 = soup.select_one("h1")
    if h1:
        return normalize_text(h1.get_text(" ", strip=True))

    return ""


def extract_answer_html(soup: BeautifulSoup) -> str:
    prose = soup.select_one("div.prose")
    if prose:
        renderer = prose.select_one('div[data-renderer="lm"]')
        if renderer:
            return renderer.decode_contents().strip()
        return prose.decode_contents().strip()
    return ""


def extract_citations_from_links_file(soup1: BeautifulSoup) -> list[dict[str, str]]:
    """Extract citation URLs from the _1_ links file."""
    anchors = soup1.select('a.gap-sm[href^="http"]')
    if not anchors:
        anchors = soup1.select('a[href^="http"][rel~="noopener"]')

    unique: dict[str, dict[str, str]] = {}
    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href.startswith("http"):
            continue
        if _is_filtered_link(href):
            continue

        display_name = ""
        for child in a.children:
            if isinstance(child, str):
                txt = child.strip()
                if txt and not txt.startswith("http"):
                    display_name = txt
                    break
        if not display_name:
            display_name = infer_name_from_url(href)

        if href not in unique:
            unique[href] = {"url": href, "display_name": display_name}

    return list(unique.values())


def extract_citations_fallback(soup0: BeautifulSoup) -> list[dict[str, str]]:
    """Fallback: extract display names from _0_ file citation spans (no real URLs)."""
    spans = soup0.select("span.citation.inline")
    results: list[dict[str, str]] = []
    for span in spans:
        text = normalize_text(span.get_text(" ", strip=True))
        text = re.sub(r"\+\d+$", "", text).strip()
        if text:
            results.append({
                "url": "",
                "display_name": text,
            })
    return results


def parse_perplexity_html(file_path: Path) -> dict[str, Any]:
    """Parse a Perplexity _0_ answer file (auto-finds _1_ pair for citations)."""
    raw0 = file_path.read_text(encoding="utf-8", errors="ignore")
    soup0 = BeautifulSoup(raw0, "lxml")

    title = normalize_text((soup0.title.string if soup0.title else "") or "")
    question = extract_question(soup0)
    answer_html = extract_answer_html(soup0)

    pair_file = find_pair_file(file_path)
    if pair_file and pair_file.exists():
        raw1 = pair_file.read_text(encoding="utf-8", errors="ignore")
        soup1 = BeautifulSoup(raw1, "lxml")
        citations = extract_citations_from_links_file(soup1)
    else:
        citations = extract_citations_fallback(soup0)

    return {
        "source_file": str(file_path),
        "pair_file": str(pair_file) if pair_file else "",
        "title": title,
        "question": question,
        "answer_html": answer_html,
        "citations": citations,
    }


def _is_answer_file(path: Path) -> bool:
    """Filter: only process _0_ files (answer pages) in batch mode."""
    idx = _get_file_index(path)
    return idx == "0"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Perplexity HTML answer + citations into JSON."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--file", type=str, help="Single _0_ HTML file path.")
    source_group.add_argument(
        "--dir", type=str, help="Directory for batch processing (recursive)."
    )
    source_group.add_argument(
        "--retry-failures", type=str,
        help="Path to _failures.jsonl to re-fetch and backfill.",
    )
    add_fetch_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.retry_failures:
        run_retry_failures(args)
    elif args.file:
        run_single_file(args, parse_perplexity_html, PLATFORM)
    else:
        run_batch(
            args, parse_perplexity_html, PLATFORM,
            "perplexity_json_output",
            file_filter=_is_answer_file,
        )


if __name__ == "__main__":
    main()
