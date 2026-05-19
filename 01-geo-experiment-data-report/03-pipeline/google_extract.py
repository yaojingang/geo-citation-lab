from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment

from fetch_utils import (
    add_fetch_args,
    infer_name_from_url,
    normalize_text,
    run_batch,
    run_retry_failures,
    run_single_file,
)

PLATFORM = "google"

GOOGLE_DOMAINS = {
    "google.com", "www.google.com", "gstatic.com", "www.gstatic.com",
    "googleapis.com", "accounts.google.com", "support.google.com",
    "play.google.com", "maps.google.com",
}


def _is_google_domain(url: str) -> bool:
    netloc = urlparse(url).netloc.lower()
    return any(netloc == d or netloc.endswith("." + d) for d in GOOGLE_DOMAINS)


def _parse_aria_label(aria: str) -> str:
    """Extract display name from aria-label like 'Title - Source. Opens in new tab.'"""
    aria = aria.replace(". Opens in new tab.", "").strip().rstrip(".")
    parts = aria.rsplit(" - ", 1)
    return parts[-1].strip() if len(parts) == 2 else aria


# ---------------------------------------------------------------------------
# Google AI Overview HTML parsing
# ---------------------------------------------------------------------------

def extract_question(soup: BeautifulSoup) -> str:
    title = soup.title.string if soup.title else ""
    title = normalize_text(title or "")
    suffix = " - Google Search"
    if title.endswith(suffix):
        title = title[: -len(suffix)].strip()
    return title


def extract_answer_html(soup: BeautifulSoup) -> str:
    mzjni = soup.select_one("div.mZJni")
    if mzjni:
        answer_div = mzjni.find("div", recursive=False)
        if answer_div:
            for comment in answer_div.find_all(string=lambda t: isinstance(t, Comment)):
                comment.extract()
            return answer_div.decode_contents().strip()

    ul = soup.select_one("ul.KsbFXc")
    if ul:
        container = ul.parent
        if container:
            for comment in container.find_all(string=lambda t: isinstance(t, Comment)):
                comment.extract()
            return container.decode_contents().strip()

    items = soup.select("li.dF3vjf")
    if items:
        return "".join(str(li) for li in items)

    return ""


def extract_citations(soup: BeautifulSoup) -> list[dict[str, str]]:
    anchors = soup.select("a.NDNGvf[href]")

    unique: dict[str, dict[str, str]] = {}
    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href.startswith("http"):
            continue
        if _is_google_domain(href):
            continue

        aria = a.get("aria-label", "")
        display_name = _parse_aria_label(aria) if aria else ""
        if not display_name:
            display_name = infer_name_from_url(href)

        if href not in unique:
            unique[href] = {"url": href, "display_name": display_name}

    return list(unique.values())


def parse_google_html(file_path: Path) -> dict[str, Any]:
    raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw_html, "lxml")

    title = normalize_text((soup.title.string if soup.title else "") or "")
    question = extract_question(soup)
    answer_html = extract_answer_html(soup)
    citations = extract_citations(soup)

    return {
        "source_file": str(file_path),
        "title": title,
        "question": question,
        "answer_html": answer_html,
        "citations": citations,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Google AI Overview HTML answer + citations into JSON."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--file", type=str, help="Single HTML file path.")
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
        run_single_file(args, parse_google_html, PLATFORM)
    else:
        run_batch(args, parse_google_html, PLATFORM, "google_json_output")


if __name__ == "__main__":
    main()
