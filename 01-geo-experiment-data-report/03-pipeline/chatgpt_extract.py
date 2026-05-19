from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from fetch_utils import (
    add_fetch_args,
    infer_name_from_url,
    normalize_text,
    run_batch,
    run_retry_failures,
    run_single_file,
    strip_badges,
)

PLATFORM = "chatgpt"


# ---------------------------------------------------------------------------
# ChatGPT HTML parsing
# ---------------------------------------------------------------------------

def find_question_in_text(text: str) -> str | None:
    text = normalize_text(text)
    if not text:
        return None
    for marker in ("你说：", "You said:"):
        idx = text.find(marker)
        if idx != -1:
            rest = text[idx + len(marker):].strip()
            for tail in ("ChatGPT 说：", "ChatGPT said:"):
                j = rest.find(tail)
                if j != -1:
                    rest = rest[:j].strip()
            return rest or None
    return None


def extract_question(raw_html: str, soup: BeautifulSoup) -> str:
    candidate_selectors = [
        '[data-turn="user"] [data-message-author-role="user"]',
        '[data-turn="user"] .whitespace-pre-wrap',
        '[data-turn="user"]',
    ]
    for selector in candidate_selectors:
        node = soup.select_one(selector)
        if not node:
            continue
        question = find_question_in_text(node.get_text(" ", strip=True))
        if question:
            return question

    regex_patterns = [
        r"(?:你说：|You said:)\s*(.+?)\s*(?:ChatGPT 说：|ChatGPT said:|data-turn=\"assistant\")",
        r"data-turn=\"user\".{0,5000}?(?:你说：|You said:)\s*(.+?)<",
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, raw_html, flags=re.S | re.I)
        if not match:
            continue
        question = normalize_text(
            BeautifulSoup(match.group(1), "lxml").get_text(" ", strip=True)
        )
        if question:
            return question
    return ""


def extract_answer_html(soup: BeautifulSoup) -> str:
    assistant_turn = soup.select_one('[data-turn="assistant"]')
    search_root = assistant_turn if assistant_turn else soup

    answer_node = search_root.select_one(".markdown")
    if answer_node:
        return answer_node.decode_contents().strip()

    p_nodes = search_root.select("p[data-start]")
    if p_nodes:
        return "".join(str(p) for p in p_nodes)
    return ""


def extract_citations(soup: BeautifulSoup) -> list[dict[str, str]]:
    assistant_turn = soup.select_one('[data-turn="assistant"]')
    search_root = assistant_turn if assistant_turn else soup

    citation_anchors = search_root.select(
        '[data-testid="webpage-citation-pill"] a[href]'
    )
    if not citation_anchors:
        citation_anchors = search_root.select("a[href]")

    unique: dict[str, dict[str, str]] = {}
    for anchor in citation_anchors:
        url = (anchor.get("href") or "").strip()
        if not url.startswith("http"):
            continue
        label = strip_badges(anchor.get_text(" ", strip=True))
        if "chatgpt.com/c/" in url and "utm_source=chatgpt.com" not in url:
            continue
        if "/cdn/assets/" in url:
            continue
        if not label:
            label = infer_name_from_url(url)
        if url not in unique:
            unique[url] = {"url": url, "display_name": label}
    return list(unique.values())


def parse_chatgpt_html(file_path: Path) -> dict[str, Any]:
    raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw_html, "lxml")

    title_node = soup.title.string if soup.title else ""
    title = normalize_text(title_node or "")
    question = extract_question(raw_html, soup)
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
        description="Extract ChatGPT HTML answer + citations into JSON."
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
        run_single_file(args, parse_chatgpt_html, PLATFORM)
    else:
        run_batch(args, parse_chatgpt_html, PLATFORM, "chatgpt_json_output")


if __name__ == "__main__":
    main()
