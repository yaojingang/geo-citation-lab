# -*- coding: utf-8 -*-
"""
Large-scale citation feature extraction pipeline.

Reads enriched JSON records (answer_html + citations[].fetched_html) and
produces a flat feature table: one row per citation with ~50 dimension fields.

Supports two input modes:
  --dir   path/to/jsons/     recursive scan of .json files
  --jsonl path/to/index.jsonl  one enriched record per line

Outputs:
  features.jsonl   streaming, resumable
  features.csv     generated after completion
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import tiktoken
import google.generativeai as genai
from openai import OpenAI

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── API keys (read from environment for public-safe use) ──────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_EMB_MODEL = "text-embedding-3-small"
EMB_MAX_TOKENS = 8100  # leave 92-token margin for API overhead
EMB_BATCH_MAX_TOKENS = 150_000  # conservative limit (API overhead makes actual usage ~25% higher)
_tiktoken_enc: tiktoken.Encoding | None = None


def _truncate_to_tokens(text: str, max_tokens: int = EMB_MAX_TOKENS) -> str:
    """Truncate text to fit within max_tokens using tiktoken (cl100k_base)."""
    global _tiktoken_enc
    try:
        if _tiktoken_enc is None:
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        tokens = _tiktoken_enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return _tiktoken_enc.decode(tokens[:max_tokens])
    except Exception:
        return text[:max_tokens * 3]

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = "gemini-2.0-flash"


# ═══════════════════════════════════════════════════════════════════════════
# Text helpers
# ═══════════════════════════════════════════════════════════════════════════

def html_to_text(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(" ", strip=True)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z\u4e00-\u9fff]{2,}", text.lower())


def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "which", "who", "whom", "what", "where",
    "when", "how", "if", "then", "than", "more", "most", "also", "very",
})


# ═══════════════════════════════════════════════════════════════════════════
# A. Record context
# ═══════════════════════════════════════════════════════════════════════════

def make_record_id(source_file: str) -> str:
    return hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:16]


def infer_category(source_file: str) -> str:
    parts = Path(source_file).parts
    for p in parts:
        if p.startswith("A_") or p.startswith("B_") or p.startswith("C_") or p.startswith("D_"):
            return p
    return ""


_QUESTION_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"(?:^|[\s,])how (?:to|do|can|should|would)\b", "how_to"),
    (r"\bvs\.?\b|\bversus\b|\bcompare|difference between|better.+or\b", "comparison"),
    (r"(?:^|[\s,])what (?:are|is) the (?:main |best |top |different )?(?:types?|kinds?|categories|ways|methods|forms?|examples?)\b", "list"),
    (r"(?:^|[\s,])(?:list|name|give|provide|enumerate)\b.+(?:types?|examples?|ways|methods)", "list"),
    (r"(?:^|[\s,])which\b", "which"),
    (r"(?:^|[\s,])why\b", "why"),
    (r"(?:^|[\s,])what (?:is|are|does|do)\b|(?:^|[\s,])define\b", "what_is"),
    (r"(?:^|[\s,])(?:should|would|is it|do you)\b.+\?", "opinion"),
]


def classify_question_type(question: str) -> str:
    q = question.lower().strip()
    for pattern, qtype in _QUESTION_TYPE_PATTERNS:
        if re.search(pattern, q):
            return qtype
    return "other"


def extract_answer_structure(answer_soup: BeautifulSoup | None) -> dict[str, Any]:
    if answer_soup is None:
        return {"answer_heading_count": 0, "answer_list_item_count": 0, "answer_has_table": False}
    return {
        "answer_heading_count": len(answer_soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])),
        "answer_list_item_count": len(answer_soup.find_all("li")),
        "answer_has_table": bool(answer_soup.find("table")),
    }


# ═══════════════════════════════════════════════════════════════════════════
# B. Citation identity / URL features
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_TYPE_RULES: list[tuple[str, str]] = [
    ("wikipedia", "encyclopedia"),
    (".edu", "academic"),
    (".ac.", "academic"),
    ("library", "academic"),
    ("libguide", "academic"),
    (".gov", "government"),
    ("nih.gov", "government_health"),
    (".org", "nonprofit"),
    ("pewresearch", "research"),
    ("research", "research"),
    ("bbc", "news_media"),
    ("cnn", "news_media"),
    ("nytimes", "news_media"),
    ("reuters", "news_media"),
    ("news", "news_media"),
    ("ibm.com", "tech_corporate"),
    ("microsoft.com", "tech_corporate"),
    ("google.com", "tech_corporate"),
    ("amazon.com", "tech_corporate"),
    ("apple.com", "tech_corporate"),
    ("zhuanlan.zhihu", "ugc_platform"),
    ("medium.com", "ugc_platform"),
    ("reddit.com", "ugc_platform"),
    ("quora.com", "ugc_platform"),
    ("stackoverflow", "ugc_platform"),
    ("runoob.com", "tutorial"),
    ("w3schools", "tutorial"),
    ("geeksforgeeks", "tutorial"),
    ("pressbooks", "academic_publishing"),
    ("springer", "academic_publishing"),
    ("sciencedirect", "academic_publishing"),
    ("nature.com", "academic_publishing"),
]


def classify_domain(host: str) -> str:
    h = host.lower()
    for pattern, label in DOMAIN_TYPE_RULES:
        if pattern in h:
            return label
    if h.endswith(".edu"):
        return "academic"
    if h.endswith(".gov"):
        return "government"
    if h.endswith(".org"):
        return "nonprofit"
    return "commercial"


def extract_url_features(url: str, question_tokens: set[str]) -> dict[str, Any]:
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    path_parts = [p for p in parsed.path.split("/") if p]
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    path_text = " ".join(path_parts).lower().replace("-", " ").replace("_", " ")
    path_tokens = set(re.findall(r"[a-z]{3,}", path_text))
    url_kw_match = bool(question_tokens & path_tokens) if question_tokens else False

    return {
        "domain": host,
        "domain_tld": tld,
        "domain_type": classify_domain(host),
        "path_depth": len(path_parts),
        "url_has_query_keyword": url_kw_match,
    }


# ═══════════════════════════════════════════════════════════════════════════
# C. Citation page structure
# ═══════════════════════════════════════════════════════════════════════════

EMPTY_STRUCTURE: dict[str, Any] = {
    "cit_char_count": 0, "cit_word_count": 0,
    "cit_heading_h1": 0, "cit_heading_h2": 0, "cit_heading_h3": 0,
    "cit_heading_h4": 0, "cit_heading_h5": 0, "cit_heading_h6": 0,
    "cit_heading_total": 0,
    "cit_paragraph_count": 0, "cit_avg_para_words": 0.0,
    "cit_list_count": 0, "cit_list_item_count": 0, "cit_list_density": 0.0,
    "cit_table_count": 0, "cit_image_count": 0,
    "cit_link_count": 0, "cit_bold_count": 0,
    "cit_code_block_count": 0, "cit_has_code": False,
}


def extract_structure(soup: BeautifulSoup, text: str, words: list[str]) -> dict[str, Any]:
    headings = {i: len(soup.find_all(f"h{i}")) for i in range(1, 7)}
    paragraphs = soup.find_all("p")
    para_wc = [len(tokenize(p.get_text(" ", strip=True))) for p in paragraphs]
    li_items = soup.find_all("li")
    n_para = max(len(paragraphs), 1)

    return {
        "cit_char_count": len(text),
        "cit_word_count": len(words),
        **{f"cit_heading_h{i}": headings[i] for i in range(1, 7)},
        "cit_heading_total": sum(headings.values()),
        "cit_paragraph_count": len(paragraphs),
        "cit_avg_para_words": round(statistics.mean(para_wc), 1) if para_wc else 0.0,
        "cit_list_count": len(soup.find_all(["ul", "ol"])),
        "cit_list_item_count": len(li_items),
        "cit_list_density": round(len(li_items) / n_para, 3),
        "cit_table_count": len(soup.find_all("table")),
        "cit_image_count": len(soup.find_all("img")),
        "cit_link_count": len(soup.find_all("a", href=True)),
        "cit_bold_count": len(soup.find_all(["strong", "b"])),
        "cit_code_block_count": len(soup.find_all(["pre", "code"])),
        "cit_has_code": bool(soup.find(["pre", "code"])),
    }


# ═══════════════════════════════════════════════════════════════════════════
# D. Content quality signals
# ═══════════════════════════════════════════════════════════════════════════

def extract_quality(text: str, cit_title: str, question: str) -> dict[str, Any]:
    q_tokens = set(tokenize(question))
    t_tokens = set(tokenize(cit_title))
    title_match = round(len(t_tokens & q_tokens) / max(len(q_tokens), 1), 3) if q_tokens else 0.0

    all_tok = re.findall(r"\S+", text.lower())
    content_tok = [t for t in all_tok if t not in STOP_WORDS and len(t) > 2]
    info_density = round(len(content_tok) / max(len(all_tok), 1), 3)

    cn_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    total_alpha_cn = len(re.findall(r"[a-zA-Z\u4e00-\u9fff]", text))
    if total_alpha_cn > 0 and cn_chars / total_alpha_cn > 0.3:
        lang = "zh"
    else:
        lang = "en"

    sentences = re.split(r"[.!?。！？]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if sentences:
        sent_word_counts = [len(s.split()) for s in sentences]
        avg_sent_len = round(statistics.mean(sent_word_counts), 1)
    else:
        avg_sent_len = 0.0

    word_list = re.findall(r"[a-zA-Z\u4e00-\u9fff]+", text.lower())
    unique_ratio = round(len(set(word_list)) / max(len(word_list), 1), 4)

    return {
        "title_question_match": title_match,
        "cit_info_density": info_density,
        "cit_has_numbers": bool(re.search(r"\d+[\.,]?\d*\s*%|\$\s*[\d,]+|\d{4,}", text)),
        "cit_has_definition": bool(re.search(r"(?:is a |refers to |defined as |means )", text, re.I)),
        "cit_has_qa_format": bool(re.search(r"\?\s*\n|^what |^how |^why |^when ", text, re.I | re.M)),
        "cit_has_howto": bool(re.search(r"step \d|how to |first\b.*\bthen\b", text, re.I)),
        "cit_has_comparison": bool(re.search(r"\bvs\.?\b|compared to |versus |difference between ", text, re.I)),
        "cit_language": lang,
        "cit_avg_sentence_length": avg_sent_len,
        "cit_unique_word_ratio": unique_ratio,
    }


# ═══════════════════════════════════════════════════════════════════════════
# E. Influence in answer
# ═══════════════════════════════════════════════════════════════════════════

def extract_influence(answer_soup: BeautifulSoup, citation_url: str) -> dict[str, Any]:
    clean_url = citation_url.split("?")[0]
    all_links = answer_soup.find_all("a", href=True)

    positions = []
    for idx, a in enumerate(all_links):
        href = (a.get("href") or "").split("?")[0]
        if href == clean_url or citation_url in (a.get("href") or ""):
            positions.append(idx)

    answer_blocks = answer_soup.find_all(["p", "li", "h3", "h4"])
    covered = set()
    for bidx, block in enumerate(answer_blocks):
        for link in block.find_all("a", href=True):
            href = (link.get("href") or "").split("?")[0]
            if href == clean_url or citation_url in (link.get("href") or ""):
                covered.add(bidx)

    n_blocks = max(len(answer_blocks), 1)
    n_links = max(len(all_links), 1)

    return {
        "ref_count": len(positions),
        "first_position_ratio": round(positions[0] / n_links, 3) if positions else 1.0,
        "covered_paragraphs": len(covered),
        "total_answer_paragraphs": len(answer_blocks),
        "paragraph_coverage_ratio": round(len(covered) / n_blocks, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# F. Text similarity
# ═══════════════════════════════════════════════════════════════════════════

def extract_similarity(answer_text: str, cit_text: str,
                       answer_tokens: list[str]) -> dict[str, float]:
    if not answer_text.strip() or not cit_text.strip():
        return {"tfidf_cosine": 0.0, "bigram_overlap": 0.0, "trigram_overlap": 0.0}

    try:
        vec = TfidfVectorizer(max_features=5000, stop_words="english")
        mat = vec.fit_transform([answer_text, cit_text])
        cos = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
    except ValueError:
        cos = 0.0

    cit_tokens = tokenize(cit_text)
    a_bi = set(ngrams(answer_tokens, 2))
    c_bi = set(ngrams(cit_tokens, 2))
    a_tri = set(ngrams(answer_tokens, 3))
    c_tri = set(ngrams(cit_tokens, 3))

    return {
        "tfidf_cosine": round(cos, 4),
        "bigram_overlap": round(len(a_bi & c_bi) / max(len(a_bi), 1), 4),
        "trigram_overlap": round(len(a_tri & c_tri) / max(len(a_tri), 1), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# G. Influence score
# ═══════════════════════════════════════════════════════════════════════════

W_REF = 0.20
W_POS = 0.15
W_COV = 0.20
W_TFIDF = 0.25
W_NGRAM = 0.20


def compute_influence_score(infl: dict, sim: dict) -> float:
    pos_score = 1.0 - infl["first_position_ratio"]
    ngram_score = (sim["bigram_overlap"] + sim["trigram_overlap"]) / 2
    score = (
        W_REF * min(infl["ref_count"] / 3.0, 1.0)
        + W_POS * pos_score
        + W_COV * infl["paragraph_coverage_ratio"]
        + W_TFIDF * sim["tfidf_cosine"]
        + W_NGRAM * ngram_score
    )
    return round(score, 4)


# ═══════════════════════════════════════════════════════════════════════════
# I. Embedding similarity (OpenAI text-embedding-3-small)
# ═══════════════════════════════════════════════════════════════════════════

_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("Missing OPENAI_API_KEY.")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_emb_cache(cache_dir: Path, text_hash: str) -> list[float] | None:
    p = cache_dir / f"{text_hash}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _save_emb_cache(cache_dir: Path, text_hash: str, vec: list[float]) -> None:
    p = cache_dir / f"{text_hash}.json"
    p.write_text(json.dumps(vec), encoding="utf-8")


def _count_tokens(text: str) -> int:
    """Count tokens for a text using tiktoken (cl100k_base)."""
    global _tiktoken_enc
    try:
        if _tiktoken_enc is None:
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        return len(_tiktoken_enc.encode(text))
    except Exception:
        return len(text) // 3


def _batch_embed(texts: list[str], cache_dir: Path) -> dict[str, list[float]]:
    """Embed a list of texts, returning {text_hash: vector}. Uses disk cache."""
    result: dict[str, list[float]] = {}
    to_embed: list[str] = []
    to_embed_hashes: list[str] = []
    to_embed_tokens: list[int] = []

    for text in texts:
        truncated = _truncate_to_tokens(text)
        h = _text_hash(truncated)
        cached = _load_emb_cache(cache_dir, h)
        if cached is not None:
            result[h] = cached
        else:
            to_embed.append(truncated)
            to_embed_hashes.append(h)
            to_embed_tokens.append(_count_tokens(truncated))

    if not to_embed:
        return result

    cached_count = len(result)
    print(f"[embedding] {cached_count} cached, {len(to_embed)} to call API")
    client = _get_openai_client()

    # Dynamic batching: split by total tokens per request
    i = 0
    batch_num = 0
    while i < len(to_embed):
        batch: list[str] = []
        batch_hashes: list[str] = []
        batch_tok = 0
        while i < len(to_embed) and (batch_tok + to_embed_tokens[i] <= EMB_BATCH_MAX_TOKENS or not batch):
            batch.append(to_embed[i])
            batch_hashes.append(to_embed_hashes[i])
            batch_tok += to_embed_tokens[i]
            i += 1

        for attempt in range(3):
            try:
                resp = client.embeddings.create(input=batch, model=OPENAI_EMB_MODEL)
                for j, emb_data in enumerate(resp.data):
                    vec = emb_data.embedding
                    h = batch_hashes[j]
                    result[h] = vec
                    _save_emb_cache(cache_dir, h, vec)
                break
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"[embedding] retry in {wait}s ({len(batch)} texts, ~{batch_tok} tok): {e}", file=sys.stderr)
                    time.sleep(wait)
                else:
                    print(f"[embedding error] batch {batch_num} ({len(batch)} texts, ~{batch_tok} tok): {e}", file=sys.stderr)
                    for h in batch_hashes:
                        if h not in result:
                            result[h] = []
        batch_num += 1

    return result


def _cosine_np(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va, vb = np.asarray(a), np.asarray(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def enrich_embeddings(
    all_rows: list[dict[str, Any]],
    text_store: dict[str, dict],
    cache_dir: Path,
) -> None:
    """Add emb_answer_cit_cosine and emb_question_cit_cosine to every row."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not OPENAI_API_KEY:
        print("[embedding] OPENAI_API_KEY not set; skipping embedding enrichment.", file=sys.stderr)
        for row in all_rows:
            row["emb_answer_cit_cosine"] = 0.0
            row["emb_question_cit_cosine"] = 0.0
        return

    unique_texts: dict[str, str] = {}
    for store in text_store.values():
        for text in [store["question"], store["answer_text"]]:
            if text.strip():
                t = _truncate_to_tokens(text)
                unique_texts[_text_hash(t)] = t
        for cit_text in store["cit_texts"].values():
            if cit_text.strip():
                t = _truncate_to_tokens(cit_text)
                unique_texts[_text_hash(t)] = t

    print(f"[embedding] {len(unique_texts)} unique texts")
    embeddings = _batch_embed(list(unique_texts.values()), cache_dir)

    for row in all_rows:
        rid = row["record_id"]
        cidx = row["citation_index"]
        store = text_store.get(rid)
        if store is None:
            row["emb_answer_cit_cosine"] = 0.0
            row["emb_question_cit_cosine"] = 0.0
            continue

        ans_h = _text_hash(_truncate_to_tokens(store["answer_text"]))
        q_h = _text_hash(_truncate_to_tokens(store["question"]))
        cit_text = store["cit_texts"].get(cidx, "")
        cit_h = _text_hash(_truncate_to_tokens(cit_text))

        ans_vec = embeddings.get(ans_h, [])
        q_vec = embeddings.get(q_h, [])
        cit_vec = embeddings.get(cit_h, [])

        row["emb_answer_cit_cosine"] = round(_cosine_np(ans_vec, cit_vec), 4)
        row["emb_question_cit_cosine"] = round(_cosine_np(q_vec, cit_vec), 4)

    print(f"[embedding] done — {len(all_rows)} rows enriched")


# ═══════════════════════════════════════════════════════════════════════════
# J. LLM semantic analysis (Gemini 2.0 Flash)
# ═══════════════════════════════════════════════════════════════════════════

_GEMINI_PROMPT_TEMPLATE = """\
Analyze the citations used by an AI assistant to answer a question.

Question: {question}

Answer (first 1500 chars):
{answer_excerpt}

Citations:
{citations_block}

For each citation, determine:
1. semantic_role — the function this citation serves. Choose ONE of: \
definition, evidence, example, background, procedure, statistical_data, \
comparison, opinion, reference
2. relevance_score — integer 1-5, how relevant the citation is to the \
question (5 = highly relevant)
3. content_quality — integer 1-5, how authoritative / well-written the \
content is (5 = very high quality)
4. influence_type — how this citation was used in the answer. Choose ONE \
of: direct_quote, paraphrase, factual_basis, structural_guide, supplementary

Return ONLY a valid JSON array (no markdown fences, no other text):
[{{"idx": 0, "semantic_role": "...", "relevance_score": N, \
"content_quality": N, "influence_type": "..."}}, ...]"""

_EMPTY_LLM = {
    "llm_semantic_role": "",
    "llm_relevance_score": 0,
    "llm_content_quality": 0,
    "llm_influence_type": "",
}

_gemini_model = None


def _get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("Missing GEMINI_API_KEY.")
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    return _gemini_model


def _build_gemini_prompt(question: str, answer_text: str,
                         cit_infos: list[dict]) -> str:
    parts: list[str] = []
    for i, c in enumerate(cit_infos):
        title = c.get("title", "N/A")
        url = c.get("url", "")
        text = c.get("text", "")[:800]
        parts.append(f"[{i}] {title} | {url}\nContent (first 800 chars): {text}")
    return _GEMINI_PROMPT_TEMPLATE.format(
        question=question,
        answer_excerpt=answer_text[:1500],
        citations_block="\n\n".join(parts),
    )


def _call_gemini(prompt: str) -> list[dict]:
    model = _get_gemini_model()
    for attempt in range(3):
        try:
            resp = model.generate_content(prompt)
            text = resp.text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text).strip()
            return json.loads(text)
        except Exception as e:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"[llm] retry in {wait}s: {e}", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"[llm error] {e}", file=sys.stderr)
                return []


def enrich_llm(
    all_rows: list[dict[str, Any]],
    text_store: dict[str, dict],
    cache_dir: Path,
) -> None:
    """Add llm_* fields to every row via Gemini analysis (one call per record)."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not GEMINI_API_KEY:
        print("[llm] GEMINI_API_KEY not set; skipping LLM enrichment.", file=sys.stderr)
        for row in all_rows:
            row.update(_EMPTY_LLM)
        return

    by_record: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_record.setdefault(row["record_id"], []).append(row)

    cached, called = 0, 0
    for rid, rows in tqdm(by_record.items(), desc="llm analysis", unit="rec"):
        cache_file = cache_dir / f"{rid}.json"

        if cache_file.exists():
            analysis = json.loads(cache_file.read_text(encoding="utf-8"))
            cached += 1
        else:
            store = text_store.get(rid)
            if store is None:
                analysis = []
            else:
                cit_infos: list[dict] = []
                for r in sorted(rows, key=lambda x: x["citation_index"]):
                    cidx = r["citation_index"]
                    cit_infos.append({
                        "title": r.get("cit_title", ""),
                        "url": r.get("url", ""),
                        "text": store["cit_texts"].get(cidx, ""),
                    })
                prompt = _build_gemini_prompt(
                    store["question"], store["answer_text"], cit_infos,
                )
                analysis = _call_gemini(prompt)
            cache_file.write_text(
                json.dumps(analysis, ensure_ascii=False), encoding="utf-8",
            )
            called += 1

        analysis_by_idx: dict[int, dict] = {}
        if isinstance(analysis, list):
            for item in analysis:
                if isinstance(item, dict) and "idx" in item:
                    analysis_by_idx[item["idx"]] = item

        for r in rows:
            cidx = r["citation_index"]
            item = analysis_by_idx.get(cidx, {})
            r["llm_semantic_role"] = item.get("semantic_role", "")
            r["llm_relevance_score"] = item.get("relevance_score", 0)
            r["llm_content_quality"] = item.get("content_quality", 0)
            r["llm_influence_type"] = item.get("influence_type", "")

    print(f"[llm] done — {cached} cached, {called} API calls")


# ═══════════════════════════════════════════════════════════════════════════
# Core: process one record → list of citation feature dicts
# ═══════════════════════════════════════════════════════════════════════════

MAX_HTML_SIZE = 5_000_000  # 5 MB limit to prevent lxml crashes


def _resolve_fetched_html(cit: dict[str, Any]) -> str:
    """Get fetched_html, falling back to fetched_html_path if needed."""
    html = cit.get("fetched_html", "")
    if html:
        return html[:MAX_HTML_SIZE] if len(html) > MAX_HTML_SIZE else html
    path_str = cit.get("fetched_html_path", "")
    if path_str:
        p = Path(path_str)
        if p.exists():
            content = p.read_text(encoding="utf-8", errors="ignore")
            return content[:MAX_HTML_SIZE] if len(content) > MAX_HTML_SIZE else content
    return ""


def extract_record_features(
    record: dict[str, Any],
    text_store: dict[str, dict] | None = None,
) -> list[dict[str, Any]]:
    source_file = record.get("source_file", "")
    record_id = make_record_id(source_file)
    platform = record.get("platform", "")
    category = record.get("category", "") or infer_category(source_file)
    question = record.get("question", "")
    answer_html = record.get("answer_html", "")
    citations = record.get("citations", [])

    answer_text = html_to_text(answer_html)
    answer_tokens = tokenize(answer_text)
    answer_soup = BeautifulSoup(answer_html, "lxml") if answer_html else None
    question_tokens = set(tokenize(question))

    if text_store is not None:
        text_store[record_id] = {
            "question": question,
            "answer_text": answer_text,
            "cit_texts": {},
        }

    context = {
        "record_id": record_id,
        "platform": platform,
        "category": category,
        "question": question,
        "question_type": classify_question_type(question),
        "answer_word_count": len(answer_tokens),
        "answer_char_count": len(answer_text),
        "total_citations": len(citations),
    }
    context.update(extract_answer_structure(answer_soup))

    rows: list[dict[str, Any]] = []

    for idx, cit in enumerate(citations):
        url = cit.get("url", "")
        fetched_html = _resolve_fetched_html(cit)
        has_content = bool(fetched_html and not cit.get("fetch_error"))
        cit_title = cit.get("title", "")

        row: dict[str, Any] = {**context, "citation_index": idx}

        # B: identity
        row["url"] = url
        row["display_name"] = cit.get("display_name", "")
        row["cit_title"] = cit_title
        row["fetch_ok"] = has_content
        row["status_code"] = cit.get("status_code")
        row["error_class"] = cit.get("error_class", "")
        row.update(extract_url_features(url, question_tokens))

        if has_content:
            cit_soup = BeautifulSoup(fetched_html, "lxml")
            cit_text = cit_soup.get_text(" ", strip=True)
            cit_words = tokenize(cit_text)

            # C: structure
            row.update(extract_structure(cit_soup, cit_text, cit_words))
            # D: quality
            row.update(extract_quality(cit_text, cit_title, question))
        else:
            cit_text = ""
            row.update(EMPTY_STRUCTURE)
            row.update({
                "title_question_match": 0.0, "cit_info_density": 0.0,
                "cit_has_numbers": False, "cit_has_definition": False,
                "cit_has_qa_format": False, "cit_has_howto": False,
                "cit_has_comparison": False, "cit_language": "",
                "cit_avg_sentence_length": 0.0, "cit_unique_word_ratio": 0.0,
            })

        if text_store is not None:
            text_store[record_id]["cit_texts"][idx] = cit_text

        # E: influence
        if answer_soup:
            infl = extract_influence(answer_soup, url)
        else:
            infl = {
                "ref_count": 0, "first_position_ratio": 1.0,
                "covered_paragraphs": 0, "total_answer_paragraphs": 0,
                "paragraph_coverage_ratio": 0.0,
            }
        row.update(infl)

        # F: similarity
        sim = extract_similarity(answer_text, cit_text, answer_tokens) if has_content else {
            "tfidf_cosine": 0.0, "bigram_overlap": 0.0, "trigram_overlap": 0.0,
        }
        row.update(sim)

        # G: score
        row["influence_score"] = compute_influence_score(infl, sim)
        rows.append(row)

    # Rank within record
    rows.sort(key=lambda r: r["influence_score"], reverse=True)
    for rank, r in enumerate(rows, 1):
        r["influence_rank"] = rank

    # H: intra-record competition context
    domain_counter: dict[str, int] = {}
    for r in rows:
        d = r.get("domain", "")
        domain_counter[d] = domain_counter.get(d, 0) + 1

    ref_counts_sorted = sorted([r["ref_count"] for r in rows], reverse=True)
    for r in rows:
        d = r.get("domain", "")
        r["same_domain_count"] = domain_counter.get(d, 0)
        r["domain_is_unique"] = domain_counter.get(d, 0) == 1
        r["rank_by_ref_count"] = ref_counts_sorted.index(r["ref_count"]) + 1

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# I/O: input loading
# ═══════════════════════════════════════════════════════════════════════════

def iter_records_from_dir(dir_path: Path):
    """Yield (identifier, record_dict) from a directory of .json files."""
    for jp in sorted(dir_path.rglob("*.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] skip {jp}: {e}", file=sys.stderr)
            continue
        if "citations" not in data:
            continue
        yield str(jp), data


def iter_records_from_jsonl(jsonl_path: Path):
    """Yield (identifier, record_dict) from a JSONL file."""
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] skip line {line_no}: {e}", file=sys.stderr)
                continue
            if "citations" not in data:
                continue
            ident = data.get("source_file", f"line:{line_no}")
            yield ident, data


# ═══════════════════════════════════════════════════════════════════════════
# I/O: output / resume
# ═══════════════════════════════════════════════════════════════════════════

def load_done_ids(jsonl_path: Path) -> set[str]:
    """Read existing features.jsonl to find already-processed record_ids."""
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done.add(row.get("record_id", ""))
            except json.JSONDecodeError:
                continue
    return done


def jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> int:
    """Convert features.jsonl to features.csv."""
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return 0
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Worker wrapper for multiprocessing
# ═══════════════════════════════════════════════════════════════════════════

def _process_one(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Standalone function for ProcessPoolExecutor."""
    try:
        return extract_record_features(record)
    except Exception as exc:
        src = record.get("source_file", "?")
        print(f"[error] {src}: {exc}", file=sys.stderr)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# CLI & main
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract citation features from enriched JSON records."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", type=str, help="Directory of enriched .json files.")
    src.add_argument("--jsonl", type=str, help="JSONL file (one record per line).")
    p.add_argument("--output", type=str, default="features",
                   help="Output directory (default: features/).")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (default: 1, single-process).")
    p.add_argument("--embedding", action="store_true",
                   help="Enrich with OpenAI embedding cosine similarity.")
    p.add_argument("--llm", action="store_true",
                   help="Enrich with Gemini LLM semantic analysis.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_out = output_dir / "features.jsonl"
    csv_out = output_dir / "features.csv"

    use_api = args.embedding or args.llm

    # Load resume state
    done_ids = load_done_ids(jsonl_out)
    if done_ids:
        print(f"[resume] found {len(done_ids)} already-processed record(s), skipping them")

    # Collect input records
    if args.dir:
        source_iter = list(iter_records_from_dir(Path(args.dir)))
    else:
        source_iter = list(iter_records_from_jsonl(Path(args.jsonl)))

    # Filter already-done
    pending = []
    for ident, record in source_iter:
        rid = make_record_id(record.get("source_file", ident))
        if rid not in done_ids:
            pending.append(record)

    total = len(pending)
    print(f"[input] {len(source_iter)} records found, {total} to process")
    if total == 0:
        print("[done] nothing to process")
        if jsonl_out.exists():
            n = jsonl_to_csv(jsonl_out, csv_out)
            print(f"[csv] {csv_out} ({n} rows)")
        return

    t0 = time.monotonic()

    if use_api:
        # ── API-enriched path: collect everything in memory, then enrich ──
        all_rows: list[dict[str, Any]] = []
        text_store: dict[str, dict] = {}

        for record in tqdm(pending, desc="local features", unit="rec"):
            try:
                rows = extract_record_features(record, text_store=text_store)
            except Exception as exc:
                src = record.get("source_file", "?")
                print(f"[error] {src}: {exc}", file=sys.stderr)
                rows = []
            all_rows.extend(rows)

        print(f"[phase1] {len(all_rows)} citation rows extracted locally")

        if args.embedding:
            emb_cache = output_dir / "_emb_cache"
            enrich_embeddings(all_rows, text_store, emb_cache)

        if args.llm:
            llm_cache = output_dir / "_llm_cache"
            enrich_llm(all_rows, text_store, llm_cache)

        with jsonl_out.open("a", encoding="utf-8") as out_f:
            for r in all_rows:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        row_count = len(all_rows)

    else:
        # ── Original streaming path (supports multiprocessing) ────────────
        row_count = 0
        workers = max(1, args.workers)

        with jsonl_out.open("a", encoding="utf-8") as out_f:
            if workers == 1:
                for record in tqdm(pending, desc="extracting", unit="rec"):
                    rows = _process_one(record)
                    for r in rows:
                        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        row_count += 1
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_process_one, rec): i
                               for i, rec in enumerate(pending)}
                    for future in tqdm(as_completed(futures), total=total,
                                       desc="extracting", unit="rec"):
                        rows = future.result()
                        for r in rows:
                            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                            row_count += 1

    elapsed = time.monotonic() - t0
    print(f"[done] {total} records, {row_count} citation rows, {elapsed:.1f}s")

    # Convert to CSV
    n = jsonl_to_csv(jsonl_out, csv_out)
    print(f"[csv] {csv_out} ({n} rows)")


if __name__ == "__main__":
    main()
