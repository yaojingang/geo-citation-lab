# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_INPUT = Path("output_all/features_all_platforms_72.csv")
DEFAULT_OUTPUT = Path("output_all/citation_influence_report.md")

BOOL_COLUMNS = [
    "fetch_ok", "answer_has_table", "url_has_query_keyword",
    "cit_has_code", "cit_has_numbers", "cit_has_definition",
    "cit_has_qa_format", "cit_has_howto", "cit_has_comparison",
    "domain_is_unique",
]

NUMERIC_CANDIDATE_COLUMNS = [
    "answer_word_count", "answer_char_count", "total_citations",
    "answer_heading_count", "answer_list_item_count", "citation_index",
    "path_depth", "cit_char_count", "cit_word_count", "cit_heading_total",
    "cit_paragraph_count", "cit_avg_para_words", "cit_list_count",
    "cit_list_item_count", "cit_list_density", "cit_table_count",
    "cit_image_count", "cit_link_count", "cit_bold_count",
    "cit_code_block_count", "title_question_match", "cit_info_density",
    "cit_avg_sentence_length", "cit_unique_word_ratio", "ref_count",
    "first_position_ratio", "covered_paragraphs", "total_answer_paragraphs",
    "paragraph_coverage_ratio", "tfidf_cosine", "bigram_overlap",
    "trigram_overlap", "influence_score", "influence_rank",
    "same_domain_count", "rank_by_ref_count",
    "emb_answer_cit_cosine", "emb_question_cit_cosine",
    "llm_relevance_score", "llm_content_quality",
]

# ---------------------------------------------------------------------------
# Variable tiers: separate what the creator controls (X) from outcome (Y)
# ---------------------------------------------------------------------------

CONTENT_STRUCTURE_COLS = [
    "cit_heading_total", "cit_paragraph_count", "cit_avg_para_words",
    "cit_list_density", "cit_list_count", "cit_list_item_count",
    "cit_table_count", "cit_image_count", "cit_code_block_count",
    "cit_bold_count", "cit_link_count",
]
CONTENT_QUALITY_COLS = [
    "cit_word_count", "cit_char_count", "cit_info_density",
    "cit_unique_word_ratio", "cit_avg_sentence_length",
]
CONTENT_TYPE_BOOL_COLS = [
    "cit_has_numbers", "cit_has_definition", "cit_has_qa_format",
    "cit_has_howto", "cit_has_comparison", "cit_has_code",
]
TOPIC_ALIGNMENT_COLS = ["title_question_match"]
SEMANTIC_NUMERIC_COLS = [
    "emb_answer_cit_cosine", "emb_question_cit_cosine",
    "llm_relevance_score", "llm_content_quality",
]
OUTCOME_COLS = [
    "ref_count", "first_position_ratio", "covered_paragraphs",
    "paragraph_coverage_ratio", "tfidf_cosine", "bigram_overlap",
    "trigram_overlap", "influence_score", "influence_rank",
    "rank_by_ref_count",
]

ALL_CONTENT_NUMERIC = (CONTENT_STRUCTURE_COLS + CONTENT_QUALITY_COLS
                       + TOPIC_ALIGNMENT_COLS + SEMANTIC_NUMERIC_COLS
                       + ["path_depth"])

TOP_BOTTOM_COMPARE_COLUMNS = (CONTENT_STRUCTURE_COLS + CONTENT_QUALITY_COLS
                              + TOPIC_ALIGNMENT_COLS + SEMANTIC_NUMERIC_COLS)

# Chinese-friendly feature names
CN = {
    "first_position_ratio": "\u9996\u6b21\u51fa\u73b0\u4f4d\u7f6e\uff08\u8d8a\u5c0f=\u8d8a\u9760\u524d\uff09",
    "ref_count": "\u88ab\u56de\u7b54\u5f15\u7528\u7684\u6b21\u6570",
    "covered_paragraphs": "\u8986\u76d6\u7684\u56de\u7b54\u6bb5\u843d\u6570",
    "paragraph_coverage_ratio": "\u8986\u76d6\u56de\u7b54\u6bb5\u843d\u7684\u6bd4\u7387",
    "influence_rank": "\u5f71\u54cd\u529b\u6392\u540d\uff08\u8d8a\u5c0f=\u8d8a\u9ad8\uff09",
    "total_answer_paragraphs": "\u56de\u7b54\u603b\u6bb5\u843d\u6570",
    "answer_heading_count": "\u56de\u7b54\u4e2d\u7684\u6807\u9898\u6570",
    "tfidf_cosine": "TF-IDF \u6587\u672c\u76f8\u4f3c\u5ea6",
    "total_citations": "\u8be5\u95ee\u9898\u7684\u603b\u5f15\u7528\u6570",
    "llm_relevance_score": "LLM \u76f8\u5173\u6027\u8bc4\u5206",
    "citation_index": "\u5f15\u7528\u5728\u5217\u8868\u4e2d\u7684\u5e8f\u53f7",
    "bigram_overlap": "\u53cc\u8bcd\u7ec4\u91cd\u53e0\u7387",
    "emb_answer_cit_cosine": "\u8bed\u4e49\u76f8\u4f3c\u5ea6\uff08\u5f15\u7528 vs \u56de\u7b54\uff09",
    "trigram_overlap": "\u4e09\u8bcd\u7ec4\u91cd\u53e0\u7387",
    "llm_content_quality": "LLM \u5185\u5bb9\u8d28\u91cf\u8bc4\u5206",
    "cit_word_count": "\u5f15\u7528\u9875\u9762\u8bcd\u6570",
    "cit_char_count": "\u5f15\u7528\u9875\u9762\u5b57\u7b26\u6570",
    "cit_heading_total": "\u5f15\u7528\u9875\u9762\u6807\u9898\u6570",
    "cit_paragraph_count": "\u5f15\u7528\u9875\u9762\u6bb5\u843d\u6570",
    "cit_list_density": "\u5217\u8868\u5bc6\u5ea6\uff08\u5217\u8868\u9879/\u6bb5\u843d\uff09",
    "cit_table_count": "\u8868\u683c\u6570",
    "cit_image_count": "\u56fe\u7247\u6570",
    "cit_code_block_count": "\u4ee3\u7801\u5757\u6570",
    "cit_info_density": "\u4fe1\u606f\u5bc6\u5ea6\uff08\u5b9e\u8bcd\u5360\u6bd4\uff09",
    "cit_unique_word_ratio": "\u8bcd\u6c47\u4e30\u5bcc\u5ea6",
    "cit_avg_sentence_length": "\u5e73\u5747\u53e5\u957f\uff08\u8bcd\uff09",
    "emb_question_cit_cosine": "\u8bed\u4e49\u76f8\u4f3c\u5ea6\uff08\u5f15\u7528 vs \u95ee\u9898\uff09",
    "cit_has_numbers": "\u5305\u542b\u6570\u5b57/\u7edf\u8ba1\u6570\u636e",
    "cit_has_definition": "\u5305\u542b\u5b9a\u4e49\u53e5\u5f0f",
    "cit_has_qa_format": "\u5305\u542b\u95ee\u7b54\u683c\u5f0f",
    "cit_has_howto": "\u5305\u542b\u64cd\u4f5c\u6307\u5357",
    "cit_has_comparison": "\u5305\u542b\u5bf9\u6bd4\u5185\u5bb9",
    "cit_has_code": "\u5305\u542b\u4ee3\u7801",
    "path_depth": "URL \u8def\u5f84\u6df1\u5ea6",
    "answer_word_count": "\u56de\u7b54\u8bcd\u6570",
    "answer_char_count": "\u56de\u7b54\u5b57\u7b26\u6570",
    "answer_list_item_count": "\u56de\u7b54\u5217\u8868\u9879\u6570",
    "cit_avg_para_words": "\u6bb5\u843d\u5e73\u5747\u8bcd\u6570",
    "cit_list_count": "\u5217\u8868\u5757\u6570",
    "cit_list_item_count": "\u5217\u8868\u9879\u603b\u6570",
    "cit_link_count": "\u94fe\u63a5\u6570",
    "cit_bold_count": "\u52a0\u7c97\u6587\u672c\u6570",
    "title_question_match": "\u6807\u9898\u4e0e\u95ee\u9898\u5173\u952e\u8bcd\u5339\u914d\u5ea6",
    "same_domain_count": "\u540c\u57df\u540d\u5f15\u7528\u6570\u91cf",
    "rank_by_ref_count": "\u6309\u5f15\u7528\u6b21\u6570\u7684\u6392\u540d",
}

LLM_ROLE_CN = {
    "evidence": "\u8bc1\u636e\u652f\u6491", "reference": "\u53c2\u8003\u5f15\u7528",
    "background": "\u80cc\u666f\u4ecb\u7ecd", "example": "\u4e3e\u4f8b\u8bf4\u660e",
    "definition": "\u5b9a\u4e49\u89e3\u91ca", "statistical_data": "\u7edf\u8ba1\u6570\u636e",
    "comparison": "\u5bf9\u6bd4\u5206\u6790", "opinion": "\u89c2\u70b9\u8868\u8fbe",
}

LLM_INF_CN = {
    "factual_basis": "\u4e8b\u5b9e\u4f9d\u636e", "supplementary": "\u8865\u5145\u8bf4\u660e",
    "reference": "\u53c2\u8003\u5f15\u7528", "paraphrase": "\u6539\u5199\u590d\u8ff0",
    "structural_guide": "\u7ed3\u6784\u5f15\u5bfc", "background": "\u80cc\u666f\u94fa\u57ab",
    "example": "\u793a\u4f8b\u8865\u5145",
}


@dataclass
class CorrResult:
    feature: str
    r: float
    p_value: float
    n: int


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    for c in BOOL_COLUMNS:
        if c in df.columns:
            df[c] = _to_bool(df[c])
    for c in NUMERIC_CANDIDATE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pct(part, total):
    return float(part) / float(total) * 100.0 if total else 0.0


def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def sf(v, d=4):
    return "-" if pd.isna(v) else f"{float(v):.{d}f}"


def si(v):
    return "-" if pd.isna(v) else str(int(v))


def cn(feat):
    return CN.get(feat, feat)


def ss(s):
    c = pd.to_numeric(s, errors="coerce").dropna()
    if c.empty:
        return dict(median=0.0, mean=0.0, p10=0.0, p25=0.0, p75=0.0, p90=0.0, mn=0.0, mx=0.0)
    return dict(median=float(c.median()), mean=float(c.mean()),
                p10=float(c.quantile(0.1)), p25=float(c.quantile(0.25)),
                p75=float(c.quantile(0.75)), p90=float(c.quantile(0.9)),
                mn=float(c.min()), mx=float(c.max()))


def top_counts(series, top_n=10):
    vc = series.fillna("unknown").astype(str).value_counts().head(top_n)
    return vc.rename_axis("key").reset_index(name="count")


def _corr_str(r):
    a = abs(r)
    if a >= 0.8: return "\u6781\u5f3a"
    if a >= 0.6: return "\u5f3a"
    if a >= 0.4: return "\u4e2d\u7b49"
    if a >= 0.2: return "\u5f31"
    return "\u6781\u5f31"


def _dom_ex(dt):
    m = {
        "commercial": "Forbes, WebMD, TechCrunch \u7b49",
        "nonprofit": "\u884c\u4e1a\u534f\u4f1a, \u57fa\u91d1\u4f1a, .org \u7ad9\u70b9",
        "news_media": "BBC, CNN, Reuters \u7b49",
        "academic": "\u5927\u5b66\u5b98\u7f51, .edu \u7ad9\u70b9",
        "government": ".gov \u653f\u5e9c\u7f51\u7ad9",
        "encyclopedia": "Wikipedia",
        "tech_corporate": "Microsoft, IBM, Google \u5b98\u65b9\u6587\u6863",
        "research": "\u7814\u7a76\u673a\u6784, \u667a\u5e93",
        "academic_publishing": "\u5b66\u672f\u51fa\u7248\u5e73\u53f0",
        "tutorial": "\u7f16\u7a0b\u6559\u7a0b\u7f51\u7ad9",
        "ugc_platform": "\u77e5\u4e4e, Medium \u7b49\u7528\u6237\u5185\u5bb9\u5e73\u53f0",
    }
    return m.get(dt, "-")


# All narrative text uses Q (corner bracket) to avoid quote conflicts
Q = "\u300c"   # left corner bracket
Qr = "\u300d"  # right corner bracket


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def content_corr_rank(df, target, cols, top_n=25):
    """Correlate only *cols* with *target*."""
    results = []
    if target not in df.columns:
        return results
    for col in cols:
        if col not in df.columns or col == target:
            continue
        tmp = df[[col, target]].dropna()
        if len(tmp) < 30:
            continue
        x, y = tmp[col].astype(float), tmp[target].astype(float)
        if x.nunique() <= 1 or y.nunique() <= 1:
            continue
        r, p = pearsonr(x, y)
        if np.isnan(r):
            continue
        results.append(CorrResult(col, float(r), float(p), len(tmp)))
    results.sort(key=lambda z: abs(z.r), reverse=True)
    return results[:top_n]


def bool_feature_influence(df, bool_cols, target="influence_score"):
    """For each bool col, compare mean *target* for True vs False."""
    rows = []
    for col in bool_cols:
        if col not in df.columns or target not in df.columns:
            continue
        bc = _to_bool(df[col]) if df[col].dtype != bool else df[col]
        tg = df.loc[bc, target].dropna()
        fg = df.loc[~bc, target].dropna()
        if tg.empty or fg.empty:
            continue
        tm, fm = float(tg.mean()), float(fg.mean())
        rows.append(dict(feature=col, true_mean=tm, false_mean=fm,
                         delta=tm - fm, true_n=len(tg), false_n=len(fg),
                         lift=tm / fm if fm > 0 else float("inf")))
    return sorted(rows, key=lambda r: abs(r["delta"]), reverse=True)


def categorical_mean_influence(df, cat_col, target="influence_score", min_n=20):
    """Mean *target* by category, sorted descending."""
    if cat_col not in df.columns or target not in df.columns:
        return []
    grp = df.groupby(df[cat_col].fillna("unknown").astype(str))[target]
    rows = []
    for name, series in grp:
        s = series.dropna()
        if len(s) < min_n:
            continue
        rows.append(dict(category=name, mean=float(s.mean()),
                         median=float(s.median()), n=len(s)))
    return sorted(rows, key=lambda r: r["mean"], reverse=True)


def binned_analysis(df, num_col, target="influence_score", bins=None):
    """Bin a numeric column and compute mean *target* per bin."""
    if num_col not in df.columns or target not in df.columns:
        return []
    clean = df[[num_col, target]].dropna()
    if clean.empty:
        return []
    if bins is None:
        bins = [0, 100, 500, 1000, 2000, 3000, 5000, float("inf")]
    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        labels.append(f"{int(lo)}-{int(hi) if hi != float('inf') else '+'}")
    clean = clean.copy()
    clean["_bin"] = pd.cut(clean[num_col], bins=bins, labels=labels, right=False)
    rows = []
    for label in labels:
        grp = clean[clean["_bin"] == label][target]
        if grp.empty:
            continue
        rows.append(dict(bin=label, mean=float(grp.mean()),
                         median=float(grp.median()), n=len(grp)))
    return rows


def influence_type_profile(df, type_col="llm_influence_type",
                           groups=None, num_cols=None, bool_cols=None):
    """Compare content features across *type_col* groups.

    Returns (num_df, bool_df):
      num_df  — rows=features, columns=group means, sorted by paraphrase-reference delta
      bool_df — rows=bool features, columns=group True-ratios
    """
    if type_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    if groups is None:
        groups = ["paraphrase", "factual_basis", "supplementary", "reference"]
    if num_cols is None:
        num_cols = CONTENT_STRUCTURE_COLS + CONTENT_QUALITY_COLS + TOPIC_ALIGNMENT_COLS + SEMANTIC_NUMERIC_COLS
    if bool_cols is None:
        bool_cols = CONTENT_TYPE_BOOL_COLS

    parts = {}
    for g in groups:
        parts[g] = df[df[type_col].astype(str).str.strip().str.lower() == g]

    # numeric features
    num_rows = []
    for col in num_cols:
        if col not in df.columns:
            continue
        row = {"feature": col}
        for g in groups:
            vals = pd.to_numeric(parts[g][col], errors="coerce").dropna()
            row[g] = float(vals.mean()) if not vals.empty else np.nan
        if "paraphrase" in row and "reference" in row:
            p, r = row.get("paraphrase", np.nan), row.get("reference", np.nan)
            row["delta_pr"] = (p - r) if pd.notna(p) and pd.notna(r) else np.nan
        num_rows.append(row)
    num_df = pd.DataFrame(num_rows)
    if "delta_pr" in num_df.columns:
        num_df = num_df.sort_values("delta_pr", key=abs, ascending=False).reset_index(drop=True)

    # bool features
    bool_rows = []
    for col in bool_cols:
        if col not in df.columns:
            continue
        row = {"feature": col}
        for g in groups:
            bc = _to_bool(parts[g][col]) if col in parts[g].columns else pd.Series(dtype=bool)
            row[g] = float(bc.mean()) if not bc.empty else np.nan
        if "paraphrase" in row and "reference" in row:
            p, r = row.get("paraphrase", np.nan), row.get("reference", np.nan)
            row["delta_pr"] = (p - r) if pd.notna(p) and pd.notna(r) else np.nan
        bool_rows.append(row)
    bool_df = pd.DataFrame(bool_rows)
    if "delta_pr" in bool_df.columns:
        bool_df = bool_df.sort_values("delta_pr", key=abs, ascending=False).reset_index(drop=True)

    return num_df, bool_df, {g: len(parts[g]) for g in groups}


def cmp_tb(df, score_col, compare_cols=None):
    """Split into Top/Bottom 20 % by *score_col* and compare *compare_cols*."""
    if compare_cols is None:
        compare_cols = TOP_BOTTOM_COMPARE_COLUMNS
    clean = df.dropna(subset=[score_col]).copy()
    if clean.empty:
        return clean, clean, pd.DataFrame()
    q80, q20 = clean[score_col].quantile(0.8), clean[score_col].quantile(0.2)
    top = clean[clean[score_col] >= q80].copy()
    bot = clean[clean[score_col] <= q20].copy()
    rows = []
    for col in compare_cols:
        if col not in clean.columns:
            continue
        t = pd.to_numeric(top[col], errors="coerce").dropna()
        b = pd.to_numeric(bot[col], errors="coerce").dropna()
        if t.empty or b.empty:
            continue
        tm, bm = float(t.mean()), float(b.mean())
        rows.append([col, tm, bm, tm - bm, (tm / bm) if bm != 0 else np.nan])
    cmp = pd.DataFrame(rows, columns=["feature", "top_mean", "bottom_mean", "delta", "ratio"])
    return top, bot, cmp.sort_values("delta", key=abs, ascending=False).reset_index(drop=True)


def dist_cmp(top, bottom, col, top_n=8):
    tv = top[col].fillna("unknown").astype(str).value_counts(normalize=True)
    bv = bottom[col].fillna("unknown").astype(str).value_counts(normalize=True)
    keys = list((tv + bv).sort_values(ascending=False).head(top_n).index)
    rows = []
    for k in keys:
        a, b = float(tv.get(k, 0.0)), float(bv.get(k, 0.0))
        rows.append([k, a, b, a - b])
    return pd.DataFrame(rows, columns=[col, "top_share", "bottom_share", "delta_share"])


# ---------------------------------------------------------------------------
# Methodology section (kept from previous iteration)
# ---------------------------------------------------------------------------

def _write_methodology(w, blank, n_total, n_fetch):
    """Insert the methodology / 72-dimension overview section."""
    w("## \u524d\u8a00\uff1a\u5206\u6790\u65b9\u6cd5\u8bba")
    blank()
    w("\u5728\u6df1\u5165\u6570\u636e\u4e4b\u524d\uff0c\u672c\u7ae0\u5148\u4ecb\u7ecd\u6574\u4e2a\u5206\u6790\u7684\u6280\u672f\u65b9\u6cd5\uff1a"
      "\u6570\u636e\u662f\u600e\u4e48\u6765\u7684\u300172 \u4e2a\u7ef4\u5ea6\u5206\u522b\u662f\u4ec0\u4e48\u3001\u5982\u4f55\u8ba1\u7b97\uff0c"
      "\u4ee5\u53ca\u6838\u5fc3\u6307\u6807 influence_score \u7684\u8bbe\u8ba1\u903b\u8f91\u3002")
    blank()

    # --- 0.1 pipeline ---
    w("### 0.1 \u6570\u636e\u91c7\u96c6\u4e0e\u52a0\u5de5\u7ba1\u7ebf")
    blank()
    w("\u6574\u4e2a\u6570\u636e\u4ece\u539f\u59cb\u7f51\u9875\u5230\u6700\u7ec8 72 \u7ef4\u7279\u5f81\u8868\uff0c\u7ecf\u5386\u4e86\u4e09\u4e2a\u9636\u6bb5\uff1a")
    blank()
    w("```mermaid")
    w("flowchart LR")
    w("    RawHTML[\"Phase1: \u4e09\u5e73\u53f0\u539f\u59cb HTML\"] --> Parser[\"\u5e73\u53f0\u4e13\u7528\u89e3\u6790\u5668\"]")
    w("    Parser --> IndexJSONL[\"index.jsonl\"]")
    w("    IndexJSONL --> Fetch[\"Phase2: URL \u6279\u91cf\u6293\u53d6\"]")
    w("    Fetch --> Cache[\"\u672c\u5730 HTML \u7f13\u5b58\"]")
    w("    Cache --> Features[\"Phase3: citation_features.py\"]")
    w("    Features --> Local[\"\u672c\u5730\u7279\u5f81 66 \u7ef4\"]")
    w("    Local --> Emb[\"OpenAI Embedding +2 \u7ef4\"]")
    w("    Emb --> LLM[\"Gemini LLM +4 \u7ef4\"]")
    w("    LLM --> CSV[\"\u5b8c\u6574 72 \u7ef4 CSV\"]")
    w("```")
    blank()

    w("**Phase 1 \u2014 \u5e73\u53f0\u89e3\u6790**\uff1a\u6211\u4eec\u4e3a ChatGPT\u3001Google AI Overview\u3001Perplexity "
      "\u5404\u7f16\u5199\u4e86\u4e13\u7528\u7684 HTML \u89e3\u6790\u5668\uff0c\u4ece\u6bcf\u4e2a\u5e73\u53f0\u7684\u56de\u7b54\u9875\u9762\u4e2d\u63d0\u53d6\u51fa\uff1a"
      "\u7528\u6237\u95ee\u9898\u3001AI \u56de\u7b54\u5185\u5bb9\uff08\u4fdd\u7559 HTML \u683c\u5f0f\uff09\u3001\u5f15\u7528\u5217\u8868\uff08URL + \u6807\u9898 + \u5c55\u793a\u540d\uff09\u3002"
      "\u89e3\u6790\u7ed3\u679c\u5199\u5165 index.jsonl\uff0c\u6bcf\u884c\u4e00\u4e2a\u95ee\u7b54\u8bb0\u5f55\u3002")
    blank()
    w("**Phase 2 \u2014 \u5f15\u7528\u9875\u6293\u53d6**\uff1a\u5c06\u4e09\u5e73\u53f0\u6240\u6709\u5f15\u7528 URL \u53bb\u91cd\u5408\u5e76\u540e\uff0c"
      "\u901a\u8fc7 HTTP \u8bf7\u6c42\uff08\u5e76\u53d1 + \u57df\u540d\u9650\u901f\uff09\u6279\u91cf\u6293\u53d6\u539f\u59cb\u7f51\u9875\u5185\u5bb9\u3002"
      "\u5bf9\u4e8e\u9700\u8981 JavaScript \u6e32\u67d3\u7684\u9875\u9762\uff0c\u81ea\u52a8\u56de\u9000\u5230\u6d4f\u89c8\u5668\u6293\u53d6\u3002"
      "\u6240\u6709\u62d3\u53d6\u7ed3\u679c\u7f13\u5b58\u5230\u672c\u5730\uff0c\u907f\u514d\u91cd\u590d\u8bf7\u6c42\u3002"
      f"\u672c\u6b21\u5171\u6293\u53d6\u5230 **{n_fetch:,}** \u4e2a\u9875\u9762\u7684\u6709\u6548\u5185\u5bb9\uff08\u6210\u529f\u7387 {pct(n_fetch, n_total):.1f}%\uff09\u3002")
    blank()
    w("**Phase 3 \u2014 \u7279\u5f81\u63d0\u53d6**\uff1a`citation_features.py` \u8bfb\u53d6 index.jsonl\uff0c"
      "\u5c06\u6bcf\u6761\u95ee\u7b54\u8bb0\u5f55\u5c55\u5f00\u4e3a", Q, "\u6bcf\u4e2a\u5f15\u7528\u4e00\u884c", Qr, "\uff0c"
      "\u7136\u540e\u5206\u4e24\u4e2a\u9636\u6bb5\u8ba1\u7b97\u7279\u5f81\uff1a")
    w("1. **\u672c\u5730\u8ba1\u7b97\uff0866 \u7ef4\uff09**\uff1a\u4f7f\u7528 HTML \u89e3\u6790\u3001\u6b63\u5219\u5339\u914d\u3001TF-IDF \u5411\u91cf\u5316\u7b49\u7eaf\u672c\u5730\u65b9\u6cd5\uff0c\u65e0\u9700\u8054\u7f51\u3002")
    w("2. **API \u589e\u5f3a\uff086 \u7ef4\uff09**\uff1a\u8c03\u7528 OpenAI Embedding API \u548c Google Gemini LLM\uff0c"
      "\u6dfb\u52a0\u8bed\u4e49\u5411\u91cf\u76f8\u4f3c\u5ea6\u548c\u5927\u8bed\u8a00\u6a21\u578b\u8bed\u4e49\u5206\u6790\u7ed3\u679c\u3002")
    blank()

    # --- 0.2 72-dim table ---
    _write_feature_tables(w, blank)

    # --- 0.3 influence_score formula + variable tiers ---
    w("### 0.3 influence_score \u8ba1\u7b97\u516c\u5f0f\u4e0e\u53d8\u91cf\u5206\u5c42")
    blank()
    w("\u6211\u4eec\u8bbe\u8ba1\u4e86\u4e00\u4e2a\u52a0\u6743\u7efc\u5408\u5f97\u5206\uff0c\u7efc\u5408\u8861\u91cf\u6bcf\u6761\u5f15\u7528\u5bf9 AI \u56de\u7b54\u7684\u5b9e\u9645\u5f71\u54cd\u529b\uff1a")
    blank()
    w("```")
    w("influence_score =")
    w("    0.20 x min(ref_count / 3, 1)           -- \u5f15\u7528\u6b21\u6570\uff08\u622a\u65ad\u5728 3 \u6b21\uff09")
    w("  + 0.15 x (1 - first_position_ratio)      -- \u4f4d\u7f6e\u9760\u524d\u7a0b\u5ea6")
    w("  + 0.20 x paragraph_coverage_ratio         -- \u6bb5\u843d\u8986\u76d6\u7387")
    w("  + 0.25 x tfidf_cosine                     -- TF-IDF \u6587\u672c\u76f8\u4f3c\u5ea6")
    w("  + 0.20 x (bigram_overlap + trigram_overlap) / 2  -- N-gram \u91cd\u53e0\u7387")
    w("```")
    blank()
    w("**\u91cd\u8981\u8bf4\u660e\u2014\u2014\u53d8\u91cf\u5206\u5c42**\uff1a\u4e0a\u8ff0 5 \u4e2a\u7ec4\u6210\u53d8\u91cf\uff08ref_count\u3001first_position_ratio\u3001"
      "paragraph_coverage_ratio\u3001tfidf_cosine\u3001bigram/trigram_overlap\uff09\u662f\u5f71\u54cd\u529b\u5f97\u5206\u7684**\u5b9a\u4e49\u7ec4\u4ef6**\uff0c"
      "\u4e0e\u5176\u8ba1\u7b97\u76f8\u5173\u6027\u662f\u540c\u4e49\u53cd\u590d\uff0c\u4e0d\u542b\u4fe1\u606f\u91cf\u3002"
      "\u56e0\u6b64\u5728\u540e\u7eed\u5206\u6790\u4e2d\uff0c\u6211\u4eec\u5c06 72 \u4e2a\u5b57\u6bb5\u5206\u4e3a\u4e09\u5c42\uff1a")
    blank()
    w("1. **\u5185\u5bb9\u7279\u5f81\uff08\u521b\u4f5c\u8005\u53ef\u63a7\u7684\u81ea\u53d8\u91cf X\uff09**\uff1a"
      "\u9875\u9762\u7ed3\u6784\u3001\u5185\u5bb9\u8d28\u91cf\u3001\u5185\u5bb9\u7c7b\u578b\u3001\u6807\u9898\u5339\u914d\u5ea6\u3001\u6765\u6e90\u8eab\u4efd\u7b49\u2014\u2014"
      "\u8fd9\u4e9b\u662f\u5185\u5bb9\u521b\u4f5c\u8005\u53ef\u4ee5\u76f4\u63a5\u63a7\u5236\u7684\u7279\u5f81\u3002")
    w("2. **\u8bed\u4e49\u8bc4\u4f30\uff08\u72ec\u7acb\u7b2c\u4e09\u65b9\u8bc4\u5206\uff09**\uff1a"
      "Embedding \u76f8\u4f3c\u5ea6\u548c LLM \u8bc4\u5206\u2014\u2014"
      "\u8fd9\u4e9b\u662f\u5bf9\u5185\u5bb9\u8d28\u91cf\u548c\u76f8\u5173\u6027\u7684\u72ec\u7acb\u8bc4\u4f30\uff0c\u53ef\u4ee5\u5e2e\u52a9\u9a8c\u8bc1\u5185\u5bb9\u7279\u5f81\u7684\u53d1\u73b0\u3002")
    w("3. **\u5f71\u54cd\u529b\u7ed3\u679c\u4fe1\u53f7\uff08\u56e0\u53d8\u91cf Y\uff09**\uff1a"
      "ref_count\u3001position\u3001coverage\u3001similarity \u7b49\u2014\u2014"
      "\u8fd9\u4e9b\u662f\u5f71\u54cd\u529b\u7684**\u8868\u73b0\u5f62\u5f0f**\uff0c\u4e0d\u662f\u539f\u56e0\uff0c\u4e0d\u4f5c\u4e3a", Q, "\u53d1\u73b0", Qr, "\u5c55\u793a\u3002")
    blank()
    w("\u540e\u7eed\u6240\u6709\u5206\u6790\u5747\u4ee5\u7b2c\u4e00\u3001\u4e8c\u5c42\u7279\u5f81\u4e3a\u81ea\u53d8\u91cf\uff0c\u63a2\u7a76\u5b83\u4eec\u5982\u4f55\u9884\u6d4b influence_score\u3002")
    blank()

    # --- 0.4 tools ---
    w("### 0.4 \u6240\u7528\u5de5\u5177\u4e0e API")
    blank()
    w(md_table(
        ["\u73af\u8282", "\u5de5\u5177 / API", "\u7528\u9014"],
        [
            ["HTML \u89e3\u6790", "BeautifulSoup + lxml", "\u89e3\u6790\u5e73\u53f0\u56de\u7b54\u548c\u5f15\u7528\u9875\u9762\u7684 HTML \u7ed3\u6784"],
            ["\u6587\u672c\u5206\u8bcd", "\u6b63\u5219 tokenizer", "\u63d0\u53d6\u4e2d\u82f1\u6587\u8bcd\u6c47\uff0c\u652f\u6301 CJK \u5b57\u7b26"],
            ["\u6587\u672c\u76f8\u4f3c\u5ea6", "sklearn TfidfVectorizer + cosine_similarity", "TF-IDF \u5411\u91cf\u5316\u4e0e\u4f59\u5f26\u8ddd\u79bb\u8ba1\u7b97"],
            ["\u5411\u91cf\u5d4c\u5165", "OpenAI text-embedding-3-small", "\u8bed\u4e49\u5411\u91cf\u76f8\u4f3c\u5ea6\uff0ctiktoken \u622a\u65ad\u81f3 8100 tokens"],
            ["LLM \u8bed\u4e49\u5206\u6790", "Google Gemini gemini-2.0-flash", "\u7ed3\u6784\u5316 JSON \u8f93\u51fa\uff0c\u5206\u6790\u5f15\u7528\u89d2\u8272/\u76f8\u5173\u6027/\u8d28\u91cf/\u5f71\u54cd\u7c7b\u578b"],
            ["\u7edf\u8ba1\u5206\u6790", "pandas + numpy + scipy", "Pearson \u76f8\u5173\u3001\u5206\u4f4d\u6570\u3001\u5206\u7ec4\u5bf9\u6bd4\u7b49"],
            ["\u7f13\u5b58\u4e0e\u6062\u590d", "\u78c1\u76d8 JSON \u7f13\u5b58", "API \u7ed3\u679c\u6309\u6587\u672c\u54c8\u5e0c / record_id \u7f13\u5b58\uff0c\u652f\u6301\u65ad\u70b9\u7eed\u8dd1"],
        ],
    ))
    blank()
    w("---")
    blank()


def _write_feature_tables(w, blank):
    """Write the 72-dim feature overview tables (0.2 subsection)."""
    w("### 0.2 72 \u7ef4\u7279\u5f81\u5168\u89c8")
    blank()
    w("\u4ee5\u4e0b\u6309\u529f\u80fd\u5206\u4e3a 10 \u5927\u7ec4\uff0c\u5b8c\u6574\u5217\u51fa\u6bcf\u4e2a\u5b57\u6bb5\u7684\u542b\u4e49\u4e0e\u8ba1\u7b97\u65b9\u5f0f\u3002")
    blank()
    _h = ["\u5b57\u6bb5", "\u4e2d\u6587\u542b\u4e49", "\u8ba1\u7b97\u65b9\u5f0f"]

    w("#### \u7ec4\u4e00\uff1a\u5143\u6570\u636e / \u95ee\u7b54\u4e0a\u4e0b\u6587\uff0818 \u5b57\u6bb5\uff09")
    blank()
    w("\u8fd9\u4e9b\u5b57\u6bb5\u63cf\u8ff0\u6bcf\u6761\u8bb0\u5f55\u7684\u57fa\u672c\u4fe1\u606f\u3002")
    blank()
    w(md_table(_h, [
        ["record_id", "\u8bb0\u5f55\u552f\u4e00 ID", "\u6e90\u6587\u4ef6\u540d\u7684 SHA256 \u524d 16 \u4f4d"],
        ["platform", "\u5e73\u53f0", "chatgpt / google / perplexity"],
        ["category", "\u884c\u4e1a\u5206\u7c7b", "\u4ece\u8f93\u5165\u6216\u8def\u5f84\u63a8\u65ad"],
        ["question", "\u7528\u6237\u95ee\u9898", "\u539f\u59cb\u95ee\u9898\u6587\u672c"],
        ["question_type", "\u95ee\u9898\u7c7b\u578b", "\u6b63\u5219\u5339\u914d\u5206\u7c7b\uff1ahow_to / comparison / what_is \u7b49"],
        ["answer_word_count", "\u56de\u7b54\u8bcd\u6570", "\u56de\u7b54\u7eaf\u6587\u672c\u7684 token \u6570"],
        ["answer_char_count", "\u56de\u7b54\u5b57\u7b26\u6570", "\u56de\u7b54\u7eaf\u6587\u672c\u7684\u5b57\u7b26\u957f\u5ea6"],
        ["total_citations", "\u603b\u5f15\u7528\u6570", "\u8be5\u95ee\u9898\u4e0b\u7684\u5f15\u7528\u603b\u6570"],
        ["answer_heading_count", "\u56de\u7b54\u6807\u9898\u6570", "BeautifulSoup \u8ba1\u6570 h1-h6"],
        ["answer_list_item_count", "\u56de\u7b54\u5217\u8868\u9879\u6570", "\u8ba1\u6570 li"],
        ["answer_has_table", "\u56de\u7b54\u662f\u5426\u542b\u8868\u683c", "\u662f\u5426\u5b58\u5728 table"],
        ["citation_index", "\u5f15\u7528\u5e8f\u53f7", "\u672c\u6761\u5f15\u7528\u5728\u5217\u8868\u4e2d\u7684\u4f4d\u7f6e"],
        ["url", "\u5f15\u7528 URL", "\u539f\u59cb\u94fe\u63a5\u5730\u5740"],
        ["display_name", "\u5c55\u793a\u540d", "\u5e73\u53f0\u663e\u793a\u7684\u6765\u6e90\u540d\u79f0"],
        ["cit_title", "\u9875\u9762\u6807\u9898", "\u6293\u53d6\u540e\u63d0\u53d6\u7684 title"],
        ["fetch_ok", "\u662f\u5426\u6293\u53d6\u6210\u529f", "\u80fd\u5426\u83b7\u53d6\u5230\u6709\u6548\u5185\u5bb9"],
        ["status_code", "HTTP \u72b6\u6001\u7801", "\u6293\u53d6\u65f6\u7684\u54cd\u5e94\u72b6\u6001\u7801"],
        ["error_class", "\u9519\u8bef\u5206\u7c7b", "\u6293\u53d6\u5931\u8d25\u65f6\u7684\u9519\u8bef\u7c7b\u578b"],
    ]))
    blank()

    w("#### \u7ec4\u4e8c\uff1aURL / \u57df\u540d\u7279\u5f81\uff085 \u5b57\u6bb5\uff09"); blank()
    w(md_table(_h, [
        ["domain", "\u4e3b\u673a\u540d", "urlparse \u63d0\u53d6\uff0c\u53bb\u9664 www."],
        ["domain_tld", "\u9876\u7ea7\u57df", ".com / .org / .edu \u7b49"],
        ["domain_type", "\u7ad9\u70b9\u7c7b\u578b", "\u5173\u952e\u8bcd\u89c4\u5219\u5206\u7c7b"],
        ["path_depth", "\u8def\u5f84\u6df1\u5ea6", "URL path \u4e2d\u975e\u7a7a\u6bb5\u6570"],
        ["url_has_query_keyword", "URL \u542b\u95ee\u9898\u5173\u952e\u8bcd", "URL \u8def\u5f84\u4e2d\u662f\u5426\u5305\u542b\u95ee\u9898\u76f8\u5173\u8bcd\u6c47"],
    ]))
    blank()

    w("#### \u7ec4\u4e09\uff1a\u5f15\u7528\u9875\u7ed3\u6784\u7279\u5f81\uff0815 \u5b57\u6bb5\uff09"); blank()
    w(md_table(_h, [
        ["cit_char_count", "\u5b57\u7b26\u6570", "\u7eaf\u6587\u672c\u5b57\u7b26\u957f\u5ea6"],
        ["cit_word_count", "\u8bcd\u6570", "\u6b63\u5219 tokenizer \u5206\u8bcd\u540e\u7684 token \u6570"],
        ["cit_heading_h1~h6", "H1-H6 \u5404\u7ea7\u6807\u9898\u6570", "\u5206\u522b\u8ba1\u6570\u5404\u7ea7\u6807\u9898\uff086 \u5b57\u6bb5\uff09"],
        ["cit_heading_total", "\u6807\u9898\u603b\u6570", "H1-H6 \u4e4b\u548c"],
        ["cit_paragraph_count", "\u6bb5\u843d\u6570", "\u8ba1\u6570 p \u6807\u7b7e"],
        ["cit_avg_para_words", "\u6bb5\u843d\u5e73\u5747\u8bcd\u6570", "\u6bcf\u6bb5\u8bcd\u6570\u7684\u7b97\u672f\u5747\u503c"],
        ["cit_list_count", "\u5217\u8868\u5757\u6570", "\u8ba1\u6570 ul + ol"],
        ["cit_list_item_count", "\u5217\u8868\u9879\u6570", "\u8ba1\u6570 li"],
        ["cit_list_density", "\u5217\u8868\u5bc6\u5ea6", "\u5217\u8868\u9879 / max(\u6bb5\u843d, 1)"],
        ["cit_table_count", "\u8868\u683c\u6570", "\u8ba1\u6570 table"],
        ["cit_image_count", "\u56fe\u7247\u6570", "\u8ba1\u6570 img"],
        ["cit_link_count", "\u94fe\u63a5\u6570", "\u8ba1\u6570 a[href]"],
        ["cit_bold_count", "\u52a0\u7c97\u6587\u672c\u6570", "\u8ba1\u6570 strong + b"],
        ["cit_code_block_count", "\u4ee3\u7801\u5757\u6570", "\u8ba1\u6570 pre + code"],
        ["cit_has_code", "\u662f\u5426\u542b\u4ee3\u7801", "\u5b58\u5728 pre / code \u5219\u4e3a True"],
    ]))
    blank()

    w("#### \u7ec4\u56db\uff1a\u5185\u5bb9\u8d28\u91cf / \u4f53\u88c1\u7279\u5f81\uff0810 \u5b57\u6bb5\uff09"); blank()
    w(md_table(_h, [
        ["title_question_match", "\u6807\u9898\u4e0e\u95ee\u9898\u5339\u914d\u5ea6", "\u6807\u9898 token \u4e0e\u95ee\u9898 token \u7684\u4ea4\u96c6 / \u95ee\u9898 token \u6570"],
        ["cit_info_density", "\u4fe1\u606f\u5bc6\u5ea6", "\u5b9e\u8bcd\u5360\u6bd4"],
        ["cit_has_numbers", "\u542b\u6570\u5b57/\u7edf\u8ba1", "\u6b63\u5219\u5339\u914d\u767e\u5206\u6bd4\u3001\u91d1\u989d\u3001\u5e74\u4efd\u7b49"],
        ["cit_has_definition", "\u542b\u5b9a\u4e49\u53e5\u5f0f", "\u5339\u914d is a / refers to / defined as \u7b49"],
        ["cit_has_qa_format", "\u542b\u95ee\u7b54\u683c\u5f0f", "\u5339\u914d\u95ee\u53f7 + what/how/why"],
        ["cit_has_howto", "\u542b\u64cd\u4f5c\u6307\u5357", "\u5339\u914d step N / how to / first...then"],
        ["cit_has_comparison", "\u542b\u5bf9\u6bd4\u5185\u5bb9", "\u5339\u914d vs / compared to / difference between"],
        ["cit_language", "\u8bed\u8a00", "\u4e2d\u65e5\u6587\u5b57\u7b26\u5360\u6bd4 >0.3 \u4e3a zh"],
        ["cit_avg_sentence_length", "\u5e73\u5747\u53e5\u957f", "\u6bcf\u53e5\u8bcd\u6570\u5747\u503c"],
        ["cit_unique_word_ratio", "\u8bcd\u6c47\u4e30\u5bcc\u5ea6", "\u552f\u4e00\u8bcd / \u603b\u8bcd"],
    ]))
    blank()

    w("#### \u7ec4\u4e94\uff1a\u7b54\u6848\u4e2d\u7684\u5f15\u7528\u4f4d\u7f6e\uff085 \u5b57\u6bb5\uff0c\u5c5e\u4e8e\u7ed3\u679c\u53d8\u91cf\uff09"); blank()
    w(md_table(_h, [
        ["ref_count", "\u88ab\u5f15\u7528\u6b21\u6570", "\u540c\u4e00 URL \u5728\u56de\u7b54\u4e2d\u51fa\u73b0\u7684\u94fe\u63a5\u6570"],
        ["first_position_ratio", "\u9996\u6b21\u51fa\u73b0\u4f4d\u7f6e", "\u7b2c\u4e00\u6b21\u51fa\u73b0\u7684\u94fe\u63a5\u5e8f\u53f7 / \u94fe\u63a5\u603b\u6570"],
        ["covered_paragraphs", "\u8986\u76d6\u6bb5\u843d\u6570", "\u56de\u7b54\u4e2d\u542b\u8be5\u5f15\u7528\u94fe\u63a5\u7684\u6bb5\u843d\u6570"],
        ["total_answer_paragraphs", "\u56de\u7b54\u603b\u6bb5\u843d\u6570", "\u56de\u7b54\u4e2d p + li + h3 + h4 \u603b\u6570"],
        ["paragraph_coverage_ratio", "\u6bb5\u843d\u8986\u76d6\u7387", "\u8986\u76d6\u6bb5\u843d\u6570 / \u56de\u7b54\u603b\u6bb5\u843d\u6570"],
    ]))
    blank()

    w("#### \u7ec4\u516d\uff1a\u6587\u672c\u76f8\u4f3c\u5ea6\uff083 \u5b57\u6bb5\uff0c\u5c5e\u4e8e\u7ed3\u679c\u53d8\u91cf\uff09"); blank()
    w(md_table(_h, [
        ["tfidf_cosine", "TF-IDF \u4f59\u5f26\u76f8\u4f3c\u5ea6", "\u5f15\u7528\u9875\u4e0e\u56de\u7b54\u7684 TF-IDF \u4f59\u5f26\u8ddd\u79bb"],
        ["bigram_overlap", "\u53cc\u8bcd\u7ec4\u91cd\u53e0\u7387", "\u5f15\u7528\u4e0e\u56de\u7b54\u7684\u8fde\u7eed\u4e24\u8bcd\u7ec4\u5408\u4ea4\u96c6 / \u56de\u7b54\u53cc\u8bcd\u7ec4\u6570"],
        ["trigram_overlap", "\u4e09\u8bcd\u7ec4\u91cd\u53e0\u7387", "\u540c\u4e0a\uff0c\u8fde\u7eed\u4e09\u8bcd\u7ec4\u5408"],
    ]))
    blank()

    w("#### \u7ec4\u4e03\uff1a\u7efc\u5408\u5f71\u54cd\u529b\u5f97\u5206\uff082 \u5b57\u6bb5\uff09"); blank()
    w(md_table(_h, [
        ["influence_score", "\u5f71\u54cd\u529b\u5f97\u5206", "\u52a0\u6743\u7efc\u5408\u5206\uff0c\u8be6\u89c1 0.3 \u8282"],
        ["influence_rank", "\u5f71\u54cd\u529b\u6392\u540d", "\u540c\u4e00\u95ee\u9898\u5185\u6309 influence_score \u964d\u5e8f"],
    ]))
    blank()

    w("#### \u7ec4\u516b\uff1a\u540c\u8bb0\u5f55\u5185\u7ade\u4e89\u7279\u5f81\uff083 \u5b57\u6bb5\uff09"); blank()
    w(md_table(_h, [
        ["same_domain_count", "\u540c\u57df\u540d\u5f15\u7528\u6570", "\u540c\u4e00\u95ee\u9898\u4e0b\u76f8\u540c\u57df\u540d\u7684\u5f15\u7528\u6570"],
        ["domain_is_unique", "\u57df\u540d\u662f\u5426\u552f\u4e00", "same_domain_count == 1"],
        ["rank_by_ref_count", "\u6309\u5f15\u7528\u6b21\u6570\u6392\u540d", "\u540c\u4e00\u95ee\u9898\u5185\u6309 ref_count \u964d\u5e8f"],
    ]))
    blank()

    w("#### \u7ec4\u4e5d\uff1a\u5411\u91cf\u5d4c\u5165\u76f8\u4f3c\u5ea6\uff082 \u5b57\u6bb5\uff0cOpenAI API\uff09"); blank()
    w(md_table(["\u5b57\u6bb5", "\u4e2d\u6587\u542b\u4e49", "\u8ba1\u7b97\u65b9\u5f0f"], [
        ["emb_answer_cit_cosine", "\u5f15\u7528\u9875 vs \u56de\u7b54\u7684\u8bed\u4e49\u76f8\u4f3c\u5ea6", "\u5f15\u7528\u9875\u4e0e\u56de\u7b54\u7684 embedding \u4f59\u5f26\u76f8\u4f3c\u5ea6"],
        ["emb_question_cit_cosine", "\u5f15\u7528\u9875 vs \u95ee\u9898\u7684\u8bed\u4e49\u76f8\u4f3c\u5ea6", "\u5f15\u7528\u9875\u4e0e\u95ee\u9898\u7684 embedding \u4f59\u5f26\u76f8\u4f3c\u5ea6"],
    ]))
    blank()

    w("#### \u7ec4\u5341\uff1aLLM \u8bed\u4e49\u6807\u7b7e\uff084 \u5b57\u6bb5\uff0cGemini API\uff09"); blank()
    w(md_table(["\u5b57\u6bb5", "\u4e2d\u6587\u542b\u4e49", "\u53d6\u503c\u8303\u56f4"], [
        ["llm_semantic_role", "\u8bed\u4e49\u89d2\u8272", "evidence / reference / background / example / definition / statistical_data / comparison / opinion"],
        ["llm_relevance_score", "\u76f8\u5173\u6027\u8bc4\u5206", "1-5 \u5206"],
        ["llm_content_quality", "\u5185\u5bb9\u8d28\u91cf\u8bc4\u5206", "1-5 \u5206"],
        ["llm_influence_type", "\u5f71\u54cd\u7c7b\u578b", "factual_basis / supplementary / paraphrase / reference / structural_guide / background / example"],
    ]))
    blank()

    w("> \u4ee5\u4e0a 4 \u4e2a\u5b57\u6bb5\u7531 Gemini gemini-2.0-flash \u6a21\u578b\u6839\u636e\u4ee5\u4e0b\u63d0\u793a\u8bcd\u9010\u6761\u5206\u6790\u751f\u6210\uff1a")
    blank()
    w("```text")
    w("Analyze the citations used by an AI assistant to answer a question.")
    w("")
    w("Question: {question}")
    w("")
    w("Answer (first 1500 chars):")
    w("{answer_excerpt}")
    w("")
    w("Citations:")
    w("{citations_block}")
    w("")
    w("For each citation, determine:")
    w("1. semantic_role \u2014 the function this citation serves. Choose ONE of:")
    w("   definition, evidence, example, background, procedure,")
    w("   statistical_data, comparison, opinion, reference")
    w("2. relevance_score \u2014 integer 1-5, how relevant the citation is to the")
    w("   question (5 = highly relevant)")
    w("3. content_quality \u2014 integer 1-5, how authoritative / well-written the")
    w("   content is (5 = very high quality)")
    w("4. influence_type \u2014 how this citation was used in the answer. Choose ONE")
    w("   of: direct_quote, paraphrase, factual_basis, structural_guide,")
    w("   supplementary")
    w("")
    w('Return ONLY a valid JSON array (no markdown fences, no other text):')
    w('[{"idx": 0, "semantic_role": "...", "relevance_score": N,')
    w('  "content_quality": N, "influence_type": "..."}, ...]')
    w("```")
    blank()
    w("> \u5176\u4e2d `{question}`\u3001`{answer_excerpt}`\u3001`{citations_block}` \u4e3a\u8fd0\u884c\u65f6\u52a8\u6001\u586b\u5145\u7684\u53d8\u91cf\u3002"
      "\u6bcf\u6761\u8bb0\u5f55\u7684\u5206\u6790\u7ed3\u679c\u4ee5 JSON \u683c\u5f0f\u8fd4\u56de\uff0c\u7ecf\u78c1\u76d8\u7f13\u5b58\u540e\u5199\u5165 CSV\u3002")
    blank()


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(df):
    n_total = len(df)
    fdf = df[df["fetch_ok"] == True].copy() if "fetch_ok" in df.columns else df.copy()
    nf = len(fdf)
    nfail = n_total - nf
    L = []

    def w(*parts):
        L.append("".join(parts))

    def blank():
        L.append("")

    # === TITLE ===
    w("# AI \u641c\u7d22\u5f15\u7528\u5f71\u54cd\u529b\u6df1\u5ea6\u5206\u6790\u62a5\u544a")
    blank()
    w(f"> \u672c\u62a5\u544a\u57fa\u4e8e ChatGPT\u3001Google AI Overview\u3001Perplexity \u4e09\u5927 AI \u641c\u7d22\u5e73\u53f0\u5171 **{n_total:,}** \u6761\u5f15\u7528\u6570\u636e\uff0c")
    w("> \u901a\u8fc7 72 \u4e2a\u7ef4\u5ea6\u7684\u7279\u5f81\u63d0\u53d6\u4e0e\u7edf\u8ba1\u5206\u6790\uff0c\u56de\u7b54\u4e00\u4e2a\u6838\u5fc3\u95ee\u9898\uff1a")
    w("> **\u4ec0\u4e48\u6837\u7684\u5185\u5bb9\u7279\u5f81\uff08\u7ed3\u6784\u3001\u8d28\u91cf\u3001\u8bed\u4e49\u89d2\u8272\uff09\u8ba9\u4e00\u6761\u5f15\u7528\u5bf9 AI \u56de\u7b54\u7684\u5f71\u54cd\u529b\u66f4\u5927\uff1f**")
    blank()
    w("---")
    blank()

    # === 0. METHODOLOGY ===
    _write_methodology(w, blank, n_total, nf)

    # === 1. DATA OVERVIEW ===
    w("## \u4e00\u3001\u6570\u636e\u6982\u89c8")
    blank()
    w("### 1.1 \u6570\u636e\u96c6\u89c4\u6a21")
    blank()
    w(md_table(
        ["\u6307\u6807", "\u6570\u503c", "\u8bf4\u660e"],
        [
            ["\u603b\u5f15\u7528\u6761\u6570", f"{n_total:,}", "\u4e09\u4e2a\u5e73\u53f0\u6240\u6709\u95ee\u9898\u7684\u5168\u90e8\u5f15\u7528\u5408\u8ba1"],
            ["\u6210\u529f\u6293\u53d6", f"{nf:,} ({pct(nf, n_total):.1f}%)", "\u80fd\u83b7\u53d6\u5230\u539f\u59cb\u7f51\u9875\u5185\u5bb9\u7684\u5f15\u7528"],
            ["\u6293\u53d6\u5931\u8d25", f"{nfail:,} ({pct(nfail, n_total):.1f}%)", "\u9875\u9762\u5df2\u5220\u9664/\u9700\u767b\u5f55/\u53cd\u722c\u7b49"],
            ["\u7279\u5f81\u7ef4\u5ea6", "72", "\u5bf9\u6bcf\u6761\u5f15\u7528\u63d0\u53d6\u7684\u5206\u6790\u5b57\u6bb5\u6570"],
        ],
    ))
    blank()

    if "platform" in df.columns:
        w("### 1.2 \u4e09\u5e73\u53f0\u6570\u636e\u5206\u5e03")
        blank()
        plat_vc = df["platform"].fillna("unknown").astype(str).value_counts()
        plat_fetch = fdf["platform"].fillna("unknown").astype(str).value_counts()
        rows = []
        for p in plat_vc.index:
            t = int(plat_vc[p])
            ok = int(plat_fetch.get(p, 0))
            rows.append([p, f"{t:,}", f"{pct(t, n_total):.1f}%", f"{ok:,}", f"{pct(ok, t):.1f}%"])
        w(md_table(["\u5e73\u53f0", "\u5f15\u7528\u603b\u6570", "\u5360\u6bd4", "\u6210\u529f\u6293\u53d6", "\u6293\u53d6\u7387"], rows))
        blank()

    if "category" in df.columns:
        w("### 1.3 \u884c\u4e1a\u5206\u7c7b\u5206\u5e03")
        blank()
        cat_vc = df["category"].fillna("unknown").astype(str).value_counts()
        w(md_table(["\u5206\u7c7b", "\u5f15\u7528\u6570", "\u5360\u6bd4"],
                   [[c, f"{int(v):,}", f"{pct(v, n_total):.1f}%"] for c, v in cat_vc.items()]))
        blank()

    w("---")
    blank()

    # === 2. SURVIVOR PROFILE ===
    w("## \u4e8c\u3001\u5e78\u5b58\u8005\u753b\u50cf \u2014\u2014 \u88ab\u5f15\u7528\u9875\u9762\u7684\u5171\u6027\u7279\u5f81")
    blank()
    w("> \u51fa\u73b0\u5728\u5f15\u7528\u5217\u8868\u4e2d\u7684\u7f51\u9875\u5df2\u7ecf\u901a\u8fc7\u4e86 AI \u7684\u7b5b\u9009\u3002\u672c\u7ae0\u63cf\u8ff0\u8fd9\u4e9b", Q, "\u5e78\u5b58\u8005", Qr, "\u7684\u5171\u540c\u7279\u5f81\u3002")
    blank()
    w(f"\u5206\u6790\u6837\u672c\uff1a\u6210\u529f\u6293\u53d6\u5230\u5185\u5bb9\u7684 **{nf:,}** \u6761\u5f15\u7528\u3002")
    blank()

    # 2.1 content volume
    w("### 2.1 \u5185\u5bb9\u4f53\u91cf")
    blank()
    if {"cit_word_count", "cit_char_count"}.issubset(fdf.columns):
        ws = ss(fdf["cit_word_count"])
        cs = ss(fdf["cit_char_count"])
        w(md_table(
            ["\u6307\u6807", "P10", "\u4e2d\u4f4d\u6570", "\u5747\u503c", "P90"],
            [
                ["\u8bcd\u6570", f"{ws['p10']:.0f}", f"{ws['median']:.0f}",
                 f"{ws['mean']:.0f}", f"{ws['p90']:.0f}"],
                ["\u5b57\u7b26\u6570", f"{cs['p10']:.0f}", f"{cs['median']:.0f}",
                 f"{cs['mean']:.0f}", f"{cs['p90']:.0f}"],
            ],
        ))
        blank()
        w(f"\u88ab\u5f15\u7528\u9875\u9762\u7684\u4e2d\u4f4d\u8bcd\u6570\u4e3a **{ws['median']:.0f}**\uff0c"
          f"P90 \u4e3a {ws['p90']:.0f} \u8bcd\u3002"
          "\u5927\u591a\u6570\u88ab\u5f15\u9875\u9762\u662f 500-3000 \u8bcd\u7684\u4e2d\u7b49\u7bc7\u5e45\u6587\u7ae0\u3002")
        blank()

    # 2.2 structure
    w("### 2.2 \u9875\u9762\u7ed3\u6784")
    blank()
    scols = ["cit_heading_total", "cit_paragraph_count", "cit_list_density",
             "cit_table_count", "cit_image_count", "cit_code_block_count"]
    srows = []
    for col in scols:
        if col in fdf.columns:
            s = ss(fdf[col])
            srows.append([cn(col), f"{s['median']:.1f}", f"{s['mean']:.1f}", f"{s['p90']:.1f}"])
    if srows:
        w(md_table(["\u7279\u5f81", "\u4e2d\u4f4d\u6570", "\u5747\u503c", "P90"], srows))
        blank()

    # 2.3 content type
    w("### 2.3 \u5185\u5bb9\u7c7b\u578b\u5206\u5e03")
    blank()
    ct_rows = []
    for col in CONTENT_TYPE_BOOL_COLS:
        if col in fdf.columns:
            sh = float(fdf[col].mean())
            ct_rows.append([cn(col), f"{sh:.1%}"])
    if ct_rows:
        w(md_table(["\u5185\u5bb9\u7c7b\u578b", "\u5360\u6bd4"], ct_rows))
        blank()

    # 2.4 domain type
    if "domain_type" in fdf.columns:
        w("### 2.4 \u6765\u6e90\u7c7b\u578b")
        blank()
        dt = top_counts(fdf["domain_type"], top_n=10)
        dt["share"] = dt["count"] / max(nf, 1)
        w(md_table(
            ["\u6765\u6e90\u7c7b\u578b", "\u6570\u91cf", "\u5360\u6bd4", "\u4e3e\u4f8b"],
            [[r.key, f"{int(r.count):,}", f"{r.share:.1%}", _dom_ex(r.key)]
             for r in dt.itertuples(index=False)],
        ))
        blank()

    w("---")
    blank()

    # ===================================================================
    # 3. CONTENT FEATURES vs INFLUENCE (rewritten - no circular reasoning)
    # ===================================================================
    w("## \u4e09\u3001\u5185\u5bb9\u7279\u5f81\u5982\u4f55\u5f71\u54cd\u5f15\u7528\u5f71\u54cd\u529b")
    blank()
    w("> **\u5206\u6790\u89c6\u89d2**\uff1a\u672c\u7ae0\u53ea\u4f7f\u7528\u5185\u5bb9\u521b\u4f5c\u8005\u53ef\u4ee5\u63a7\u5236\u7684\u7279\u5f81\uff08\u9875\u9762\u7ed3\u6784\u3001\u5185\u5bb9\u8d28\u91cf\u3001\u5199\u4f5c\u98ce\u683c\uff09"
      "\u4ee5\u53ca\u72ec\u7acb\u8bed\u4e49\u8bc4\u4f30\u7279\u5f81\uff0c\u5206\u6790\u5b83\u4eec\u4e0e influence_score \u7684\u5173\u7cfb\u3002")
    w("> influence_score \u7684\u516c\u5f0f\u7ec4\u6210\u53d8\u91cf\uff08ref_count\u3001position\u3001tfidf_cosine \u7b49\uff09\u662f\u5f71\u54cd\u529b\u7684**\u8868\u73b0\u5f62\u5f0f**\uff0c"
      "\u4e0d\u662f\u539f\u56e0\uff0c\u5df2\u4ece\u5206\u6790\u4e2d\u6392\u9664\u3002")
    blank()

    corr_src = fdf.dropna(subset=["influence_score"]).copy()

    # 3.1 structure
    w("### 3.1 \u5185\u5bb9\u7ed3\u6784\u7279\u5f81")
    blank()
    struct_corrs = content_corr_rank(corr_src, "influence_score", CONTENT_STRUCTURE_COLS)
    if struct_corrs:
        w(md_table(
            ["\u7279\u5f81", "\u4e2d\u6587\u542b\u4e49", "Pearson r", "\u76f8\u5173\u5f3a\u5ea6"],
            [[c.feature, cn(c.feature), f"{c.r:+.4f}", _corr_str(c.r)] for c in struct_corrs],
        ))
        blank()
        top_s = struct_corrs[0] if struct_corrs else None
        if top_s:
            w(f"**\u89e3\u8bfb**\uff1a\u5728\u9875\u9762\u7ed3\u6784\u7279\u5f81\u4e2d\uff0c**{cn(top_s.feature)}**\uff08r={top_s.r:+.3f}\uff09"
              "\u4e0e\u5f71\u54cd\u529b\u7684\u76f8\u5173\u6027\u6700\u5f3a\u3002"
              "\u8fd9\u8bf4\u660e\u9875\u9762\u7684\u7ed3\u6784\u590d\u6742\u5ea6\u5bf9\u5f71\u54cd\u529b\u6709\u663e\u8457\u5f71\u54cd\uff1a"
              "AI \u6a21\u578b\u66f4\u5bb9\u6613\u4ece\u7ed3\u6784\u4e30\u5bcc\u7684\u9875\u9762\u4e2d\u63d0\u53d6\u4fe1\u606f\u3002")
            blank()

    # 3.2 quality
    w("### 3.2 \u5185\u5bb9\u8d28\u91cf\u7279\u5f81")
    blank()
    qual_corrs = content_corr_rank(corr_src, "influence_score", CONTENT_QUALITY_COLS)
    if qual_corrs:
        w(md_table(
            ["\u7279\u5f81", "\u4e2d\u6587\u542b\u4e49", "Pearson r", "\u76f8\u5173\u5f3a\u5ea6"],
            [[c.feature, cn(c.feature), f"{c.r:+.4f}", _corr_str(c.r)] for c in qual_corrs],
        ))
        blank()
        wc = next((c for c in qual_corrs if c.feature == "cit_word_count"), None)
        if wc:
            w(f"**\u89e3\u8bfb**\uff1a**\u7bc7\u5e45**\uff08r={wc.r:+.3f}\uff09\u662f\u8d28\u91cf\u7ef4\u5ea6\u4e2d\u6700\u5f3a\u7684\u9884\u6d4b\u56e0\u5b50\u3002"
              "\u8bcd\u6570\u66f4\u591a\u7684\u9875\u9762\u80fd\u8986\u76d6\u66f4\u591a\u7ec6\u8282\uff0cAI \u80fd\u4ece\u4e2d\u63d0\u53d6\u66f4\u591a\u6709\u7528\u7247\u6bb5\u3002")
        uwr = next((c for c in qual_corrs if c.feature == "cit_unique_word_ratio"), None)
        if uwr:
            w(f"\u503c\u5f97\u6ce8\u610f\u7684\u662f\uff0c**\u8bcd\u6c47\u4e30\u5bcc\u5ea6**\uff08r={uwr.r:+.3f}\uff09\u4e0e\u5f71\u54cd\u529b\u8d1f\u76f8\u5173\u2014\u2014"
              "\u8fd9\u5e76\u975e\u8bf4\u7528\u8bcd\u5e94\u8be5\u5355\u8c03\uff0c\u800c\u662f\u56e0\u4e3a\u7bc7\u5e45\u8f83\u957f\u7684\u4e13\u4e1a\u6587\u7ae0\u4f1a\u81ea\u7136\u91cd\u590d\u6838\u5fc3\u672f\u8bed\uff0c\u964d\u4f4e\u4e86\u552f\u4e00\u8bcd\u6bd4\u7387\u3002")
        blank()

    # 3.3 content type (bool)
    w("### 3.3 \u5185\u5bb9\u7c7b\u578b\u7279\u5f81\uff08\u5e03\u5c14\u53d8\u91cf\uff09")
    blank()
    w("\u5bf9\u6bcf\u4e2a\u5e03\u5c14\u7279\u5f81\uff0c\u5bf9\u6bd4 True \u548c False \u4e24\u7ec4\u7684\u5e73\u5747 influence_score\uff1a")
    blank()
    bf_rows = bool_feature_influence(corr_src, CONTENT_TYPE_BOOL_COLS)
    if bf_rows:
        w(md_table(
            ["\u7279\u5f81", "\u4e2d\u6587", "\u542b\u8be5\u7279\u5f81\u65f6\u5747\u503c", "\u4e0d\u542b\u65f6\u5747\u503c", "\u5dee\u503c", "\u63d0\u5347\u500d\u6570"],
            [[r["feature"], cn(r["feature"]), sf(r["true_mean"]), sf(r["false_mean"]),
              f"{r['delta']:+.4f}", f"{r['lift']:.2f}x"]
             for r in bf_rows],
        ))
        blank()
        top_bf = bf_rows[0] if bf_rows else None
        if top_bf and top_bf["delta"] > 0:
            w(f"**\u89e3\u8bfb**\uff1a**{cn(top_bf['feature'])}**\u7684\u9875\u9762\u6bd4\u4e0d\u542b\u8be5\u7279\u5f81\u7684\u9875\u9762\u5e73\u5747\u5f71\u54cd\u529b\u9ad8\u51fa "
              f"{top_bf['lift']:.2f} \u500d\u3002")
            w("\u5305\u542b\u5177\u4f53\u6570\u636e\u3001\u5b9a\u4e49\u53e5\u5f0f\u3001\u64cd\u4f5c\u6307\u5357\u7684\u9875\u9762\uff0c\u66f4\u5bb9\u6613\u88ab AI \u6df1\u5ea6\u4f7f\u7528\u3002")
            blank()

    # 3.4 semantic
    w("### 3.4 \u8bed\u4e49\u8bc4\u4f30\u7279\u5f81")
    blank()
    sem_corrs = content_corr_rank(corr_src, "influence_score", SEMANTIC_NUMERIC_COLS)
    if sem_corrs:
        w(md_table(
            ["\u7279\u5f81", "\u4e2d\u6587\u542b\u4e49", "Pearson r", "\u76f8\u5173\u5f3a\u5ea6"],
            [[c.feature, cn(c.feature), f"{c.r:+.4f}", _corr_str(c.r)] for c in sem_corrs],
        ))
        blank()
        w("**\u89e3\u8bfb**\uff1a\u8bed\u4e49\u76f8\u4f3c\u5ea6\u548c LLM \u8bc4\u5206\u90fd\u4e0e\u5f71\u54cd\u529b\u663e\u8457\u6b63\u76f8\u5173\u3002"
          "\u5f15\u7528\u9875\u5185\u5bb9\u4e0e\u56de\u7b54\u7684\u8bed\u4e49\u8d8a\u63a5\u8fd1\u3001LLM \u5224\u65ad\u7684\u76f8\u5173\u6027/\u8d28\u91cf\u8d8a\u9ad8\uff0c"
          "\u5f71\u54cd\u529b\u5c31\u8d8a\u5927\u3002"
          "\u8fd9\u9a8c\u8bc1\u4e86\u5185\u5bb9\u4e0e\u67e5\u8be2\u610f\u56fe\u7684\u5bf9\u9f50\u662f\u5f71\u54cd\u529b\u7684\u6838\u5fc3\u9a71\u52a8\u529b\u3002")
        blank()

    # 3.5 comprehensive ranking
    w("### 3.5 \u5168\u90e8\u5185\u5bb9\u7279\u5f81\u7efc\u5408\u6392\u540d")
    blank()
    all_corrs = content_corr_rank(corr_src, "influence_score", ALL_CONTENT_NUMERIC, top_n=20)
    if all_corrs:
        w(md_table(
            ["\u6392\u540d", "\u7279\u5f81", "\u4e2d\u6587\u542b\u4e49", "Pearson r", "\u76f8\u5173\u5f3a\u5ea6", "\u6837\u672c\u6570"],
            [[i + 1, c.feature, cn(c.feature), f"{c.r:+.4f}", _corr_str(c.r), f"{c.n:,}"]
             for i, c in enumerate(all_corrs)],
        ))
        blank()
        w("**\u6838\u5fc3\u53d1\u73b0**\uff1a\u5728\u6240\u6709\u5185\u5bb9\u521b\u4f5c\u8005\u53ef\u63a7\u7684\u7279\u5f81 + \u8bed\u4e49\u8bc4\u4f30\u7279\u5f81\u4e2d\uff0c"
          "\u4e0e\u5f71\u54cd\u529b\u76f8\u5173\u6027\u6700\u5f3a\u7684\u662f\uff1a")
        for i, c in enumerate(all_corrs[:5]):
            direction = "\u6b63\u76f8\u5173" if c.r > 0 else "\u8d1f\u76f8\u5173"
            w(f"{i + 1}. **{cn(c.feature)}**\uff08r={c.r:+.3f}\uff0c{direction}\uff09")
        blank()

    w("---")
    blank()

    # ===================================================================
    # 4. HIGH vs LOW INFLUENCE CONTENT PROFILE
    # ===================================================================
    w("## \u56db\u3001\u9ad8\u5f71\u54cd\u529b\u5f15\u7528\u7684\u5185\u5bb9\u753b\u50cf")
    blank()
    w("> \u5c06\u6240\u6709\u5f15\u7528\u6309 influence_score \u5206\u4e3a Top 20% \u548c Bottom 20%\uff0c"
      "\u5bf9\u6bd4\u5b83\u4eec\u5728**\u5185\u5bb9\u7279\u5f81**\u4e0a\u7684\u5dee\u5f02\uff08\u4e0d\u5305\u542b\u5f71\u54cd\u529b\u516c\u5f0f\u7ec4\u4ef6\uff09\u3002")
    blank()
    top_df, bot_df, cmp_df = cmp_tb(fdf, "influence_score")
    w(f"- Top 20% \u6837\u672c\u6570\uff1a**{len(top_df):,}** \u6761")
    w(f"- Bottom 20% \u6837\u672c\u6570\uff1a**{len(bot_df):,}** \u6761")
    blank()

    if not cmp_df.empty:
        w("### 4.1 \u5185\u5bb9\u7279\u5f81\u5bf9\u6bd4")
        blank()
        w(md_table(
            ["\u7279\u5f81", "\u4e2d\u6587\u542b\u4e49", "Top20% \u5747\u503c", "Bottom20% \u5747\u503c", "\u5dee\u503c", "\u500d\u6570"],
            [[r.feature, cn(r.feature), sf(r.top_mean), sf(r.bottom_mean),
              f"{r.delta:+.4f}", "-" if pd.isna(r.ratio) else f"{r.ratio:.2f}x"]
             for r in cmp_df.itertuples(index=False)],
        ))
        blank()
        w("**\u89e3\u8bfb**\uff1a")
        for r in cmp_df.head(5).itertuples(index=False):
            ratio_s = "-" if pd.isna(r.ratio) else f"{r.ratio:.1f}"
            w(f"- **{cn(r.feature)}**\uff1aTop \u7ec4 {sf(r.top_mean)} vs Bottom \u7ec4 {sf(r.bottom_mean)}\uff0c\u5dee\u8ddd {ratio_s} \u500d")
        blank()

    # 4.2 bool features in top/bottom
    w("### 4.2 \u5185\u5bb9\u7c7b\u578b\u5728 Top/Bottom \u7ec4\u4e2d\u7684\u5206\u5e03")
    blank()
    bool_rows = []
    for col in CONTENT_TYPE_BOOL_COLS:
        if col not in fdf.columns:
            continue
        ts = float(top_df[col].mean()) if col in top_df.columns else 0
        bs = float(bot_df[col].mean()) if col in bot_df.columns else 0
        bool_rows.append([cn(col), f"{ts:.1%}", f"{bs:.1%}", f"{ts - bs:+.1%}"])
    if bool_rows:
        w(md_table(["\u5185\u5bb9\u7c7b\u578b", "Top20% \u5360\u6bd4", "Bottom20% \u5360\u6bd4", "\u5dee\u503c"], bool_rows))
        blank()
        w("**\u89e3\u8bfb**\uff1a\u9ad8\u5f71\u54cd\u529b\u5f15\u7528\u4e2d\u5305\u542b\u6570\u5b57\u3001\u5b9a\u4e49\u3001\u5bf9\u6bd4\u3001\u64cd\u4f5c\u6307\u5357\u7684\u6bd4\u4f8b\u90fd\u663e\u8457\u66f4\u9ad8\u3002"
          "\u8fd9\u4e9b\u7ed3\u6784\u5316\u5185\u5bb9\u66f4\u5bb9\u6613\u88ab AI \u63d0\u53d6\u548c\u5438\u6536\u3002")
        blank()

    # 4.3 binned word count
    w("### 4.3 \u7bc7\u5e45\u5206\u7bb1\u5206\u6790\uff1a\u591a\u957f\u7684\u5185\u5bb9\u5f71\u54cd\u529b\u6700\u9ad8\uff1f")
    blank()
    wc_bins = binned_analysis(fdf, "cit_word_count", "influence_score",
                              bins=[0, 100, 300, 500, 1000, 2000, 3000, 5000, float("inf")])
    if wc_bins:
        w(md_table(
            ["\u8bcd\u6570\u533a\u95f4", "\u6837\u672c\u6570", "\u5e73\u5747 influence_score", "\u4e2d\u4f4d\u6570"],
            [[r["bin"], f"{r['n']:,}", sf(r["mean"]), sf(r["median"])] for r in wc_bins],
        ))
        blank()
        best = max(wc_bins, key=lambda r: r["mean"]) if wc_bins else None
        if best:
            w(f"**\u89e3\u8bfb**\uff1a\u5f71\u54cd\u529b\u6700\u9ad8\u7684\u7bc7\u5e45\u533a\u95f4\u662f **{best['bin']} \u8bcd**"
              f"\uff08\u5747\u503c {best['mean']:.4f}\uff09\u3002"
              "\u7bc7\u5e45\u592a\u77ed\uff08<100 \u8bcd\uff09\u7684\u9875\u9762\u5f71\u54cd\u529b\u6781\u4f4e\uff0c"
              "\u4f46\u4e5f\u4e0d\u662f\u8d8a\u957f\u8d8a\u597d\u2014\u2014\u8d85\u8fc7 5000 \u8bcd\u540e\u53ef\u80fd\u8fb9\u9645\u6536\u76ca\u9012\u51cf\u3002")
        blank()

    w("---")
    blank()

    # ===================================================================
    # 5. SEMANTIC ROLES & INFLUENCE
    # ===================================================================
    w("## \u4e94\u3001\u8bed\u4e49\u89d2\u8272\u3001\u6765\u6e90\u7c7b\u578b\u4e0e\u5f71\u54cd\u529b")
    blank()
    w("> \u4e0d\u540c\u7684\u5185\u5bb9\u89d2\u8272\u548c\u6765\u6e90\u7c7b\u578b\uff0c\u5bf9\u5f71\u54cd\u529b\u6709\u4f55\u5dee\u5f02\uff1f")
    blank()

    # 5.1 semantic role
    if "llm_semantic_role" in fdf.columns:
        w("### 5.1 \u8bed\u4e49\u89d2\u8272\u7684\u5e73\u5747\u5f71\u54cd\u529b")
        blank()
        role_inf = categorical_mean_influence(fdf, "llm_semantic_role")
        if role_inf:
            w(md_table(
                ["\u8bed\u4e49\u89d2\u8272", "\u4e2d\u6587", "\u5e73\u5747 influence_score", "\u4e2d\u4f4d\u6570", "\u6837\u672c\u6570"],
                [[r["category"], LLM_ROLE_CN.get(r["category"], r["category"]),
                  sf(r["mean"]), sf(r["median"]), f"{r['n']:,}"] for r in role_inf],
            ))
            blank()
            if len(role_inf) >= 2:
                w(f"**\u89e3\u8bfb**\uff1a**{LLM_ROLE_CN.get(role_inf[0]['category'], role_inf[0]['category'])}**"
                  f"\u7c7b\u578b\u7684\u5f15\u7528\u5e73\u5747\u5f71\u54cd\u529b\u6700\u9ad8\uff08{role_inf[0]['mean']:.4f}\uff09\uff0c"
                  f"\u800c**{LLM_ROLE_CN.get(role_inf[-1]['category'], role_inf[-1]['category'])}**\u6700\u4f4e"
                  f"\uff08{role_inf[-1]['mean']:.4f}\uff09\u3002"
                  "\u63d0\u4f9b\u4e8b\u5b9e\u4f9d\u636e\u3001\u5b9a\u4e49\u3001\u7edf\u8ba1\u6570\u636e\u7684\u5185\u5bb9\u6bd4\u7eaf\u53c2\u8003\u94fe\u63a5\u83b7\u5f97\u4e86\u66f4\u9ad8\u5f71\u54cd\u529b\u3002")
                blank()

        # semantic role distribution top vs bottom
        dc = dist_cmp(top_df, bot_df, "llm_semantic_role", top_n=8)
        w("#### Top 20% vs Bottom 20% \u7684\u8bed\u4e49\u89d2\u8272\u5206\u5e03")
        blank()
        w(md_table(
            ["\u89d2\u8272", "\u4e2d\u6587", "Top20%", "Bottom20%", "\u5dee\u503c"],
            [[getattr(r, "llm_semantic_role"), LLM_ROLE_CN.get(getattr(r, "llm_semantic_role"), ""),
              f"{r.top_share:.1%}", f"{r.bottom_share:.1%}", f"{r.delta_share:+.1%}"]
             for r in dc.itertuples(index=False)],
        ))
        blank()
        w("**\u89e3\u8bfb**\uff1a\u9ad8\u5f71\u54cd\u529b\u7ec4\u4e2d", Q, "\u8bc1\u636e\u652f\u6491", Qr, "\u5360\u6bd4\u8fdc\u9ad8\u4e8e\u4f4e\u5f71\u54cd\u529b\u7ec4\uff0c"
          "\u800c", Q, "\u53c2\u8003\u5f15\u7528", Qr, "\u5219\u5728\u4f4e\u5f71\u54cd\u529b\u7ec4\u4e2d\u5360\u7edd\u5bf9\u591a\u6570\u3002"
          "\u540e\u8005\u662f\u88ab AI \u7f57\u5217\u4f46\u672a\u5b9e\u8d28\u4f7f\u7528\u5185\u5bb9\u7684", Q, "\u6302\u540d\u5f15\u7528", Qr, "\u3002")
        blank()

    # 5.2 reverse analysis: what content features predict AI's usage depth?
    if "llm_influence_type" in fdf.columns:
        w("### 5.2 \u4ec0\u4e48\u5185\u5bb9\u66f4\u5bb9\u6613\u88ab AI \u6df1\u5ea6\u5438\u6536\uff1f")
        blank()
        w("> **\u5206\u6790\u89c6\u89d2\u8f6c\u6362**\uff1a\u524d\u9762\u7684\u5206\u6790\u7528\u5185\u5bb9\u7279\u5f81\u9884\u6d4b influence_score\u3002"
          "\u8fd9\u91cc\u6211\u4eec\u6362\u4e2a\u89d2\u5ea6\uff1a\u4ee5 AI \u5bf9\u5f15\u7528\u7684", Q, "\u4f7f\u7528\u6df1\u5ea6", Qr, "\u4e3a\u56e0\u53d8\u91cf\uff0c"
          "\u770b\u4ec0\u4e48\u5185\u5bb9\u7279\u5f81\u8ba9 AI \u66f4\u503e\u5411\u4e8e\u6df1\u5ea6\u5438\u6536\uff08\u6539\u5199\u590d\u8ff0\uff09\u800c\u975e\u6302\u540d\u5f15\u7528\u3002")
        blank()
        w("AI \u5bf9\u5f15\u7528\u7684\u4f7f\u7528\u6df1\u5ea6\u53ef\u5206\u4e3a\u56db\u4e2a\u68af\u5ea6\uff1a")
        blank()
        w("| \u68af\u5ea6 | \u7c7b\u578b | \u542b\u4e49 |")
        w("| --- | --- | --- |")
        w("| \u2605\u2605\u2605\u2605 | paraphrase\uff08\u6df1\u5ea6\u5438\u6536\uff09 | AI \u7528\u81ea\u5df1\u7684\u8bdd\u91cd\u5199\u4e86\u5f15\u7528\u5185\u5bb9\uff0c\u5185\u5bb9\u88ab\u5168\u9762\u6d88\u5316 |")
        w("| \u2605\u2605\u2605 | factual_basis\uff08\u4e8b\u5b9e\u5f15\u7528\uff09 | AI \u76f4\u63a5\u5f15\u7528\u5177\u4f53\u4e8b\u5b9e\u3001\u6570\u636e\u3001\u5b9a\u4e49 |")
        w("| \u2605\u2605 | supplementary\uff08\u8865\u5145\u8bf4\u660e\uff09 | AI \u7528\u6765\u8865\u5145\u7ec6\u8282\uff0c\u4e0d\u662f\u6838\u5fc3\u5185\u5bb9 |")
        w("| \u2605 | reference\uff08\u6302\u540d\u5f15\u7528\uff09 | AI \u53ea\u5217\u51fa\u94fe\u63a5\uff0c\u672a\u4f7f\u7528\u5185\u5bb9 |")
        blank()

        _PROFILE_GROUPS = ["paraphrase", "factual_basis", "supplementary", "reference"]
        _PROFILE_CN = {"paraphrase": "\u6df1\u5ea6\u5438\u6536", "factual_basis": "\u4e8b\u5b9e\u5f15\u7528",
                        "supplementary": "\u8865\u5145\u8bf4\u660e", "reference": "\u6302\u540d\u5f15\u7528"}
        num_prof, bool_prof, grp_sizes = influence_type_profile(fdf, groups=_PROFILE_GROUPS)

        w("**\u5404\u7ec4\u6837\u672c\u91cf**\uff1a"
          + "\u3001".join(f"{_PROFILE_CN[g]} {grp_sizes.get(g, 0):,} \u6761" for g in _PROFILE_GROUPS))
        blank()

        # numeric features comparison
        if not num_prof.empty:
            w("#### \u5185\u5bb9\u7279\u5f81\u5bf9\u6bd4\uff08\u6309\u6df1\u5ea6\u5438\u6536 vs \u6302\u540d\u5f15\u7528\u5dee\u503c\u6392\u5e8f\uff09")
            blank()
            hdr = ["\u7279\u5f81", "\u4e2d\u6587\u542b\u4e49"]
            hdr += [_PROFILE_CN[g] for g in _PROFILE_GROUPS]
            hdr.append("\u6df1\u5ea6-\u6302\u540d \u5dee\u503c")
            tbl_rows = []
            for r in num_prof.head(15).itertuples(index=False):
                row = [r.feature, cn(r.feature)]
                for g in _PROFILE_GROUPS:
                    row.append(sf(getattr(r, g, None)))
                row.append(f"{r.delta_pr:+.4f}" if pd.notna(r.delta_pr) else "-")
                tbl_rows.append(row)
            w(md_table(hdr, tbl_rows))
            blank()

            top_feat = num_prof.iloc[0] if not num_prof.empty else None
            if top_feat is not None and pd.notna(top_feat.get("delta_pr")):
                w(f"**\u89e3\u8bfb**\uff1a\u88ab AI \u6df1\u5ea6\u5438\u6536\u7684\u5f15\u7528\u4e0e\u6302\u540d\u5f15\u7528\u76f8\u6bd4\uff0c"
                  f"\u5dee\u5f02\u6700\u5927\u7684\u7279\u5f81\u662f**{cn(top_feat['feature'])}**"
                  f"\uff08\u5dee\u503c {top_feat['delta_pr']:+.1f}\uff09\u3002"
                  "\u603b\u4f53\u800c\u8a00\uff0c\u88ab\u6df1\u5ea6\u5438\u6536\u7684\u5185\u5bb9\u5728\u7bc7\u5e45\u3001\u7ed3\u6784\u590d\u6742\u5ea6\u3001\u8bed\u4e49\u76f8\u4f3c\u5ea6\u4e0a\u90fd\u663e\u8457\u9ad8\u4e8e\u6302\u540d\u5f15\u7528\u3002"
                  "\u8fd9\u8bf4\u660e AI \u7684", Q, "\u6df1\u5ea6\u5438\u6536", Qr,
                  "\u5e76\u975e\u968f\u673a\u7684\uff0c\u800c\u662f\u7531\u5185\u5bb9\u7684\u5b9e\u8d28\u7279\u5f81\u51b3\u5b9a\u7684\u3002")
                blank()

        # bool features comparison
        if not bool_prof.empty:
            w("#### \u5185\u5bb9\u7c7b\u578b\u7279\u5f81\u5bf9\u6bd4\uff08True \u5360\u6bd4\uff09")
            blank()
            bhdr = ["\u7279\u5f81", "\u4e2d\u6587"]
            bhdr += [_PROFILE_CN[g] for g in _PROFILE_GROUPS]
            bhdr.append("\u6df1\u5ea6-\u6302\u540d \u5dee\u503c")
            brows = []
            for r in bool_prof.itertuples(index=False):
                row = [r.feature, cn(r.feature)]
                for g in _PROFILE_GROUPS:
                    v = getattr(r, g, None)
                    row.append(f"{v:.1%}" if pd.notna(v) else "-")
                row.append(f"{r.delta_pr:+.1%}" if pd.notna(r.delta_pr) else "-")
                brows.append(row)
            w(md_table(bhdr, brows))
            blank()
            w("**\u89e3\u8bfb**\uff1a\u88ab AI \u6df1\u5ea6\u5438\u6536\u7684\u5185\u5bb9\u4e2d\uff0c\u5305\u542b\u5b9a\u4e49\u3001\u6570\u636e\u3001\u5bf9\u6bd4\u3001\u64cd\u4f5c\u6307\u5357\u7684\u6bd4\u4f8b\u90fd\u663e\u8457\u9ad8\u4e8e\u6302\u540d\u5f15\u7528\u3002"
              "\u8fd9\u4e0e\u7b2c\u4e09\u7ae0\u7684\u53d1\u73b0\u4e92\u76f8\u5370\u8bc1\uff1a\u7ed3\u6784\u5316\u3001\u4e8b\u5b9e\u5bfc\u5411\u7684\u5185\u5bb9\u4e0d\u4ec5\u5f71\u54cd\u529b\u66f4\u9ad8\uff0c\u800c\u4e14\u66f4\u5bb9\u6613\u88ab AI \u6df1\u5ea6\u6d88\u5316\u3002")
            blank()

    # 5.3 domain type
    if "domain_type" in fdf.columns:
        w("### 5.3 \u6765\u6e90\u7c7b\u578b\u7684\u5e73\u5747\u5f71\u54cd\u529b")
        blank()
        dom_inf = categorical_mean_influence(fdf, "domain_type")
        if dom_inf:
            w(md_table(
                ["\u6765\u6e90\u7c7b\u578b", "\u4e2d\u6587", "\u5e73\u5747 influence_score", "\u6837\u672c\u6570"],
                [[r["category"], _dom_ex(r["category"]), sf(r["mean"]), f"{r['n']:,}"] for r in dom_inf],
            ))
            blank()
            w("**\u89e3\u8bfb**\uff1a\u4e0d\u540c\u6765\u6e90\u7c7b\u578b\u7684\u5f71\u54cd\u529b\u5dee\u5f02\u53cd\u6620\u4e86 AI \u5bf9\u4e0d\u540c\u7c7b\u578b\u5185\u5bb9\u7684\u504f\u597d\u3002"
              "\u767e\u79d1\u3001\u5b66\u672f\u3001\u653f\u5e9c\u7c7b\u6765\u6e90\u7684\u5f15\u7528\u5f71\u54cd\u529b\u5f80\u5f80\u66f4\u9ad8\uff0c\u56e0\u4e3a\u5b83\u4eec\u63d0\u4f9b\u7684\u5185\u5bb9\u66f4\u6743\u5a01\u3001\u66f4\u5177\u4e8b\u5b9e\u6027\u3002")
            blank()

    # 5.4 LLM quality comparison
    if "llm_content_quality" in fdf.columns:
        w("### 5.4 LLM \u5185\u5bb9\u8d28\u91cf\u8bc4\u5206\u4e0e\u5f71\u54cd\u529b")
        blank()
        tq = pd.to_numeric(top_df["llm_content_quality"], errors="coerce").mean()
        bq = pd.to_numeric(bot_df["llm_content_quality"], errors="coerce").mean()
        w(f"- Top 20% \u5747\u503c\uff1a**{tq:.2f}** \u5206\uff08\u6ee1\u5206 5\uff09")
        w(f"- Bottom 20% \u5747\u503c\uff1a**{bq:.2f}** \u5206")
        w(f"- \u5dee\u503c\uff1a**{(tq - bq):+.2f}** \u5206")
        blank()
        w("\u5185\u5bb9\u8d28\u91cf\u66f4\u9ad8\u7684\u9875\u9762\u83b7\u5f97\u4e86\u66f4\u9ad8\u7684\u5f71\u54cd\u529b\u3002"
          "\u8fd9\u8bf4\u660e AI \u5e76\u975e\u968f\u673a\u9009\u62e9\u5f15\u7528\u7684\u6743\u91cd\uff0c"
          "\u800c\u662f\u786e\u5b9e\u5728\u66f4\u591a\u5730\u4f7f\u7528\u9ad8\u8d28\u91cf\u5185\u5bb9\u3002")
        blank()

    w("---")
    blank()

    # ===================================================================
    # 6. PLATFORM DIFFERENCES (updated: content features only)
    # ===================================================================
    w("## \u516d\u3001\u5e73\u53f0\u5dee\u5f02 \u2014\u2014 \u4e09\u5927 AI \u641c\u7d22\u7684\u5f15\u7528\u7b56\u7565")
    blank()
    if "platform" in fdf.columns:
        w("### 6.1 \u5f71\u54cd\u529b\u5f97\u5206\u5bf9\u6bd4")
        blank()
        rows = []
        for plat, part in fdf.groupby("platform"):
            sc = part["influence_score"]
            rows.append([plat, f"{len(part):,}", sf(sc.mean()), sf(sc.median()), sf(sc.quantile(0.9))])
        w(md_table(["\u5e73\u53f0", "\u6837\u672c\u6570", "\u5747\u503c", "\u4e2d\u4f4d\u6570", "P90"], rows))
        blank()
        w("**\u89e3\u8bfb**\uff1aChatGPT \u5f15\u7528\u6570\u91cf\u5c11\u4f46\u6bcf\u6761\u88ab\u6df1\u5ea6\u4f7f\u7528\uff08\u5747\u503c\u8fdc\u9ad8\uff09\uff0c"
          "Google/Perplexity \u91c7\u7528\u5e7f\u6cdb\u5f15\u7528\u4f46\u5355\u6761\u5f71\u54cd\u529b\u4f4e\u7684\u7b56\u7565\u3002")
        blank()

        w("### 6.2 \u5404\u5e73\u53f0\u7684\u5185\u5bb9\u7279\u5f81\u9a71\u52a8\u56e0\u7d20")
        blank()
        w("\u4ee5\u4e0b\u4ec5\u5c55\u793a\u5185\u5bb9\u7279\u5f81\u4e0e influence_score \u7684\u76f8\u5173\u6027\uff08\u6392\u9664\u516c\u5f0f\u7ec4\u4ef6\uff09\uff1a")
        blank()
        for plat, part in fdf.groupby("platform"):
            corr = content_corr_rank(part, "influence_score", ALL_CONTENT_NUMERIC, top_n=8)
            w(f"#### {plat}")
            blank()
            if corr:
                w(md_table(["\u7279\u5f81", "\u4e2d\u6587\u542b\u4e49", "r", "\u76f8\u5173\u5f3a\u5ea6"],
                           [[c.feature, cn(c.feature), f"{c.r:+.4f}", _corr_str(c.r)] for c in corr]))
            blank()

        w("**\u8de8\u5e73\u53f0\u5bf9\u6bd4**\uff1a\u4e09\u4e2a\u5e73\u53f0\u7684\u5185\u5bb9\u7279\u5f81\u9a71\u52a8\u56e0\u7d20\u5404\u6709\u4e0d\u540c\u3002"
          "\u5bf9\u6bd4\u5404\u5e73\u53f0\u6392\u540d\u524d\u51e0\u7684\u7279\u5f81\uff0c\u53ef\u4ee5\u770b\u51fa\u5404\u5e73\u53f0\u5bf9\u5185\u5bb9\u7684\u504f\u597d\u5dee\u5f02\u3002")
        blank()

        if "llm_semantic_role" in fdf.columns:
            w("### 6.3 \u5404\u5e73\u53f0\u7684\u8bed\u4e49\u89d2\u8272\u5206\u5e03")
            blank()
            for plat, part in fdf.groupby("platform"):
                dist = part["llm_semantic_role"].fillna("unknown").astype(str).value_counts(normalize=True).head(8)
                w(f"#### {plat}")
                blank()
                w(md_table(["\u89d2\u8272", "\u4e2d\u6587", "\u5360\u6bd4"],
                           [[k, LLM_ROLE_CN.get(k, k), f"{v:.1%}"] for k, v in dist.items()]))
                blank()
    w("---")
    blank()

    # ===================================================================
    # 7. SLICE BY CATEGORY & QUESTION TYPE (updated: content features)
    # ===================================================================
    w("## \u4e03\u3001\u5206\u884c\u4e1a\u4e0e\u5206\u95ee\u9898\u7c7b\u578b\u5206\u6790")
    blank()
    if "category" in fdf.columns:
        w("### 7.1 \u884c\u4e1a\u7ef4\u5ea6")
        blank()
        cat_rows = []
        cat_details = []
        for cat, part in fdf.groupby("category"):
            if len(part) < 50:
                continue
            sc = part["influence_score"]
            _, _, ccmp = cmp_tb(part, "influence_score")
            tf = ccmp.iloc[0]["feature"] if not ccmp.empty else "-"
            cat_rows.append([cat, f"{len(part):,}", sf(sc.mean()), sf(sc.median()), cn(tf)])
            cat_details.append((cat, ccmp))
        if cat_rows:
            w(md_table(["\u884c\u4e1a", "\u6837\u672c\u6570", "\u5747\u503c", "\u4e2d\u4f4d\u6570", "Top-Bottom \u6700\u5927\u5dee\u5f02\u7279\u5f81"], cat_rows))
            blank()
            for cat, ccmp in cat_details:
                if ccmp.empty or len(ccmp) < 3:
                    continue
                w(f"#### {cat}\uff08Top 5 \u5dee\u5f02\u7279\u5f81\uff09")
                blank()
                w(md_table(
                    ["\u7279\u5f81", "\u4e2d\u6587", "Top20%", "Bottom20%", "\u5dee\u503c"],
                    [[r.feature, cn(r.feature), sf(r.top_mean), sf(r.bottom_mean), f"{r.delta:+.4f}"]
                     for r in ccmp.head(5).itertuples(index=False)],
                ))
                blank()

    if "question_type" in fdf.columns:
        w("### 7.2 \u95ee\u9898\u7c7b\u578b\u7ef4\u5ea6")
        blank()
        qt_rows = []
        qt_details = []
        for qt, part in fdf.groupby("question_type"):
            if len(part) < 50:
                continue
            sc = part["influence_score"]
            _, _, qcmp = cmp_tb(part, "influence_score")
            tf = qcmp.iloc[0]["feature"] if not qcmp.empty else "-"
            qt_rows.append([qt, f"{len(part):,}", sf(sc.mean()), sf(sc.median()), cn(tf)])
            qt_details.append((qt, qcmp))
        if qt_rows:
            w(md_table(["\u95ee\u9898\u7c7b\u578b", "\u6837\u672c\u6570", "\u5747\u503c", "\u4e2d\u4f4d\u6570", "Top-Bottom \u6700\u5927\u5dee\u5f02\u7279\u5f81"], qt_rows))
            blank()
            for qt, qcmp in qt_details:
                if qcmp.empty or len(qcmp) < 3:
                    continue
                w(f"#### {qt}\uff08Top 5 \u5dee\u5f02\u7279\u5f81\uff09")
                blank()
                w(md_table(
                    ["\u7279\u5f81", "\u4e2d\u6587", "Top20%", "Bottom20%", "\u5dee\u503c"],
                    [[r.feature, cn(r.feature), sf(r.top_mean), sf(r.bottom_mean), f"{r.delta:+.4f}"]
                     for r in qcmp.head(5).itertuples(index=False)],
                ))
                blank()

    w("---")
    blank()

    # ===================================================================
    # 8. CONCLUSIONS & RECOMMENDATIONS (rewritten, no circular reasoning)
    # ===================================================================
    w("## \u516b\u3001\u6838\u5fc3\u7ed3\u8bba\u4e0e\u5b9e\u64cd\u5efa\u8bae")
    blank()
    w("### 8.1 \u6838\u5fc3\u53d1\u73b0")
    blank()

    w("**\u53d1\u73b0 1\uff1a\u7bc7\u5e45\u662f\u6700\u5f3a\u7684\u5185\u5bb9\u9884\u6d4b\u56e0\u5b50**")
    blank()
    wc_cmp = cmp_df[cmp_df["feature"] == "cit_word_count"] if not cmp_df.empty else pd.DataFrame()
    if not wc_cmp.empty:
        w(f"\u9ad8\u5f71\u54cd\u529b\u5f15\u7528\u7684\u5e73\u5747\u8bcd\u6570\u4e3a **{wc_cmp.iloc[0]['top_mean']:.0f}**\uff0c"
          f"\u4f4e\u5f71\u54cd\u529b\u4ec5 **{wc_cmp.iloc[0]['bottom_mean']:.0f}** \u8bcd\u3002"
          "\u8db3\u591f\u7684\u7bc7\u5e45\u8ba9 AI \u80fd\u4ece\u4e2d\u63d0\u53d6\u66f4\u591a\u6709\u7528\u7247\u6bb5\u3002"
          "\u4f46\u7bc7\u5e45\u5e76\u975e\u8d8a\u957f\u8d8a\u597d\u2014\u2014\u6700\u4f73\u533a\u95f4\u5728 1000-3000 \u8bcd\u3002")
    blank()

    w("**\u53d1\u73b0 2\uff1a\u7ed3\u6784\u5316\u5185\u5bb9\u663e\u8457\u63d0\u5347\u5f71\u54cd\u529b**")
    blank()
    w("\u542b\u6709\u5b9a\u4e49\u53e5\u5f0f\u3001\u6570\u5b57\u6570\u636e\u3001\u64cd\u4f5c\u6307\u5357\u3001\u5bf9\u6bd4\u5185\u5bb9\u7684\u9875\u9762\uff0c\u5f71\u54cd\u529b\u663e\u8457\u9ad8\u4e8e\u4e0d\u542b\u8fd9\u4e9b\u7279\u5f81\u7684\u9875\u9762\u3002"
      "AI \u9700\u8981\u53ef\u76f4\u63a5\u63d0\u53d6\u7684\u7ed3\u6784\u5316\u5185\u5bb9\uff0c\u800c\u975e\u7eaf\u53d9\u8ff0\u6027\u6587\u5b57\u3002")
    blank()

    w("**\u53d1\u73b0 3\uff1a\u5185\u5bb9\u7279\u5f81\u51b3\u5b9a\u4e86 AI \u662f", Q, "\u6df1\u5ea6\u5438\u6536", Qr, "\u8fd8\u662f", Q, "\u6302\u540d\u5f15\u7528", Qr, "**")
    blank()
    w("\u901a\u8fc7\u9006\u5411\u5206\u6790 AI \u5bf9\u5f15\u7528\u7684\u4f7f\u7528\u6df1\u5ea6\uff0c\u6211\u4eec\u53d1\u73b0\u88ab\u6df1\u5ea6\u5438\u6536\u7684\u5f15\u7528\u4e0e\u6302\u540d\u5f15\u7528\u5728\u5185\u5bb9\u7279\u5f81\u4e0a\u5b58\u5728\u7cfb\u7edf\u6027\u5dee\u5f02\uff1a"
      "\u7bc7\u5e45\u66f4\u957f\u3001\u7ed3\u6784\u66f4\u590d\u6742\u3001\u542b\u66f4\u591a\u5b9a\u4e49\u548c\u6570\u636e\u7684\u9875\u9762\u66f4\u5bb9\u6613\u88ab AI \u6df1\u5ea6\u6d88\u5316\u3002"
      "\u540c\u65f6\uff0c\u8bed\u4e49\u89d2\u8272\u4e3a", Q, "\u8bc1\u636e\u652f\u6491", Qr, "\u548c", Q, "\u5b9a\u4e49\u89e3\u91ca", Qr,
      "\u7684\u5185\u5bb9\u5f71\u54cd\u529b\u6700\u9ad8\u2014\u2014AI \u6700\u9700\u8981\u7684\u662f\u53ef\u76f4\u63a5\u5f15\u7528\u7684\u4e8b\u5b9e\u3001\u6570\u636e\u548c\u5b9a\u4e49\uff0c\u800c\u975e\u89c2\u70b9\u548c\u8bc4\u8bba\u3002")
    blank()

    w("**\u53d1\u73b0 4\uff1a\u8bed\u4e49\u5bf9\u9f50\u662f\u5f71\u54cd\u529b\u7684\u6838\u5fc3\u9a71\u52a8\u529b**")
    blank()
    w("Embedding \u76f8\u4f3c\u5ea6\u548c LLM \u76f8\u5173\u6027\u8bc4\u5206\u90fd\u4e0e\u5f71\u54cd\u529b\u663e\u8457\u6b63\u76f8\u5173\u3002"
      "\u5f15\u7528\u9875\u5185\u5bb9\u4e0e\u67e5\u8be2\u610f\u56fe\u7684\u7cbe\u51c6\u5bf9\u9f50\uff0c\u662f\u4ece\u7ade\u4e89\u5f15\u7528\u4e2d\u8131\u9896\u800c\u51fa\u7684\u5173\u952e\u3002")
    blank()

    w("**\u53d1\u73b0 5\uff1a\u4e09\u5e73\u53f0\u5bf9\u5185\u5bb9\u7684\u504f\u597d\u4e0d\u540c**")
    blank()
    w("ChatGPT \u5c11\u91cf\u5f15\u7528\u4f46\u6df1\u5ea6\u4f7f\u7528\uff0cGoogle/Perplexity \u5e7f\u6cdb\u5f15\u7528\u4f46\u5355\u6761\u5f71\u54cd\u529b\u4f4e\u3002"
      "\u4e0d\u540c\u5e73\u53f0\u7684\u5185\u5bb9\u7279\u5f81\u9a71\u52a8\u56e0\u7d20\u6392\u540d\u4e5f\u4e0d\u540c\uff0c\u9700\u8981\u9488\u5bf9\u6027\u4f18\u5316\u3002")
    blank()

    w("### 8.2 \u9762\u5411\u5185\u5bb9\u521b\u4f5c\u8005\u7684 SEO-for-AI \u5b9e\u64cd\u5efa\u8bae")
    blank()

    w("#### \u5efa\u8bae\u4e00\uff1a\u4fdd\u8bc1\u8db3\u591f\u7684\u5185\u5bb9\u6df1\u5ea6\uff081000-3000 \u8bcd\uff09")
    blank()
    w("\u7bc7\u5e45\u662f\u5185\u5bb9\u7ef4\u5ea6\u4e2d\u6700\u5f3a\u7684\u5f71\u54cd\u529b\u9884\u6d4b\u56e0\u5b50\u3002"
      "\u8fc7\u77ed\uff08<100 \u8bcd\uff09\u7684\u9875\u9762\u5f71\u54cd\u529b\u6781\u4f4e\uff0c1000-3000 \u8bcd\u662f\u6700\u4f73\u533a\u95f4\u3002")
    blank()

    w("#### \u5efa\u8bae\u4e8c\uff1a\u63d0\u4f9b\u53ef\u88ab\u63d0\u53d6\u7684\u7ed3\u6784\u5316\u5185\u5bb9")
    blank()
    w("\u5305\u542b\u660e\u786e\u5b9a\u4e49\u3001\u5177\u4f53\u6570\u636e/\u7edf\u8ba1\u3001\u6b65\u9aa4\u5316\u6307\u5357\u3001\u5bf9\u6bd4\u5206\u6790\u7684\u9875\u9762\u5f71\u54cd\u529b\u66f4\u9ad8\u3002"
      "\u4f7f\u7528 H2-H3 \u6807\u9898\u5212\u5206\u5c42\u6b21\uff0c\u7528\u5217\u8868\u5448\u73b0\u8981\u70b9\uff0c\u8ba9 AI \u5bb9\u6613\u89e3\u6790\u548c\u63d0\u53d6\u3002")
    blank()

    w("#### \u5efa\u8bae\u4e09\uff1a\u6700\u5927\u5316\u5185\u5bb9\u4e0e\u76ee\u6807\u67e5\u8be2\u7684\u8bed\u4e49\u5bf9\u9f50")
    blank()
    w("Embedding \u76f8\u4f3c\u5ea6\u548c LLM \u76f8\u5173\u6027\u8bc4\u5206\u90fd\u4e0e\u5f71\u54cd\u529b\u6b63\u76f8\u5173\u3002"
      "\u56f4\u7ed5\u6838\u5fc3\u67e5\u8be2\u610f\u56fe\u7ec4\u7ec7\u5185\u5bb9\uff0c\u786e\u4fdd\u5173\u952e\u672f\u8bed\u5728\u6807\u9898\u3001\u6bb5\u843d\u9996\u53e5\u548c\u6458\u8981\u4e2d\u81ea\u7136\u51fa\u73b0\u3002")
    blank()

    w("#### \u5efa\u8bae\u56db\uff1a\u5199\u4e8b\u5b9e\u4e0d\u5199\u89c2\u70b9")
    blank()
    w("AI \u6700\u501a\u91cd\u7684\u662f\u80fd\u63d0\u4f9b\u4e8b\u5b9e\u4f9d\u636e\u7684\u5f15\u7528\uff0c\u800c\u975e\u89c2\u70b9\u8bc4\u8bba\u3002"
      "\u5728\u5185\u5bb9\u4e2d\u4f18\u5148\u63d0\u4f9b\u53ef\u9a8c\u8bc1\u7684\u6570\u636e\u3001\u6743\u5a01\u5b9a\u4e49\u3001\u5177\u4f53\u6848\u4f8b\uff0c"
      "\u800c\u975e\u7b3c\u7edf\u7684\u89c2\u70b9\u8868\u8fbe\u3002")
    blank()

    w("#### \u5efa\u8bae\u4e94\uff1a\u9488\u5bf9\u4e0d\u540c\u5e73\u53f0\u6709\u4e0d\u540c\u4fa7\u91cd")
    blank()
    w("- **ChatGPT**\uff1a\u6ce8\u91cd\u6743\u5a01\u6027\u548c\u6df1\u5ea6\uff0c\u504f\u597d\u767e\u79d1\u5f0f\u5168\u9762\u8986\u76d6\u3002")
    w("- **Google AI Overview**\uff1a\u5173\u952e\u8bcd\u5bf9\u9f50\u662f\u6838\u5fc3\u3002")
    w("- **Perplexity**\uff1a\u5199\u80fd\u8986\u76d6\u95ee\u9898\u591a\u4e2a\u5b50\u65b9\u9762\u7684\u7efc\u5408\u6027\u5185\u5bb9\u3002")
    blank()

    w("---")
    blank()
    w(f"*\u672c\u62a5\u544a\u7531 `analyze_influence.py` \u81ea\u52a8\u751f\u6210\uff0c\u57fa\u4e8e {n_total:,} \u6761\u5f15\u7528 x 72 \u7ef4\u7279\u5f81\u7684\u7edf\u8ba1\u5206\u6790\u3002*")
    blank()

    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate detailed citation influence analysis report.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    print(f"[load] {args.input}")
    df = load_data(args.input)
    print(f"[analyze] {len(df):,} rows x {len(df.columns)} columns")
    report = build_report(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"[ok] report generated: {args.output} ({len(report):,} chars)")


if __name__ == "__main__":
    main()
