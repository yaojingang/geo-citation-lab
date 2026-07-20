#!/usr/bin/env python3
"""Build reproducible Parquet and DuckDB layers from the immutable JSONL release."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import re
import shutil
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import duckdb
from duckdb import sqltypes


DERIVED_DIRS = ("curated", "features", "marts", "quality", "catalog")
PIPELINE_VERSION = "1.0.1"
CHINA_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")
TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "msclkid",
    "share_source",
    "share_token",
    "spm",
    "spm_id_from",
}
EXPIRED_URL_MARKERS = {"expired_url", "expired", "url_expired"}
SOURCE_CATEGORY_L1 = {
    "platform_community": "平台与社区",
    "news_media": "新闻与媒体",
    "vertical_professional": "垂直专业内容",
    "business_services": "商业信息与服务",
    "research_documentation": "研究与文档",
    "government_public": "政府与公共机构",
    "brand_corporate": "品牌与企业官网",
    "search_page_proxy": "搜索与页面代理",
    "unclassified_long_tail": "未分类长尾",
}
SOURCE_TYPE_BY_CATEGORY = {
    "platform_community": {
        "content_platform": "内容平台",
        "social_content_platform": "社交内容平台",
        "professional_community": "专业技术社区",
        "knowledge_community": "知识社区",
        "finance_community": "财经社区",
        "interest_community": "兴趣与生活社区",
        "video_community": "视频社区",
    },
    "news_media": {
        "news_media": "综合新闻媒体",
        "regional_news_media": "地方新闻媒体",
        "business_media": "商业财经媒体",
        "industry_news_media": "行业新闻媒体",
    },
    "vertical_professional": {
        "health_content": "医疗健康内容",
        "legal_content": "法律专业内容",
        "finance_content": "财经专业内容",
        "technology_content": "科技专业内容",
        "automotive_content": "汽车专业内容",
        "consumer_lifestyle_content": "消费与生活内容",
        "industry_content": "行业专业内容",
        "education_content": "教育专业内容",
        "travel_content": "旅游专业内容",
    },
    "business_services": {
        "commercial_recommendation": "商业推荐与榜单内容",
        "local_service_platform": "本地服务与用户内容",
        "business_information_service": "商业信息与企业服务",
        "business_content_platform": "商业内容平台",
        "business_franchise": "商业信息与加盟服务",
        "ecommerce": "电商平台",
        "job_service": "招聘与职业服务",
        "shopping_information_service": "购物信息与服务",
        "travel_service": "旅行与酒店信息服务",
    },
    "research_documentation": {
        "document_platform": "文档平台",
        "knowledge_base": "专业知识库",
        "industry_research": "行业研究与数据",
    },
    "government_public": {
        "government_public": "政府机构",
        "public_institution": "公共机构",
    },
    "brand_corporate": {"brand_corporate": "品牌与企业官网"},
    "search_page_proxy": {
        "search_proxy": "搜索与页面代理",
        "map_page_proxy": "地图与页面代理",
        "content_aggregation_proxy": "内容聚合与页面代理",
    },
    "unclassified_long_tail": {"unclassified": "未分类"},
}
REFERENCE_GOVERNANCE_COMBINATIONS = {
    ("legacy_curated", "curated", "high"),
    ("manual_review", "reviewed", "high"),
    ("manual_review", "reviewed", "medium"),
    ("manual_review", "reviewed_unclassified", "low"),
}
SOURCE_TYPE_REFERENCE_FIELDS = {
    "domain",
    "source_category_l1",
    "source_category_l1_cn",
    "source_type",
    "source_type_cn",
    "ecosystem",
    "classification_status",
    "classification_method",
    "classification_confidence",
    "classification_evidence",
}


def sql_literal(value: str | Path) -> str:
    """Escape a string for use as a DuckDB SQL literal."""
    return str(value).replace("'", "''")


def blank_to_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@lru_cache(maxsize=300_000)
def _url_parts(raw_value: str | None) -> tuple[str | None, str, str | None]:
    value = blank_to_none(raw_value)
    if value is None:
        return None, "missing", None
    if value.lower() in EXPIRED_URL_MARKERS:
        return None, "expired", None
    try:
        parts = urlsplit(value)
    except ValueError:
        return None, "invalid", None
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        return None, "invalid_scheme" if scheme else "invalid", None
    try:
        hostname = parts.hostname.lower() if parts.hostname else None
        port = parts.port
    except ValueError:
        return None, "invalid", None
    if hostname is None:
        return None, "invalid", None
    host = hostname
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{hostname}:{port}"
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    query_pairs = []
    for key, query_value in parse_qsl(parts.query, keep_blank_values=True):
        normalized_key = key.lower()
        if normalized_key.startswith("utm_") or normalized_key in TRACKING_PARAMETERS:
            continue
        query_pairs.append((key, query_value))
    query_pairs.sort(key=lambda item: (item[0], item[1]))
    canonical = urlunsplit((scheme, host, path, urlencode(query_pairs, doseq=True), ""))
    return canonical, "valid_http", hostname


def canonicalize_url(value: str | None) -> str | None:
    return _url_parts(value)[0]


def classify_url(value: str | None) -> str:
    return _url_parts(value)[1]


def url_hostname(value: str | None) -> str | None:
    return _url_parts(value)[2]


def normalize_domain(raw_domain: str | None, canonical_url: str | None) -> str | None:
    domain = blank_to_none(raw_domain)
    if domain:
        domain = domain.lower().rstrip(".")
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    if canonical_url:
        return url_hostname(canonical_url)
    return None


def validate_source_type_reference(path: Path) -> dict[str, int]:
    """Fail fast when the governed exact-domain classification reference is invalid."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(SOURCE_TYPE_REFERENCE_FIELDS - fieldnames)
        if missing:
            raise RuntimeError(f"source type reference missing fields: {', '.join(missing)}")
        rows = list(reader)

    seen_domains: set[str] = set()
    method_counts: dict[str, int] = {}
    for line_number, row in enumerate(rows, start=2):
        domain = row["domain"]
        normalized_domain = normalize_domain(domain, None)
        if not domain or domain != normalized_domain:
            raise RuntimeError(
                f"source type reference domain is not normalized at line {line_number}: {domain!r}"
            )
        if domain in seen_domains:
            raise RuntimeError(f"source type reference duplicate domain: {domain}")
        seen_domains.add(domain)

        category = row["source_category_l1"]
        category_cn = SOURCE_CATEGORY_L1.get(category)
        if category_cn is None or row["source_category_l1_cn"] != category_cn:
            raise RuntimeError(
                f"source type reference invalid source_category_l1 at line {line_number}: {category}"
            )
        source_type = row["source_type"]
        source_type_cn = SOURCE_TYPE_BY_CATEGORY[category].get(source_type)
        if source_type_cn is None or row["source_type_cn"] != source_type_cn:
            raise RuntimeError(
                f"source type reference invalid source_type pair at line {line_number}: "
                f"{category}/{source_type}"
            )

        governance = (
            row["classification_method"],
            row["classification_status"],
            row["classification_confidence"],
        )
        if governance not in REFERENCE_GOVERNANCE_COMBINATIONS:
            raise RuntimeError(
                f"source type reference invalid governance combination at line {line_number}: "
                f"{'/'.join(governance)}"
            )
        is_uncertain = row["classification_status"] == "reviewed_unclassified"
        if (category == "unclassified_long_tail") != is_uncertain or (
            source_type == "unclassified"
        ) != is_uncertain:
            raise RuntimeError(
                f"source type reference governance and taxonomy disagree at line {line_number}"
            )
        if not row["classification_evidence"].strip():
            raise RuntimeError(
                f"source type reference evidence is blank at line {line_number}: {domain}"
            )
        method_counts[row["classification_method"]] = (
            method_counts.get(row["classification_method"], 0) + 1
        )

    return {"rows": len(rows), "unique_domains": len(seen_domains), **method_counts}


@lru_cache(maxsize=200_000)
def _date_parts(raw_value: str | None) -> tuple[str | None, str]:
    value = blank_to_none(raw_value)
    if value is None:
        return None, "missing"
    if value in {"0", "0.0"}:
        return None, "placeholder_zero"

    if re.fullmatch(r"\d{9,10}(?:\.0+)?", value):
        try:
            parsed = datetime.fromtimestamp(float(value), tz=CHINA_TIMEZONE)
            if 1990 <= parsed.year <= 2100:
                return parsed.isoformat(), "parsed_unix_seconds"
        except (OSError, OverflowError, ValueError):
            pass
    if re.fullmatch(r"\d{13}(?:\.0+)?", value):
        try:
            parsed = datetime.fromtimestamp(float(value) / 1000, tz=CHINA_TIMEZONE)
            if 2000 <= parsed.year <= 2100:
                return parsed.isoformat(), "parsed_unix_milliseconds"
        except (OSError, OverflowError, ValueError):
            pass

    chinese_match = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日?", value)
    if chinese_match:
        try:
            parsed_date = date(*(int(part) for part in chinese_match.groups()))
            return parsed_date.isoformat(), "parsed_chinese_date"
        except ValueError:
            return None, "unparsed"

    year_match = re.fullmatch(r"(\d{4})年", value)
    if year_match:
        year = int(year_match.group(1))
        if 1900 <= year <= 2100:
            return str(year), "partial_year"

    normalized = value.replace("/", "-").replace(".", "-")
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", normalized):
        try:
            parsed_date = date.fromisoformat("-".join(f"{int(part):02d}" if i else part for i, part in enumerate(normalized.split("-"))))
            return parsed_date.isoformat(), "parsed_date"
        except ValueError:
            return None, "unparsed"

    iso_candidate = value.replace("Z", "+00:00")
    try:
        parsed_datetime = datetime.fromisoformat(iso_candidate)
        return parsed_datetime.isoformat(), "parsed_iso_datetime"
    except ValueError:
        return None, "unparsed"


def normalize_published_at(value: str | None) -> str | None:
    return _date_parts(value)[0]


def classify_published_at(value: str | None) -> str:
    return _date_parts(value)[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_release(repo_root: Path, manifest: dict) -> dict[str, str]:
    """Verify raw shards and reject credentials embedded in citation URLs."""
    expected_files = []
    for category in manifest["categories"]:
        expected_files.extend(category["files"])
    expected_paths = [entry["path"] for entry in expected_files]
    if len(expected_paths) != len(set(expected_paths)):
        raise RuntimeError("manifest.json 中存在重复的原始分片路径")
    raw_root = (repo_root / "data" / "records").resolve()
    actual_paths = sorted((repo_root / "data" / "records").rglob("*.jsonl"))
    if len(actual_paths) != len(expected_files):
        raise RuntimeError(f"原始分片数量异常：期望 {len(expected_files)}，实际 {len(actual_paths)}")
    checksums: dict[str, str] = {}
    for file_entry in expected_files:
        relative_path = file_entry["path"]
        path = repo_root / relative_path
        try:
            path.resolve().relative_to(raw_root)
        except ValueError as error:
            raise RuntimeError(f"原始分片路径超出 data/records：{relative_path}") from error
        if not path.is_file():
            raise RuntimeError(f"缺少原始分片：{relative_path}")
        actual_hash = sha256_file(path)
        if actual_hash != file_entry["sha256"]:
            raise RuntimeError(f"原始分片校验失败：{relative_path}")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"原始分片 JSON 无效：{relative_path}:{line_number}"
                    ) from error
                quote_url = record.get("quote_url")
                if not isinstance(quote_url, str) or not quote_url:
                    continue
                parts = urlsplit(quote_url)
                if parts.username is not None or parts.password is not None:
                    raise RuntimeError(
                        f"引用 URL 含用户凭据：{relative_path}:{line_number}"
                    )
        checksums[relative_path] = actual_hash
    return checksums


def register_cleaning_functions(connection: duckdb.DuckDBPyConnection) -> None:
    options = {"null_handling": "special"}
    connection.create_function("canonicalize_url", canonicalize_url, [sqltypes.VARCHAR], sqltypes.VARCHAR, **options)
    connection.create_function("classify_url", classify_url, [sqltypes.VARCHAR], sqltypes.VARCHAR, **options)
    connection.create_function("normalize_domain", normalize_domain, [sqltypes.VARCHAR, sqltypes.VARCHAR], sqltypes.VARCHAR, **options)
    connection.create_function("normalize_published_at", normalize_published_at, [sqltypes.VARCHAR], sqltypes.VARCHAR, **options)
    connection.create_function("classify_published_at", classify_published_at, [sqltypes.VARCHAR], sqltypes.VARCHAR, **options)


def copy_parquet(connection: duckdb.DuckDBPyConnection, query: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"COPY ({query}) TO '{sql_literal(destination)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )


def build_staging(
    connection: duckdb.DuckDBPyConnection,
    repo_root: Path,
    manifest: dict,
) -> None:
    raw_glob = repo_root / "data" / "records" / "**" / "*.jsonl"
    repo_prefix = sql_literal(f"{repo_root}/")
    release_date = manifest["release_date"]
    connection.execute(
        f"""
        CREATE TEMP TABLE stg_citations AS
        WITH raw_source AS (
            SELECT *,
                   json_extract_string(quote_index, '$') AS quote_position_raw,
                   canonicalize_url(quote_url) AS canonical_url,
                   normalize_published_at(published_at) AS published_at_normalized,
                   regexp_replace(lower(trim(prompt)), '\\s+', ' ', 'g') AS prompt_normalized
            FROM read_json_auto(
                '{sql_literal(raw_glob)}',
                format='newline_delimited',
                union_by_name=true,
                maximum_depth=-1,
                filename=true
            )
        ), normalized AS (
            SELECT *,
                   classify_url(quote_url) AS url_status,
                   normalize_domain(domain, canonical_url) AS domain_normalized,
                   classify_published_at(published_at) AS published_at_parse_status,
                   CASE
                       WHEN quote_position_raw IS NULL OR trim(quote_position_raw) = '' THEN NULL
                       WHEN try_cast(quote_position_raw AS INTEGER) IS NOT NULL THEN try_cast(quote_position_raw AS INTEGER)
                       WHEN regexp_matches(quote_position_raw, '^web_search:[0-9]+#[0-9]+$')
                           THEN try_cast(regexp_extract(quote_position_raw, '#([0-9]+)$', 1) AS INTEGER)
                       ELSE NULL
                   END AS quote_position_normalized,
                   CASE
                       WHEN quote_position_raw IS NULL OR trim(quote_position_raw) = '' THEN 'missing'
                       WHEN try_cast(quote_position_raw AS INTEGER) IS NOT NULL THEN 'numeric'
                       WHEN regexp_matches(quote_position_raw, '^web_search:[0-9]+#[0-9]+$') THEN 'web_search'
                       ELSE 'unparsed'
                   END AS quote_index_parse_status,
                   CASE
                       WHEN prompt_id IS NOT NULL THEN 'q-' || lpad(cast(prompt_id AS VARCHAR), 6, '0')
                       ELSE 'q-u-' || substr(sha256(prompt_normalized), 1, 16)
                   END AS question_id
            FROM raw_source
        ), identified AS (
            SELECT *,
                   CASE WHEN domain_normalized IS NULL THEN NULL ELSE 'src-' || substr(sha256(domain_normalized), 1, 20) END AS source_id,
                   CASE WHEN canonical_url IS NULL THEN NULL ELSE 'page-' || substr(sha256(canonical_url), 1, 20) END AS page_id,
                   count(*) OVER (PARTITION BY record_hash) AS occurrence_count,
                   row_number() OVER (PARTITION BY record_hash ORDER BY record_id) = 1 AS is_preferred_exact_record,
                   row_number() OVER (PARTITION BY filename ORDER BY record_id) AS source_row_number
            FROM normalized
        )
        SELECT
            record_id AS citation_id,
            question_id,
            CAST(NULL AS VARCHAR) AS response_id,
            platform_code,
            page_id,
            source_id,
            quote_position_raw,
            quote_position_normalized,
            quote_index_parse_status,
            quote_url AS quote_url_raw,
            canonical_url,
            url_status,
            nullif(trim(quote_title), '') AS quote_title,
            nullif(trim(site_name), '') AS site_name_raw,
            published_at AS published_at_raw,
            published_at_normalized,
            try_cast(substr(published_at_normalized, 1, 10) AS DATE) AS published_date,
            published_at_parse_status,
            nullif(trim(snippet), '') AS snippet,
            nullif(lower(trim(domain)), '') AS raw_domain,
            domain_normalized,
            layer AS source_layer,
            subcat AS source_subcat,
            record_hash,
            occurrence_count,
            occurrence_count > 1 AS is_exact_duplicate,
            is_preferred_exact_record,
            concat_ws('|',
                CASE WHEN published_at_parse_status IN ('missing', 'placeholder_zero', 'partial_year', 'unparsed') THEN 'published_date_unavailable' END,
                CASE WHEN quote_index_parse_status = 'missing' THEN 'quote_index_unavailable' END,
                CASE WHEN domain_normalized IS NULL THEN 'domain_unavailable' END,
                CASE WHEN nullif(trim(quote_title), '') IS NULL THEN 'title_unavailable' END,
                CASE WHEN nullif(trim(site_name), '') IS NULL THEN 'site_name_unavailable' END,
                CASE WHEN nullif(trim(snippet), '') IS NULL THEN 'snippet_unavailable' END
            ) AS availability_flags,
            concat_ws('|',
                CASE WHEN url_status <> 'valid_http' THEN 'url_' || url_status END,
                CASE WHEN published_at_parse_status IN ('unparsed', 'placeholder_zero') THEN 'published_at_' || published_at_parse_status END,
                CASE WHEN quote_index_parse_status IN ('missing', 'unparsed') THEN 'quote_index_' || quote_index_parse_status END,
                CASE WHEN domain_normalized IS NULL THEN 'domain_missing' END,
                CASE WHEN nullif(trim(quote_title), '') IS NULL THEN 'title_missing' END,
                CASE WHEN nullif(trim(site_name), '') IS NULL THEN 'site_name_missing' END,
                CASE WHEN nullif(trim(snippet), '') IS NULL THEN 'snippet_missing' END,
                CASE WHEN occurrence_count > 1 THEN 'exact_duplicate_group' END
            ) AS quality_flags,
            replace(filename, '{repo_prefix}', '') AS source_file,
            source_row_number,
            DATE '{release_date}' AS release_date,
            prompt_id AS legacy_prompt_id,
            prompt,
            prompt_normalized
        FROM identified
        """
    )


def build_curated_tables(
    connection: duckdb.DuckDBPyConnection,
    repo_root: Path,
    build_root: Path,
    release_date: str,
) -> dict[str, int]:
    reference_root = repo_root / "data" / "reference"
    taxonomy_csv = reference_root / "question_taxonomy.csv"
    platforms_csv = reference_root / "ai_platforms.csv"
    source_types_csv = reference_root / "source_types.csv"

    connection.execute(
        f"CREATE TEMP VIEW ref_taxonomy AS SELECT * FROM read_csv('{sql_literal(taxonomy_csv)}', header=true, all_varchar=true)"
    )
    connection.execute(
        f"CREATE TEMP VIEW ref_platforms AS SELECT * FROM read_csv('{sql_literal(platforms_csv)}', header=true, all_varchar=true)"
    )
    connection.execute(
        f"CREATE TEMP VIEW ref_source_types AS SELECT * FROM read_csv('{sql_literal(source_types_csv)}', header=true, all_varchar=true)"
    )

    curated = build_root / "curated"
    citation_path = curated / "citation_observations" / f"release_date={release_date}" / "part-0001.parquet"
    copy_parquet(
        connection,
        """
        SELECT * EXCLUDE (legacy_prompt_id, prompt, prompt_normalized)
        FROM stg_citations
        ORDER BY citation_id
        """,
        citation_path,
    )

    questions_query = f"""
        SELECT
            question_id,
            min(legacy_prompt_id) AS legacy_prompt_id,
            mode(prompt ORDER BY prompt) AS prompt,
            mode(prompt_normalized ORDER BY prompt_normalized) AS prompt_normalized,
            mode(source_layer ORDER BY source_layer) AS source_layer,
            mode(source_subcat ORDER BY source_subcat) AS source_subcat,
            bool_or(legacy_prompt_id IS NOT NULL) AS is_classified,
            min(citation_id) AS first_record_id,
            count(*) AS citation_record_count,
            count(DISTINCT platform_code) AS platform_count,
            count(DISTINCT source_id) AS source_count,
            DATE '{release_date}' AS created_from_release
        FROM stg_citations
        GROUP BY question_id
        ORDER BY legacy_prompt_id NULLS LAST, question_id
    """
    copy_parquet(connection, questions_query, curated / "questions.parquet")

    labels_query = f"""
        SELECT DISTINCT
            s.question_id,
            t.label_dimension,
            t.label_value,
            t.label_cn,
            1.0::DOUBLE AS confidence,
            'legacy_directory' AS label_source,
            '1.0.0' AS taxonomy_version,
            s.source_layer,
            s.source_subcat,
            DATE '{release_date}' AS created_from_release
        FROM stg_citations s
        INNER JOIN ref_taxonomy t
          ON s.source_layer = t.source_layer AND s.source_subcat = t.source_subcat
        ORDER BY s.question_id, t.label_dimension, t.label_value
    """
    copy_parquet(connection, labels_query, curated / "question_labels.parquet")

    platforms_query = f"""
        SELECT
            p.platform_code,
            p.platform_name_cn,
            p.product_family,
            p.terminal,
            p.company_ecosystem,
            p.mapping_status,
            p.mapping_note,
            count(s.citation_id) AS citation_record_count,
            count(DISTINCT s.question_id) AS question_count,
            DATE '{release_date}' AS created_from_release
        FROM ref_platforms p
        LEFT JOIN stg_citations s USING (platform_code)
        GROUP BY ALL
        ORDER BY citation_record_count DESC, p.platform_code
    """
    copy_parquet(connection, platforms_query, curated / "ai_platforms.parquet")

    sources_query = f"""
        SELECT
            s.source_id,
            s.domain_normalized AS domain,
            CASE
                WHEN s.domain_normalized IN ('sm.cn', 'quark.cn')
                    THEN s.domain_normalized
                ELSE mode(s.site_name_raw ORDER BY s.site_name_raw)
            END AS source_display_name,
            CASE
                WHEN max(t.domain) IS NOT NULL THEN max(t.source_category_l1)
                WHEN s.domain_normalized = 'gov.cn' OR ends_with(s.domain_normalized, '.gov.cn')
                    THEN 'government_public'
                ELSE 'unclassified_long_tail'
            END AS source_category_l1,
            CASE
                WHEN max(t.domain) IS NOT NULL THEN max(t.source_category_l1_cn)
                WHEN s.domain_normalized = 'gov.cn' OR ends_with(s.domain_normalized, '.gov.cn')
                    THEN '政府与公共机构'
                ELSE '未分类长尾'
            END AS source_category_l1_cn,
            CASE
                WHEN max(t.domain) IS NOT NULL THEN max(t.source_type)
                WHEN s.domain_normalized = 'gov.cn' OR ends_with(s.domain_normalized, '.gov.cn')
                    THEN 'government_public'
                ELSE 'unclassified'
            END AS source_type,
            CASE
                WHEN max(t.domain) IS NOT NULL THEN max(t.source_type_cn)
                WHEN s.domain_normalized = 'gov.cn' OR ends_with(s.domain_normalized, '.gov.cn')
                    THEN '政府机构'
                ELSE '未分类'
            END AS source_type_cn,
            CASE
                WHEN max(t.domain) IS NOT NULL THEN max(t.ecosystem)
                WHEN s.domain_normalized = 'gov.cn' OR ends_with(s.domain_normalized, '.gov.cn')
                    THEN '政府与公共机构'
                ELSE NULL
            END AS ecosystem,
            CASE
                WHEN max(t.domain) IS NOT NULL THEN max(t.classification_status)
                WHEN s.domain_normalized = 'gov.cn' OR ends_with(s.domain_normalized, '.gov.cn')
                    THEN 'rule_classified'
                ELSE 'pending'
            END AS classification_status,
            CASE
                WHEN max(t.domain) IS NOT NULL THEN max(t.classification_method)
                WHEN s.domain_normalized = 'gov.cn' OR ends_with(s.domain_normalized, '.gov.cn')
                    THEN 'domain_suffix_rule'
                ELSE 'default_unclassified'
            END AS classification_method,
            CASE
                WHEN max(t.domain) IS NOT NULL THEN max(t.classification_confidence)
                WHEN s.domain_normalized = 'gov.cn' OR ends_with(s.domain_normalized, '.gov.cn')
                    THEN 'high'
                ELSE 'low'
            END AS classification_confidence,
            CASE
                WHEN max(t.domain) IS NOT NULL THEN max(t.classification_evidence)
                WHEN s.domain_normalized = 'gov.cn' OR ends_with(s.domain_normalized, '.gov.cn')
                    THEN '域名为 gov.cn 或以 .gov.cn 结尾'
                ELSE '未命中精确域名映射或政府域名确定性规则'
            END AS classification_evidence,
            count(*) AS citation_record_count,
            count(DISTINCT s.record_hash) AS unique_citation_count,
            count(DISTINCT s.page_id) AS page_count,
            count(DISTINCT s.question_id) AS question_count,
            count(DISTINCT s.platform_code) AS platform_count,
            DATE '{release_date}' AS created_from_release
        FROM stg_citations s
        LEFT JOIN ref_source_types t ON s.domain_normalized = t.domain
        WHERE s.source_id IS NOT NULL
        GROUP BY s.source_id, s.domain_normalized
        ORDER BY citation_record_count DESC, domain
    """
    copy_parquet(connection, sources_query, curated / "sources.parquet")

    pages_query = f"""
        SELECT
            page_id,
            mode(source_id ORDER BY source_id) AS source_id,
            canonical_url,
            mode(quote_title ORDER BY quote_title) AS page_title,
            min(published_at_normalized) FILTER (WHERE published_at_normalized IS NOT NULL) AS earliest_published_at,
            min(published_date) AS earliest_published_date,
            CASE
                WHEN count(DISTINCT published_date) = 0 THEN 'unknown'
                WHEN count(DISTINCT published_date) = 1 THEN 'consistent'
                ELSE 'conflicting'
            END AS published_date_status,
            count(DISTINCT published_date) AS published_date_value_count,
            CASE
                WHEN count(DISTINCT published_date) = 1 THEN min(published_date)
                ELSE NULL
            END AS representative_published_date,
            min(published_date) AS observed_published_date_min,
            max(published_date) AS observed_published_date_max,
            mode(site_name_raw ORDER BY site_name_raw) AS source_display_name,
            count(*) AS citation_record_count,
            count(DISTINCT record_hash) AS unique_citation_count,
            count(DISTINCT question_id) AS question_count,
            count(DISTINCT platform_code) AS platform_count,
            max(length(snippet)) AS max_snippet_length,
            DATE '{release_date}' AS created_from_release
        FROM stg_citations
        WHERE page_id IS NOT NULL
        GROUP BY page_id, canonical_url
        ORDER BY citation_record_count DESC, page_id
    """
    copy_parquet(connection, pages_query, curated / "pages.parquet")

    responses_query = """
        SELECT
            CAST(NULL AS VARCHAR) AS response_id,
            CAST(NULL AS VARCHAR) AS question_id,
            CAST(NULL AS VARCHAR) AS platform_code,
            CAST(NULL AS VARCHAR) AS model_name,
            CAST(NULL AS VARCHAR) AS model_version,
            CAST(NULL AS TIMESTAMPTZ) AS collected_at,
            CAST(NULL AS VARCHAR) AS response_text,
            CAST(NULL AS VARCHAR) AS recovery_status
        WHERE false
    """
    copy_parquet(connection, responses_query, curated / "responses.parquet")

    return {
        "citation_observations": connection.execute("SELECT count(*) FROM stg_citations").fetchone()[0],
        "questions": connection.execute(f"SELECT count(*) FROM ({questions_query})").fetchone()[0],
        "question_labels": connection.execute(f"SELECT count(*) FROM ({labels_query})").fetchone()[0],
        "ai_platforms": connection.execute(f"SELECT count(*) FROM ({platforms_query})").fetchone()[0],
        "sources": connection.execute(f"SELECT count(*) FROM ({sources_query})").fetchone()[0],
        "pages": connection.execute(f"SELECT count(*) FROM ({pages_query})").fetchone()[0],
        "responses": 0,
    }


def build_features_and_marts(
    connection: duckdb.DuckDBPyConnection,
    build_root: Path,
    release_date: str,
) -> dict[str, int]:
    features = build_root / "features"
    marts = build_root / "marts"

    page_features_query = f"""
        SELECT
            page_id,
            source_id,
            length(page_title) AS title_length,
            max_snippet_length,
            regexp_matches(coalesce(page_title, ''), '(20[0-9]{{2}})') AS title_contains_year,
            regexp_matches(coalesce(page_title, ''), '(榜|排行|TOP|top|十佳|十大)') AS title_contains_ranking,
            regexp_matches(coalesce(page_title, ''), '(对比|比较|区别|vs|VS)') AS title_contains_comparison,
            regexp_matches(coalesce(page_title, ''), '(指南|攻略|教程|步骤|怎么)') AS title_contains_guide,
            CASE
                WHEN regexp_matches(coalesce(page_title, ''), '(榜|排行|TOP|top|十佳|十大)') THEN 'ranking'
                WHEN regexp_matches(coalesce(page_title, ''), '(对比|比较|区别|vs|VS)') THEN 'comparison'
                WHEN regexp_matches(coalesce(page_title, ''), '(指南|攻略|教程|步骤|怎么)') THEN 'guide'
                WHEN page_title IS NULL THEN 'unknown'
                ELSE 'general'
            END AS content_format_hint,
            'deterministic_v1' AS feature_version,
            DATE '{release_date}' AS created_from_release
        FROM (
            SELECT
                page_id,
                mode(source_id ORDER BY source_id) AS source_id,
                mode(quote_title ORDER BY quote_title) AS page_title,
                max(length(snippet)) AS max_snippet_length
            FROM stg_citations
            WHERE page_id IS NOT NULL
            GROUP BY page_id
        )
        ORDER BY page_id
    """
    copy_parquet(connection, page_features_query, features / "page_features.parquet")

    source_visibility_query = f"""
        SELECT
            s.source_id,
            s.domain_normalized AS domain,
            mode(s.site_name_raw ORDER BY s.site_name_raw) AS source_display_name,
            s.platform_code,
            count(*) AS deduplicated_citation_count,
            count(DISTINCT s.question_id) AS question_count,
            count(DISTINCT s.page_id) AS page_count,
            avg(s.quote_position_normalized) AS average_quote_position,
            count(*) FILTER (WHERE s.quote_position_normalized IS NOT NULL)::BIGINT AS positioned_citation_count,
            DATE '{release_date}' AS release_date
        FROM stg_citations s
        WHERE s.is_preferred_exact_record AND s.source_id IS NOT NULL
        GROUP BY s.source_id, s.domain_normalized, s.platform_code
        ORDER BY deduplicated_citation_count DESC, domain, platform_code
    """
    copy_parquet(connection, source_visibility_query, marts / "source_visibility.parquet")

    connection.execute(
        """
        CREATE TEMP TABLE platform_question_pages AS
        SELECT DISTINCT platform_code, question_id, page_id
        FROM stg_citations
        WHERE is_preferred_exact_record AND page_id IS NOT NULL
        """
    )
    platform_overlap_query = f"""
        WITH platforms AS (
            SELECT DISTINCT platform_code FROM platform_question_pages
        ), pairs AS (
            SELECT a.platform_code AS platform_a, b.platform_code AS platform_b
            FROM platforms a CROSS JOIN platforms b
            WHERE a.platform_code < b.platform_code
        ), platform_sizes AS (
            SELECT platform_code, count(*) AS question_page_count
            FROM platform_question_pages
            GROUP BY platform_code
        ), shared AS (
            SELECT
                a.platform_code AS platform_a,
                b.platform_code AS platform_b,
                count(*) AS shared_question_page_count,
                count(DISTINCT a.question_id) AS shared_question_count
            FROM platform_question_pages a
            INNER JOIN platform_question_pages b
              ON a.question_id = b.question_id
             AND a.page_id = b.page_id
             AND a.platform_code < b.platform_code
            GROUP BY a.platform_code, b.platform_code
        )
        SELECT
            pairs.platform_a,
            pairs.platform_b,
            coalesce(shared.shared_question_page_count, 0) AS shared_question_page_count,
            coalesce(shared.shared_question_count, 0) AS shared_question_count,
            a.question_page_count AS platform_a_question_page_count,
            b.question_page_count AS platform_b_question_page_count,
            coalesce(shared.shared_question_page_count, 0)::DOUBLE /
                nullif(a.question_page_count + b.question_page_count - coalesce(shared.shared_question_page_count, 0), 0)
                AS jaccard_similarity,
            DATE '{release_date}' AS release_date
        FROM pairs
        LEFT JOIN shared USING (platform_a, platform_b)
        INNER JOIN platform_sizes a ON pairs.platform_a = a.platform_code
        INNER JOIN platform_sizes b ON pairs.platform_b = b.platform_code
        ORDER BY jaccard_similarity DESC, platform_a, platform_b
    """
    copy_parquet(connection, platform_overlap_query, marts / "platform_overlap.parquet")

    content_performance_query = f"""
        SELECT
            s.page_id,
            mode(s.canonical_url ORDER BY s.canonical_url) AS canonical_url,
            mode(s.source_id ORDER BY s.source_id) AS source_id,
            mode(s.domain_normalized ORDER BY s.domain_normalized) AS domain,
            mode(s.quote_title ORDER BY s.quote_title) AS page_title,
            count(*) AS deduplicated_citation_count,
            count(DISTINCT s.question_id) AS question_count,
            count(DISTINCT s.platform_code) AS platform_count,
            avg(s.quote_position_normalized) AS average_quote_position,
            count(*) FILTER (WHERE s.quote_position_normalized IS NOT NULL)::BIGINT AS positioned_citation_count,
            DATE '{release_date}' AS release_date
        FROM stg_citations s
        WHERE s.is_preferred_exact_record AND s.page_id IS NOT NULL
        GROUP BY s.page_id
        ORDER BY deduplicated_citation_count DESC, s.page_id
    """
    copy_parquet(connection, content_performance_query, marts / "content_performance.parquet")

    data_quality_query = f"""
        WITH total AS (SELECT count(*)::DOUBLE AS n FROM stg_citations), metrics AS (
            SELECT 'total_records' AS metric, '全部引用观察' AS metric_cn, count(*) AS affected_records FROM stg_citations
            UNION ALL SELECT 'exact_duplicate_extra_records', '精确重复中的额外记录', count(*) - count(DISTINCT record_hash) FROM stg_citations
            UNION ALL SELECT 'url_not_valid_http', 'URL 非有效 HTTP(S)', count_if(url_status <> 'valid_http') FROM stg_citations
            UNION ALL SELECT 'published_at_missing', '发布时间缺失', count_if(published_at_parse_status = 'missing') FROM stg_citations
            UNION ALL SELECT 'published_at_placeholder_zero', '发布时间为零占位符', count_if(published_at_parse_status = 'placeholder_zero') FROM stg_citations
            UNION ALL SELECT 'published_at_unparsed', '发布时间无法解析', count_if(published_at_parse_status = 'unparsed') FROM stg_citations
            UNION ALL SELECT 'quote_index_missing', '引用序号缺失', count_if(quote_index_parse_status = 'missing') FROM stg_citations
            UNION ALL SELECT 'quote_index_unparsed', '引用序号无法解析', count_if(quote_index_parse_status = 'unparsed') FROM stg_citations
            UNION ALL SELECT 'domain_missing', '规范域名缺失', count_if(domain_normalized IS NULL) FROM stg_citations
            UNION ALL SELECT 'title_missing', '引用标题缺失', count_if(quote_title IS NULL) FROM stg_citations
            UNION ALL SELECT 'site_name_missing', '信源名称缺失', count_if(site_name_raw IS NULL) FROM stg_citations
            UNION ALL SELECT 'snippet_missing', '引用摘要缺失', count_if(snippet IS NULL) FROM stg_citations
            UNION ALL SELECT 'legacy_prompt_id_missing', '原始问题 ID 缺失', count_if(legacy_prompt_id IS NULL) FROM stg_citations
        )
        SELECT metric, metric_cn, affected_records::BIGINT AS affected_records,
               affected_records::DOUBLE / total.n AS affected_ratio,
               DATE '{release_date}' AS release_date
        FROM metrics CROSS JOIN total
        ORDER BY metric
    """
    copy_parquet(connection, data_quality_query, marts / "data_quality.parquet")

    return {
        "page_features": connection.execute(f"SELECT count(*) FROM ({page_features_query})").fetchone()[0],
        "source_visibility": connection.execute(f"SELECT count(*) FROM ({source_visibility_query})").fetchone()[0],
        "platform_overlap": connection.execute(f"SELECT count(*) FROM ({platform_overlap_query})").fetchone()[0],
        "content_performance": connection.execute(f"SELECT count(*) FROM ({content_performance_query})").fetchone()[0],
        "data_quality": connection.execute(f"SELECT count(*) FROM ({data_quality_query})").fetchone()[0],
    }


def build_quality_report(
    connection: duckdb.DuckDBPyConnection,
    build_root: Path,
    manifest: dict,
    table_rows: dict[str, int],
    raw_checksums: dict[str, str],
    source_reference_summary: dict[str, int],
) -> dict:
    scalar = lambda query: connection.execute(query).fetchone()[0]
    total_records = scalar("SELECT count(*) FROM stg_citations")
    unique_record_ids = scalar("SELECT count(DISTINCT citation_id) FROM stg_citations")
    unique_record_hashes = scalar("SELECT count(DISTINCT record_hash) FROM stg_citations")
    unmapped_platforms = scalar(
        """
        SELECT count(DISTINCT s.platform_code)
        FROM stg_citations s LEFT JOIN ref_platforms p USING (platform_code)
        WHERE p.platform_code IS NULL
        """
    )
    unmapped_taxonomy_pairs = scalar(
        """
        SELECT count(*) FROM (
            SELECT DISTINCT s.source_layer, s.source_subcat
            FROM stg_citations s
            LEFT JOIN ref_taxonomy t
              ON s.source_layer = t.source_layer AND s.source_subcat = t.source_subcat
            WHERE t.source_layer IS NULL
        )
        """
    )
    prompt_id_conflicts = scalar(
        """
        SELECT count(*) FROM (
            SELECT legacy_prompt_id
            FROM stg_citations
            WHERE legacy_prompt_id IS NOT NULL
            GROUP BY legacy_prompt_id
            HAVING count(DISTINCT prompt_normalized) > 1
        )
        """
    )
    url_status = dict(connection.execute("SELECT url_status, count(*) FROM stg_citations GROUP BY 1 ORDER BY 1").fetchall())
    date_status = dict(
        connection.execute("SELECT published_at_parse_status, count(*) FROM stg_citations GROUP BY 1 ORDER BY 1").fetchall()
    )
    quote_index_status = dict(
        connection.execute("SELECT quote_index_parse_status, count(*) FROM stg_citations GROUP BY 1 ORDER BY 1").fetchall()
    )
    checks = {
        "source_type_reference_passed_preflight": (
            source_reference_summary.get("rows")
            == source_reference_summary.get("unique_domains")
        ),
        "raw_shard_checksums_match_manifest": len(raw_checksums) == sum(len(c["files"]) for c in manifest["categories"]),
        "raw_and_curated_row_counts_match": total_records == manifest["summary"]["records"],
        "citation_ids_are_unique": unique_record_ids == total_records,
        "exact_duplicate_count_matches_manifest": total_records - unique_record_hashes == manifest["summary"]["exact_duplicate_records"],
        "all_platform_codes_are_mapped": unmapped_platforms == 0,
        "all_legacy_category_pairs_are_mapped": unmapped_taxonomy_pairs == 0,
        "each_legacy_prompt_id_maps_to_one_question": prompt_id_conflicts == 0,
        "all_nonempty_publication_values_are_classified": date_status.get("unparsed", 0) == 0,
        "responses_table_is_intentionally_empty": table_rows.get("responses") == 0,
    }
    report = {
        "dataset": manifest["dataset"],
        "dataset_version": manifest["version"],
        "release_date": manifest["release_date"],
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "raw_layer": {
            "verified_shards": len(raw_checksums),
            "records": manifest["summary"]["records"],
            "immutability_check": "all manifest SHA-256 checksums verified before and after build",
        },
        "curated_summary": {
            "citation_observations": total_records,
            "unique_record_ids": unique_record_ids,
            "unique_record_hashes": unique_record_hashes,
            "extra_exact_duplicates": total_records - unique_record_hashes,
            "unmapped_platform_codes": unmapped_platforms,
            "unmapped_taxonomy_pairs": unmapped_taxonomy_pairs,
            "legacy_prompt_id_conflicts": prompt_id_conflicts,
            "source_type_reference": source_reference_summary,
        },
        "parse_status": {
            "url": url_status,
            "published_at": date_status,
            "quote_index": quote_index_status,
        },
        "table_rows": table_rows,
        "checks": checks,
        "known_limitations_cn": [
            "旧数据缺少可靠的回答批次边界，responses 当前保留空表结构，citation_observations.response_id 保持为空。",
            "source_types.csv 保留逐域人工复核证据，政府域名使用确定性后缀规则，其余长尾信源继续标记为未分类。",
            "页面特征来自可复现的确定性规则，当前未生成向量、品牌实体和情感特征。",
        ],
    }
    quality_dir = build_root / "quality" / f"release_date={manifest['release_date']}"
    quality_dir.mkdir(parents=True, exist_ok=True)
    with (quality_dir / "quality_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    lines = [
        f"# 数据质量报告：{manifest['release_date']}",
        "",
        f"- 验收状态：{'通过' if report['status'] == 'passed' else '未通过'}",
        f"- 原始记录：{total_records:,} 条",
        f"- 规范问题：{table_rows['questions']:,} 个",
        f"- 规范信源：{table_rows['sources']:,} 个",
        f"- 规范页面：{table_rows['pages']:,} 个",
        f"- 额外精确重复：{total_records - unique_record_hashes:,} 条",
        f"- 原始分片校验：{len(raw_checksums)} 个全部通过 SHA-256 校验",
        "",
        "## 自动验收",
        "",
    ]
    for check_name, passed in checks.items():
        lines.append(f"- {'通过' if passed else '失败'}：`{check_name}`")
    lines.extend(
        [
            "",
            "## 已知限制",
            "",
            *[f"- {item}" for item in report["known_limitations_cn"]],
            "",
            "完整机器可读结果见同目录 `quality_report.json`。",
        ]
    )
    (quality_dir / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def catalog_table_paths(data_root: Path) -> dict[str, Path]:
    return {
        "questions": data_root / "curated" / "questions.parquet",
        "question_labels": data_root / "curated" / "question_labels.parquet",
        "ai_platforms": data_root / "curated" / "ai_platforms.parquet",
        "sources": data_root / "curated" / "sources.parquet",
        "pages": data_root / "curated" / "pages.parquet",
        "responses": data_root / "curated" / "responses.parquet",
        "citation_observations": data_root / "curated" / "citation_observations" / "**" / "*.parquet",
        "page_features": data_root / "features" / "page_features.parquet",
        "source_visibility": data_root / "marts" / "source_visibility.parquet",
        "platform_overlap": data_root / "marts" / "platform_overlap.parquet",
        "content_performance": data_root / "marts" / "content_performance.parquet",
        "data_quality": data_root / "marts" / "data_quality.parquet",
    }


def materialize_catalog_tables(connection: duckdb.DuckDBPyConnection, data_root: Path) -> None:
    for table_name, path in catalog_table_paths(data_root).items():
        if table_name == "citation_observations":
            continue
        connection.execute(
            f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{sql_literal(path)}', hive_partitioning=true)"
        )


def build_catalog(build_root: Path, release_date: str, dataset_version: str) -> None:
    catalog_path = build_root / "catalog" / "cn_geo.duckdb"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(catalog_path))
    materialize_catalog_tables(connection, build_root)
    connection.execute(
        """
        CREATE TABLE data_dictionary (
            table_name VARCHAR,
            table_name_cn VARCHAR,
            grain_cn VARCHAR,
            recommended_use_cn VARCHAR
        )
        """
    )
    dictionary_rows = [
        ("questions", "问题表", "每个规范问题一行", "问题覆盖、分类与跨平台分析"),
        ("question_labels", "问题多标签表", "每个问题与标签一行", "按行业、意图、风格和场景筛选"),
        ("ai_platforms", "AI 平台表", "每个平台代码一行", "平台中文名称和终端映射"),
        ("sources", "信源表", "每个规范域名一行", "信源覆盖与类型分析"),
        ("pages", "页面表", "每个规范 URL 一行", "页面级内容与表现分析"),
        ("citation_observations", "引用观察事实表（外部 Parquet）", "每条原始引用一行", "全量追溯、清洗状态与自定义分析"),
        ("source_visibility", "信源可见度集市", "每个信源与平台一行", "直接比较各平台的来源覆盖"),
        ("platform_overlap", "平台重合度集市", "每个平台组合一行", "跨平台引用共识分析"),
        ("content_performance", "内容表现集市", "每个页面一行", "寻找高覆盖页面和内容形式"),
    ]
    connection.executemany("INSERT INTO data_dictionary VALUES (?, ?, ?, ?)", dictionary_rows)
    connection.execute(
        "CREATE TABLE warehouse_metadata AS SELECT ? AS dataset_version, ? AS release_date, "
        "? AS pipeline_version, CAST(now() AS TIMESTAMP) AS catalog_built_at",
        [dataset_version, release_date, PIPELINE_VERSION],
    )
    connection.close()


def install_outputs(data_root: Path, build_root: Path, force: bool, validator=None) -> None:
    existing = [name for name in DERIVED_DIRS if (data_root / name).exists()]
    if existing and not force:
        joined = ", ".join(existing)
        raise RuntimeError(f"以下输出目录已存在：{joined}。确认重建时请使用 --force。")

    backup_root = data_root / f".warehouse_backup_{uuid.uuid4().hex}"
    moved_existing: list[str] = []
    installed: list[str] = []
    cleanup_backup = False
    try:
        if existing:
            backup_root.mkdir(parents=True)
            for name in existing:
                (data_root / name).replace(backup_root / name)
                moved_existing.append(name)
        for name in DERIVED_DIRS:
            source = build_root / name
            if not source.exists():
                raise RuntimeError(f"构建输出缺失：{name}")
            source.replace(data_root / name)
            installed.append(name)
        if validator is not None:
            validator()
        cleanup_backup = True
    except Exception as original_error:
        rollback_errors: list[str] = []
        for name in reversed(installed):
            current = data_root / name
            if current.exists():
                try:
                    current.replace(build_root / name)
                except OSError as error:
                    rollback_errors.append(f"无法移回新目录 {name}：{error}")
        for name in reversed(moved_existing):
            backup = backup_root / name
            if backup.exists():
                try:
                    backup.replace(data_root / name)
                except OSError as error:
                    rollback_errors.append(f"无法恢复旧目录 {name}：{error}")
        if rollback_errors:
            details = "；".join(rollback_errors)
            raise RuntimeError(f"派生目录安装失败且回滚未完整完成：{details}。备份保留在 {backup_root}") from original_error
        cleanup_backup = True
        raise
    finally:
        if cleanup_backup and backup_root.exists():
            shutil.rmtree(backup_root)


def verify_installed_catalog(repo_root: Path, expected_table_rows: dict[str, int]) -> None:
    catalog_path = repo_root / "data" / "catalog" / "cn_geo.duckdb"
    connection = duckdb.connect(str(catalog_path))
    try:
        for table_name, expected_rows in expected_table_rows.items():
            if table_name == "citation_observations":
                continue
            actual_rows = connection.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
            if actual_rows != expected_rows:
                raise RuntimeError(
                    f"目录验收失败：{table_name} 期望 {expected_rows} 行，实际 {actual_rows} 行"
                )
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 CN-GEO 标准数据层和分析集市")
    parser.add_argument("--force", action="store_true", help="安全替换已存在的派生输出目录")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="项目根目录，默认使用脚本所在项目",
    )
    return parser.parse_args()


def run_build(args: argparse.Namespace, repo_root: Path, data_root: Path) -> int:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        manifest["release_date"] = date.fromisoformat(str(manifest["release_date"])).isoformat()
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("manifest.json 缺少有效的 ISO 发布日期") from error

    existing = [name for name in DERIVED_DIRS if (data_root / name).exists()]
    if existing and not args.force:
        joined = ", ".join(existing)
        raise RuntimeError(f"以下输出目录已存在：{joined}。确认重建时请使用 --force。")

    print("[1/6] 校验信源分类参考表和原始分片")
    source_reference_summary = validate_source_type_reference(
        data_root / "reference" / "source_types.csv"
    )
    before_checksums = validate_raw_release(repo_root, manifest)
    build_root = data_root / f".warehouse_build_{uuid.uuid4().hex}"
    build_root.mkdir(parents=True)
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET threads = 4")
    register_cleaning_functions(connection)
    try:
        print("[2/6] 读取并规范化原始记录")
        build_staging(connection, repo_root, manifest)
        print("[3/6] 生成标准实体表与引用事实表")
        table_rows = build_curated_tables(connection, repo_root, build_root, manifest["release_date"])
        print("[4/6] 生成确定性特征与分析集市")
        table_rows.update(build_features_and_marts(connection, build_root, manifest["release_date"]))
        print("[5/6] 生成质量报告和 DuckDB 查询目录")
        report = build_quality_report(
            connection,
            build_root,
            manifest,
            table_rows,
            before_checksums,
            source_reference_summary,
        )
        if report["status"] != "passed":
            failed_checks = ", ".join(name for name, passed in report["checks"].items() if not passed)
            raise RuntimeError(f"自动质量验收未通过：{failed_checks}；未安装构建结果")
        build_catalog(build_root, manifest["release_date"], manifest["version"])
        connection.close()

        after_checksums = validate_raw_release(repo_root, manifest)
        if before_checksums != after_checksums:
            raise RuntimeError("构建前后原始分片校验值发生变化")

        print("[6/6] 安装并复核派生数据")
        install_outputs(
            data_root,
            build_root,
            args.force,
            validator=lambda: verify_installed_catalog(repo_root, table_rows),
        )
    except Exception:
        connection.close()
        raise
    finally:
        if build_root.exists():
            shutil.rmtree(build_root)

    print("清洗完成：原始 JSONL 未改动，标准层、特征层、集市和查询目录已生成。")
    return 0


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_root = repo_root / "data"
    if not data_root.is_dir():
        raise RuntimeError(f"缺少数据目录：{data_root}")

    lock_path = data_root / ".warehouse_build.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("已有清洗任务正在运行，请等待它完成后重试") from error

        stale_backups = sorted(data_root.glob(".warehouse_backup_*"))
        if stale_backups:
            paths = ", ".join(str(path) for path in stale_backups)
            raise RuntimeError(f"发现上次异常留下的派生数据备份，请先检查并恢复：{paths}")
        return run_build(args, repo_root, data_root)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"构建失败：{error}", file=sys.stderr)
        raise SystemExit(1)
