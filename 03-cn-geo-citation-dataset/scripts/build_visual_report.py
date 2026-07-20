#!/usr/bin/env python3
"""Build the self-contained CN-GEO visual analysis report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "catalog" / "cn_geo.duckdb"
TEMPLATE_PATH = ROOT / "reports" / "src" / "report_template.html"
RUNTIME_PATH = ROOT / "reports" / "src" / "report_runtime.js"
ECHARTS_PATH = ROOT / "reports" / "vendor" / "echarts-6.0.0.min.js"
DEFAULT_OUTPUT = ROOT / "reports" / "CN-GEO_多维数据分析报告.html"
ECHARTS_VERSION = "6.0.0"
ECHARTS_SHA256 = "baa8dfe7e1d9336b98e8986ba7e20ea15e7cdbea1ef42a59d59478632fa45a1d"
PREFERENCE_ENDPOINTS = (
    ("DB", "豆包", "web", 1, 1),
    ("DOUBA", "豆包", "mobile", 2, 1),
    ("DP", "DeepSeek", "web", 3, 2),
    ("DPA", "DeepSeek", "mobile", 4, 2),
    ("TXYB", "腾讯元宝", "web", 5, 3),
    ("TXYBA", "腾讯元宝", "mobile", 6, 3),
    ("TYQW", "千问", "web", 7, 4),
    ("TYQWA", "千问", "mobile", 8, 4),
)


def prepare_preference_endpoints(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE preference_endpoints (
          platform_code VARCHAR,
          product_family VARCHAR,
          terminal VARCHAR,
          endpoint_order INTEGER,
          family_order INTEGER
        )
        """
    )
    con.executemany(
        "INSERT INTO preference_endpoints VALUES (?, ?, ?, ?, ?)",
        PREFERENCE_ENDPOINTS,
    )


def serializable(value: Any) -> Any:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 12)
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_echarts_text(path: Path | None) -> str:
    source_path = path or ECHARTS_PATH
    if not source_path.exists():
        raise SystemExit(f"ECharts library not found: {source_path}")
    text = source_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != ECHARTS_SHA256:
        raise RuntimeError(
            f"Expected ECharts {ECHARTS_VERSION} ({ECHARTS_SHA256}), received {digest}"
        )
    return text


def rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    cursor = con.execute(sql)
    columns = [item[0] for item in cursor.description]
    return [
        {column: serializable(value) for column, value in zip(columns, record)}
        for record in cursor.fetchall()
    ]


def one(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
    result = rows(con, sql)
    if len(result) != 1:
        raise RuntimeError(f"Expected one row, received {len(result)}")
    return result[0]


def build_payload(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    con.execute("SET threads = 1")
    fact_path = (
        ROOT
        / "data"
        / "curated"
        / "citation_observations"
        / "**"
        / "*.parquet"
    ).as_posix().replace("'", "''")
    con.execute(
        "CREATE OR REPLACE TEMP VIEW citation_observations AS "
        f"SELECT * FROM read_parquet('{fact_path}', hive_partitioning=true)"
    )
    prepare_preference_endpoints(con)
    overview = one(
        con,
        """
        SELECT
          count(*) AS raw_citations,
          count(*) FILTER (WHERE is_preferred_exact_record) AS dedup_citations,
          count(DISTINCT question_id) AS questions,
          count(DISTINCT platform_code) AS platforms,
          (SELECT count(*) FROM sources) AS sources,
          (SELECT count(*) FROM pages) AS pages,
          count(*) FILTER (WHERE url_status = 'valid_http') AS valid_urls,
          count(*) FILTER (WHERE published_date IS NOT NULL) AS parsed_dates,
          count(*) FILTER (WHERE published_date IS NOT NULL) AS publication_metadata_available,
          count(*) FILTER (WHERE snippet IS NOT NULL AND trim(snippet) <> '') AS snippets,
          count(*) FILTER (WHERE quote_position_normalized IS NOT NULL) AS positioned,
          (SELECT count(*) FROM pages WHERE page_title IS NOT NULL) AS titled_pages,
          (SELECT count(*) FROM pages WHERE max_snippet_length IS NOT NULL) AS snippet_pages,
          (SELECT count(*) FROM pages WHERE published_date_status = 'consistent') AS dated_pages,
          (SELECT count(*) FROM pages WHERE published_date_status = 'unknown') AS unknown_date_pages,
          (SELECT count(*) FROM pages WHERE published_date_status = 'conflicting') AS conflicting_date_pages,
          min(release_date) AS release_date
        FROM citation_observations
        """,
    )

    field_availability = rows(
        con,
        """
        WITH base AS (SELECT * FROM citation_observations), totals AS (SELECT count(*) n FROM base)
        SELECT * FROM (
          SELECT 1 sort, '核心标识' availability_group, '问题标识' metric,
                 count(*) FILTER (WHERE question_id IS NOT NULL) available_records,
                 count(*) FILTER (WHERE question_id IS NOT NULL)::DOUBLE / n ratio FROM base, totals GROUP BY n
          UNION ALL SELECT 2, '核心标识', '平台标识', count(*) FILTER (WHERE platform_code IS NOT NULL), count(*) FILTER (WHERE platform_code IS NOT NULL)::DOUBLE / n FROM base, totals GROUP BY n
          UNION ALL SELECT 3, '链接解析', '有效 URL', count(*) FILTER (WHERE url_status = 'valid_http'), count(*) FILTER (WHERE url_status = 'valid_http')::DOUBLE / n FROM base, totals GROUP BY n
          UNION ALL SELECT 4, '内容元数据', '引用标题', count(*) FILTER (WHERE quote_title IS NOT NULL AND trim(quote_title) <> ''), count(*) FILTER (WHERE quote_title IS NOT NULL AND trim(quote_title) <> '')::DOUBLE / n FROM base, totals GROUP BY n
          UNION ALL SELECT 5, '内容元数据', '信源名称', count(*) FILTER (WHERE site_name_raw IS NOT NULL AND trim(site_name_raw) <> ''), count(*) FILTER (WHERE site_name_raw IS NOT NULL AND trim(site_name_raw) <> '')::DOUBLE / n FROM base, totals GROUP BY n
          UNION ALL SELECT 6, '内容元数据', '引用摘要', count(*) FILTER (WHERE snippet IS NOT NULL AND trim(snippet) <> ''), count(*) FILTER (WHERE snippet IS NOT NULL AND trim(snippet) <> '')::DOUBLE / n FROM base, totals GROUP BY n
          UNION ALL SELECT 7, '位置元数据', '引用序号', count(*) FILTER (WHERE quote_position_normalized IS NOT NULL), count(*) FILTER (WHERE quote_position_normalized IS NOT NULL)::DOUBLE / n FROM base, totals GROUP BY n
          UNION ALL SELECT 8, '时间元数据', '发布时间', count(*) FILTER (WHERE published_date IS NOT NULL), count(*) FILTER (WHERE published_date IS NOT NULL)::DOUBLE / n FROM base, totals GROUP BY n
          UNION ALL SELECT 9, '链接解析', '规范域名', count(*) FILTER (WHERE domain_normalized IS NOT NULL), count(*) FILTER (WHERE domain_normalized IS NOT NULL)::DOUBLE / n FROM base, totals GROUP BY n
        ) ORDER BY sort
        """,
    )

    analysis_applicability = rows(
        con,
        """
        WITH base AS (SELECT * FROM citation_observations), totals AS (SELECT count(*) n FROM base)
        SELECT * FROM (
          SELECT 1 sort, '引用观察统计' analysis, count(*) available_records, 1.0 ratio,
                 '问题与平台标识完整' boundary FROM base
          UNION ALL SELECT 2, '页面与链接分析', count(*) FILTER (WHERE url_status = 'valid_http'), count(*) FILTER (WHERE url_status = 'valid_http')::DOUBLE / n, '仅使用有效 HTTP(S) 链接' FROM base, totals GROUP BY n
          UNION ALL SELECT 3, '信源与域名分析', count(*) FILTER (WHERE domain_normalized IS NOT NULL), count(*) FILTER (WHERE domain_normalized IS NOT NULL)::DOUBLE / n, '仅使用可解析规范域名' FROM base, totals GROUP BY n
          UNION ALL SELECT 4, '标题特征分析', count(*) FILTER (WHERE quote_title IS NOT NULL), count(*) FILTER (WHERE quote_title IS NOT NULL)::DOUBLE / n, '标题未提供记录单独标记' FROM base, totals GROUP BY n
          UNION ALL SELECT 5, '摘要内容分析', count(*) FILTER (WHERE snippet IS NOT NULL), count(*) FILTER (WHERE snippet IS NOT NULL)::DOUBLE / n, '摘要未提供记录不进入长度与语义分析' FROM base, totals GROUP BY n
          UNION ALL SELECT 6, '引用位置分析', count(*) FILTER (WHERE quote_position_normalized IS NOT NULL), count(*) FILTER (WHERE quote_position_normalized IS NOT NULL)::DOUBLE / n, '仅表示来源序号，缺少完整回答边界' FROM base, totals GROUP BY n
          UNION ALL SELECT 7, '发布时间分析', count(*) FILTER (WHERE published_date IS NOT NULL), count(*) FILTER (WHERE published_date IS NOT NULL)::DOUBLE / n, '发布时间未知记录保留在其他分析中' FROM base, totals GROUP BY n
        ) ORDER BY sort
        """,
    )

    platform_availability = rows(
        con,
        """
        WITH p AS (
          SELECT platform_code, count(*) n,
            count(*) FILTER (WHERE published_date IS NOT NULL) date_available,
            count(*) FILTER (WHERE snippet IS NOT NULL AND trim(snippet) <> '') snippet_available,
            count(*) FILTER (WHERE quote_position_normalized IS NOT NULL) position_available,
            count(*) FILTER (WHERE url_status = 'valid_http') url_available,
            count(*) FILTER (WHERE domain_normalized IS NOT NULL) domain_available,
            count(*) FILTER (WHERE quote_title IS NOT NULL AND trim(quote_title) <> '') title_available
          FROM citation_observations GROUP BY platform_code
        )
        SELECT p.platform_code, a.platform_name_cn, field, available::DOUBLE / n ratio
        FROM p
        JOIN ai_platforms a USING (platform_code)
        CROSS JOIN LATERAL (
          VALUES ('发布时间', date_available), ('引用摘要', snippet_available),
                 ('引用序号', position_available), ('有效 URL', url_available),
                 ('规范域名', domain_available), ('引用标题', title_available)
        ) AS x(field, available)
        ORDER BY a.citation_record_count DESC, field
        """,
    )

    processing_status = rows(
        con,
        """
        WITH records AS (SELECT count(*)::DOUBLE n FROM citation_observations),
             page_total AS (SELECT count(*)::DOUBLE n FROM pages)
        SELECT * FROM (
          SELECT 1 sort, '重复观察' item, '引用记录' denominator,
                 (SELECT count(*) - count(DISTINCT record_hash) FROM citation_observations) affected_records,
                 (SELECT count(*) - count(DISTINCT record_hash) FROM citation_observations)::DOUBLE / records.n ratio,
                 '汇总分析默认精确去重，原始层继续保留' handling FROM records
          UNION ALL
          SELECT 2, 'URL 无法规范化', '引用记录', count(*) FILTER (WHERE url_status <> 'valid_http'),
                 count(*) FILTER (WHERE url_status <> 'valid_http')::DOUBLE / records.n,
                 '引用事实保留，页面与域名分析排除' FROM citation_observations, records GROUP BY records.n
          UNION ALL
          SELECT 3, '发布时间未提供或占位', '引用记录', count(*) FILTER (WHERE published_at_parse_status IN ('missing','placeholder_zero')),
                 count(*) FILTER (WHERE published_at_parse_status IN ('missing','placeholder_zero'))::DOUBLE / records.n,
                 '非时间分析继续使用，时间分析标记为未知' FROM citation_observations, records GROUP BY records.n
          UNION ALL
          SELECT 4, '页面发布时间冲突', '规范页面', count(*) FILTER (WHERE published_date_status = 'conflicting'),
                 count(*) FILTER (WHERE published_date_status = 'conflicting')::DOUBLE / page_total.n,
                 '不直接选取最早日期，时间分析单独标记' FROM pages, page_total GROUP BY page_total.n
        ) ORDER BY sort
        """,
    )

    platforms = rows(
        con,
        """
        SELECT
          a.platform_code, a.platform_name_cn, a.product_family, a.terminal,
          a.company_ecosystem, a.mapping_status,
          count(*) AS raw_count,
          count(*) FILTER (WHERE c.is_preferred_exact_record) AS dedup_count,
          count(DISTINCT c.question_id) FILTER (WHERE c.is_preferred_exact_record) AS question_count,
          count(DISTINCT c.source_id) FILTER (WHERE c.is_preferred_exact_record AND c.source_id IS NOT NULL) AS source_count,
          count(DISTINCT c.page_id) FILTER (WHERE c.is_preferred_exact_record AND c.page_id IS NOT NULL) AS page_count,
          avg(c.quote_position_normalized) FILTER (WHERE c.is_preferred_exact_record) AS avg_position,
          count(*) FILTER (WHERE c.is_preferred_exact_record AND c.quote_position_normalized IS NOT NULL) AS positioned_count
        FROM ai_platforms a
        JOIN citation_observations c USING (platform_code)
        GROUP BY ALL
        ORDER BY raw_count DESC
        """,
    )

    platform_density = rows(
        con,
        """
        WITH q AS (
          SELECT platform_code, question_id,
                 count(*) raw_count,
                 count(*) FILTER (WHERE is_preferred_exact_record) dedup_count
          FROM citation_observations
          GROUP BY platform_code, question_id
        )
        SELECT q.platform_code, a.platform_name_cn,
               min(raw_count) raw_min, quantile_cont(raw_count, .25) raw_q1,
               median(raw_count) raw_median, quantile_cont(raw_count, .75) raw_q3, max(raw_count) raw_max,
               min(dedup_count) dedup_min, quantile_cont(dedup_count, .25) dedup_q1,
               median(dedup_count) dedup_median, quantile_cont(dedup_count, .75) dedup_q3, max(dedup_count) dedup_max
        FROM q JOIN ai_platforms a USING (platform_code)
        GROUP BY q.platform_code, a.platform_name_cn
        ORDER BY median(raw_count) DESC
        """,
    )

    terminal_pairs = rows(
        con,
        """
        SELECT a.product_family, a.terminal,
               count(*) raw_count,
               count(*) FILTER (WHERE c.is_preferred_exact_record) dedup_count,
               count(DISTINCT c.question_id) FILTER (WHERE c.is_preferred_exact_record) question_count
        FROM citation_observations c JOIN ai_platforms a USING (platform_code)
        WHERE a.product_family IN (
          SELECT product_family FROM ai_platforms GROUP BY product_family HAVING count(DISTINCT terminal) = 2
        )
        GROUP BY a.product_family, a.terminal ORDER BY a.product_family, a.terminal
        """,
    )

    source_types = rows(
        con,
        """
        WITH base AS (
          SELECT c.platform_code, coalesce(s.source_type_cn, '未分类') source_type_cn,
                 c.is_preferred_exact_record, c.source_id, c.page_id, c.question_id
          FROM citation_observations c LEFT JOIN sources s USING (source_id)
        ), expanded AS (
          SELECT coalesce(g.platform_code, 'ALL') platform_code, b.* EXCLUDE(platform_code)
          FROM base b
          JOIN (SELECT platform_code FROM ai_platforms UNION ALL SELECT NULL) g
            ON g.platform_code = b.platform_code OR g.platform_code IS NULL
        )
        SELECT platform_code, source_type_cn,
               count(*) raw_count,
               count(*) FILTER (WHERE is_preferred_exact_record) dedup_count,
               count(DISTINCT source_id) FILTER (WHERE is_preferred_exact_record) source_count,
               count(DISTINCT page_id) FILTER (WHERE is_preferred_exact_record) page_count,
               count(DISTINCT question_id) FILTER (WHERE is_preferred_exact_record) question_count
        FROM expanded GROUP BY platform_code, source_type_cn
        ORDER BY platform_code, raw_count DESC
        """,
    )

    source_filter_rows = rows(
        con,
        """
        WITH base AS (
          SELECT c.platform_code, coalesce(s.source_type_cn, '未分类') source_type_cn,
                 c.source_id, coalesce(s.source_display_name, s.domain, '未知信源') source_name,
                 s.domain, c.is_preferred_exact_record, c.page_id, c.question_id
          FROM citation_observations c JOIN sources s USING (source_id)
        ), expanded AS (
          SELECT coalesce(pg.platform_code, 'ALL') platform_filter,
                 coalesce(tg.source_type_cn, 'ALL') type_filter,
                 b.source_id, b.source_name, b.domain, b.is_preferred_exact_record, b.page_id, b.question_id
          FROM base b
          JOIN (SELECT platform_code FROM ai_platforms UNION ALL SELECT NULL) pg
            ON pg.platform_code = b.platform_code OR pg.platform_code IS NULL
          JOIN (SELECT DISTINCT source_type_cn FROM base UNION ALL SELECT NULL) tg
            ON tg.source_type_cn = b.source_type_cn OR tg.source_type_cn IS NULL
        ), totals AS (
          SELECT platform_filter, type_filter,
                 count(*) filter_raw_total,
                 count(*) FILTER (WHERE is_preferred_exact_record) filter_dedup_total,
                 count(DISTINCT question_id)
                   FILTER (WHERE is_preferred_exact_record) filter_question_count
          FROM expanded
          GROUP BY platform_filter, type_filter
        ), agg AS (
          SELECT platform_filter, type_filter, source_id, source_name, domain,
                 count(*) raw_count,
                 count(*) FILTER (WHERE is_preferred_exact_record) dedup_count,
                 count(DISTINCT page_id) FILTER (WHERE is_preferred_exact_record) page_count,
                 count(DISTINCT question_id) FILTER (WHERE is_preferred_exact_record) question_count
          FROM expanded GROUP BY ALL
        ), ranked AS (
          SELECT *,
                 row_number() OVER (
                   PARTITION BY platform_filter, type_filter
                   ORDER BY raw_count DESC, source_id
                 ) rank_raw,
                 row_number() OVER (
                   PARTITION BY platform_filter, type_filter
                   ORDER BY dedup_count DESC, source_id
                 ) rank_dedup
          FROM agg
        )
        SELECT r.* EXCLUDE(rank_raw, rank_dedup),
               t.filter_raw_total, t.filter_dedup_total, t.filter_question_count
        FROM ranked r
        JOIN totals t USING (platform_filter, type_filter)
        WHERE rank_raw <= 32 OR rank_dedup <= 32
        ORDER BY platform_filter, type_filter, raw_count DESC
        """,
    )

    source_pareto = rows(
        con,
        """
        WITH ranked AS (
          SELECT row_number() OVER (ORDER BY unique_citation_count DESC) rank,
                 coalesce(source_display_name, domain) source_name,
                 unique_citation_count dedup_count,
                 sum(unique_citation_count) OVER (ORDER BY unique_citation_count DESC ROWS UNBOUNDED PRECEDING)::DOUBLE /
                 sum(unique_citation_count) OVER () cumulative_share
          FROM sources
        )
        SELECT * FROM ranked WHERE rank <= 80 ORDER BY rank
        """,
    )

    source_platform = rows(
        con,
        """
        WITH top_sources AS (
          SELECT source_id FROM sources ORDER BY unique_citation_count DESC LIMIT 16
        )
        SELECT v.source_id, coalesce(s.source_display_name, s.domain) source_name,
               v.platform_code, a.platform_name_cn, v.deduplicated_citation_count citation_count
        FROM source_visibility v
        JOIN top_sources t USING (source_id)
        JOIN sources s USING (source_id)
        JOIN ai_platforms a USING (platform_code)
        ORDER BY s.unique_citation_count DESC, a.citation_record_count DESC
        """,
    )

    ecosystem = rows(
        con,
        """
        SELECT coalesce(ecosystem, '未归属生态') ecosystem,
               count(*) source_count,
               sum(unique_citation_count) dedup_count,
               sum(citation_record_count) raw_count,
               sum(page_count) page_count
        FROM sources GROUP BY coalesce(ecosystem, '未归属生态')
        ORDER BY dedup_count DESC
        """,
    )

    ecosystem_unclassified_summary = one(
        con,
        """
        WITH ranked AS (
          SELECT unique_citation_count,
                 row_number() OVER (ORDER BY unique_citation_count DESC, source_id) rank
          FROM sources WHERE ecosystem IS NULL
        )
        SELECT count(*) source_count,
               sum(unique_citation_count) dedup_count,
               sum(unique_citation_count) FILTER (WHERE rank <= 50) top50_dedup_count,
               count(*) FILTER (WHERE unique_citation_count <= 5) low_frequency_source_count,
               sum(unique_citation_count) FILTER (WHERE unique_citation_count <= 5) low_frequency_dedup_count
        FROM ranked
        """,
    )

    ecosystem_unclassified_top = rows(
        con,
        """
        SELECT coalesce(source_display_name, domain, '未知信源') source_name,
               domain, unique_citation_count dedup_count,
               page_count, question_count
        FROM sources
        WHERE ecosystem IS NULL
        ORDER BY unique_citation_count DESC, source_id
        LIMIT 10
        """,
    )

    source_preference = rows(
        con,
        """
        WITH selected AS (
          SELECT platform_code, endpoint_order FROM preference_endpoints
        ), balanced_questions AS (
          SELECT question_id
          FROM citation_observations
          WHERE is_preferred_exact_record
            AND platform_code IN (SELECT platform_code FROM selected)
          GROUP BY question_id
          HAVING count(DISTINCT platform_code)
                   FILTER (WHERE source_id IS NOT NULL)
                 = (SELECT count(*) FROM selected)
        ), source_question AS (
          SELECT o.platform_code, o.question_id, o.source_id,
                 count(*) citation_count
          FROM citation_observations o
          JOIN balanced_questions b USING (question_id)
          WHERE o.is_preferred_exact_record
            AND o.source_id IS NOT NULL
            AND o.platform_code IN (SELECT platform_code FROM selected)
          GROUP BY o.platform_code, o.question_id, o.source_id
        ), question_totals AS (
          SELECT platform_code, question_id, sum(citation_count) total_citations
          FROM source_question
          GROUP BY platform_code, question_id
        ), source_platform AS (
          SELECT sq.source_id, sq.platform_code,
                 sum(sq.citation_count) citation_count,
                 count(DISTINCT sq.question_id) question_count,
                 sum(sq.citation_count::DOUBLE / qt.total_citations)
                   / (SELECT count(*) FROM balanced_questions) weighted_share
          FROM source_question sq
          JOIN question_totals qt USING (platform_code, question_id)
          GROUP BY sq.source_id, sq.platform_code
        ), qualified AS (
          SELECT source_id, sum(citation_count) total_citations,
                 count(DISTINCT question_id) source_question_count
          FROM source_question
          GROUP BY source_id
          HAVING sum(citation_count) >= 20
             AND count(DISTINCT question_id) >= 5
        ), source_grid AS (
          SELECT q.source_id, q.total_citations, q.source_question_count,
                 s.platform_code, s.endpoint_order,
                 coalesce(sp.citation_count, 0) citation_count,
                 coalesce(sp.question_count, 0) question_count,
                 coalesce(sp.weighted_share, 0) weighted_share
          FROM qualified q
          CROSS JOIN selected s
          LEFT JOIN source_platform sp USING (source_id, platform_code)
        ), indexed AS (
          SELECT *, avg(weighted_share) OVER (PARTITION BY source_id) baseline_share
          FROM source_grid
        ), scored AS (
          SELECT *, 100 * weighted_share / nullif(baseline_share, 0) preference_index
          FROM indexed
        ), source_scores AS (
          SELECT source_id,
                 (max(preference_index) - min(preference_index))
                   * ln(1 + max(source_question_count)) distinctiveness_score
          FROM scored
          GROUP BY source_id
        ), ranked AS (
          SELECT source_id, distinctiveness_score,
                 row_number() OVER (
                   ORDER BY distinctiveness_score DESC, source_id
                 ) source_rank
          FROM source_scores
        )
        SELECT r.source_rank, s.source_id,
               coalesce(src.source_display_name, src.domain, '未知信源') source_name,
               src.domain, coalesce(src.source_type_cn, '未分类') source_type_cn,
               s.platform_code, a.platform_name_cn, a.product_family, a.terminal,
               sel.endpoint_order, s.citation_count, s.question_count,
               s.total_citations, s.source_question_count,
               s.weighted_share, s.preference_index, r.distinctiveness_score,
               (SELECT count(*) FROM balanced_questions) balanced_question_count,
               (SELECT count(*) FROM qualified) qualified_source_count
        FROM scored s
        JOIN ranked r USING (source_id)
        JOIN preference_endpoints sel USING (platform_code)
        JOIN ai_platforms a USING (platform_code)
        JOIN sources src USING (source_id)
        WHERE r.source_rank <= 20
        ORDER BY r.source_rank, sel.endpoint_order
        """,
    )
    if not source_preference:
        raise RuntimeError("Source preference analysis is empty")
    selected_codes = {item[0] for item in PREFERENCE_ENDPOINTS}
    inferred_endpoints = [
        item["platform_name_cn"]
        for item in platforms
        if item["platform_code"] in selected_codes
        and item["mapping_status"] == "inferred"
    ]
    preference_meta = {
        "balanced_question_count": source_preference[0].pop("balanced_question_count"),
        "qualified_source_count": source_preference[0].pop("qualified_source_count"),
        "endpoint_count": len(PREFERENCE_ENDPOINTS),
        "product_count": len({item[1] for item in PREFERENCE_ENDPOINTS}),
        "minimum_citations": 20,
        "minimum_questions": 5,
        "matrix_source_count": len({item["source_id"] for item in source_preference}),
        "inferred_endpoints": inferred_endpoints,
    }
    for item in source_preference[1:]:
        item.pop("balanced_question_count")
        item.pop("qualified_source_count")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE preference_source_rank_all AS
        WITH selected AS (
          SELECT platform_code, endpoint_order FROM preference_endpoints
        ), common_questions AS (
          SELECT question_id
          FROM citation_observations
          WHERE is_preferred_exact_record
            AND platform_code IN (SELECT platform_code FROM selected)
          GROUP BY question_id
          HAVING count(DISTINCT platform_code)
                   FILTER (WHERE source_id IS NOT NULL)
                 = (SELECT count(*) FROM selected)
        ), scope_questions AS (
          SELECT 'common' scope_name, s.platform_code, q.question_id
          FROM common_questions q
          CROSS JOIN selected s
          UNION ALL
          SELECT 'full' scope_name, o.platform_code, o.question_id
          FROM citation_observations o
          JOIN selected s USING (platform_code)
          WHERE o.is_preferred_exact_record AND o.source_id IS NOT NULL
          GROUP BY o.platform_code, o.question_id
        ), source_question AS (
          SELECT q.scope_name, o.platform_code, o.question_id, o.source_id,
                 count(*) citation_count
          FROM scope_questions q
          JOIN citation_observations o
            ON o.platform_code = q.platform_code
           AND o.question_id = q.question_id
          WHERE o.is_preferred_exact_record AND o.source_id IS NOT NULL
          GROUP BY q.scope_name, o.platform_code, o.question_id, o.source_id
        ), question_totals AS (
          SELECT scope_name, platform_code, question_id,
                 sum(citation_count) total_citations
          FROM source_question
          GROUP BY scope_name, platform_code, question_id
        ), scope_counts AS (
          SELECT scope_name, platform_code, count(*) scope_question_count
          FROM scope_questions
          GROUP BY scope_name, platform_code
        ), source_shares AS (
          SELECT sq.scope_name, sq.platform_code, sq.source_id,
                 sum(sq.citation_count) citation_count,
                 count(DISTINCT sq.question_id) question_count,
                 round(
                   sum(sq.citation_count::DOUBLE / qt.total_citations)
                     / sc.scope_question_count,
                   12
                 ) weighted_share,
                 sc.scope_question_count
          FROM source_question sq
          JOIN question_totals qt USING (scope_name, platform_code, question_id)
          JOIN scope_counts sc USING (scope_name, platform_code)
          GROUP BY sq.scope_name, sq.platform_code, sq.source_id,
                   sc.scope_question_count
        )
        SELECT ss.*,
               row_number() OVER (
                 PARTITION BY scope_name, platform_code
                 ORDER BY weighted_share DESC, citation_count DESC, source_id
               ) rank
        FROM source_shares ss
        """
    )

    source_top_ranks = rows(
        con,
        """
        SELECT r.scope_name AS "scope", r.platform_code endpoint, r.platform_code,
               a.platform_name_cn platform, a.platform_name_cn,
               a.product_family, a.terminal, e.endpoint_order,
               r.source_id,
               coalesce(s.source_display_name, s.domain, '未知信源') source_name,
               s.domain,
               coalesce(s.source_category_l1_cn, '未分类长尾') source_category_l1_cn,
               coalesce(s.source_type_cn, '未分类') source_type_cn,
               r.rank, r.weighted_share AS "share", r.citation_count, r.question_count,
               r.scope_question_count
        FROM preference_source_rank_all r
        JOIN preference_endpoints e USING (platform_code)
        JOIN ai_platforms a USING (platform_code)
        JOIN sources s USING (source_id)
        WHERE r.rank <= 20
        ORDER BY CASE r.scope_name WHEN 'common' THEN 1 ELSE 2 END,
                 e.endpoint_order, r.rank
        """,
    )
    preference_meta["common_scope_question_count"] = next(
        item["scope_question_count"]
        for item in source_top_ranks
        if item["scope"] == "common"
    )
    preference_meta["full_scope_question_counts"] = [
        {
            "platform_code": platform_code,
            "question_count": next(
                item["scope_question_count"]
                for item in source_top_ranks
                if item["scope"] == "full"
                and item["platform_code"] == platform_code
            ),
        }
        for platform_code, *_ in PREFERENCE_ENDPOINTS
    ]

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE preference_anchor_sources AS
        WITH anchor_pool AS (
          SELECT DISTINCT source_id
          FROM preference_source_rank_all
          WHERE scope_name = 'common'
            AND platform_code IN ('DB', 'DOUBA', 'DP', 'DPA')
            AND rank <= 10
        ), anchor_endpoints(platform_code) AS (
          VALUES ('DB'), ('DOUBA'), ('DP'), ('DPA')
        ), anchor_grid AS (
          SELECT p.source_id, e.platform_code
          FROM anchor_pool p
          CROSS JOIN anchor_endpoints e
        )
        SELECT g.source_id,
               count(*) FILTER (WHERE r.rank <= 10) anchor_top10_occurrences,
               avg(coalesce(r.weighted_share, 0)) anchor_average_share
        FROM anchor_grid g
        LEFT JOIN preference_source_rank_all r
          ON r.scope_name = 'common'
         AND r.source_id = g.source_id
         AND r.platform_code = g.platform_code
        GROUP BY g.source_id
        """
    )
    anchor_source_migration = rows(
        con,
        """
        WITH common_scope AS (
          SELECT max(scope_question_count) scope_question_count
          FROM preference_source_rank_all
          WHERE scope_name = 'common'
        ), anchor_ranked AS (
          SELECT *, row_number() OVER (
            ORDER BY anchor_top10_occurrences DESC,
                     anchor_average_share DESC, source_id
          ) anchor_order
          FROM preference_anchor_sources
        ), anchor_grid AS (
          SELECT a.*, e.platform_code, e.endpoint_order
          FROM anchor_ranked a
          CROSS JOIN preference_endpoints e
        )
        SELECT 'common' AS "scope", g.anchor_order,
               g.source_id,
               coalesce(s.source_display_name, s.domain, '未知信源') source_name,
               s.domain,
               coalesce(s.source_category_l1_cn, '未分类长尾') source_category_l1_cn,
               coalesce(s.source_type_cn, '未分类') source_type_cn,
               g.anchor_top10_occurrences, g.anchor_average_share,
               g.platform_code endpoint, g.platform_code,
               p.platform_name_cn platform, p.platform_name_cn,
               p.product_family, p.terminal, g.endpoint_order,
               g.platform_code IN ('DB', 'DOUBA', 'DP', 'DPA') is_anchor_endpoint,
               r.rank,
               coalesce(r.weighted_share, 0) AS "share",
               coalesce(r.citation_count, 0) citation_count,
               coalesce(r.question_count, 0) question_count,
               coalesce(r.scope_question_count, c.scope_question_count) scope_question_count
        FROM anchor_grid g
        CROSS JOIN common_scope c
        JOIN sources s USING (source_id)
        JOIN ai_platforms p USING (platform_code)
        LEFT JOIN preference_source_rank_all r
          ON r.scope_name = 'common'
         AND r.source_id = g.source_id
         AND r.platform_code = g.platform_code
        ORDER BY g.anchor_order, g.endpoint_order
        """,
    )
    preference_meta["anchor_pool_size"] = len(
        {item["source_id"] for item in anchor_source_migration}
    )
    preference_meta["anchor_top20_carryover"] = rows(
        con,
        """
        WITH counts AS (
          SELECT e.platform_code, e.endpoint_order,
                 count(*) FILTER (WHERE r.rank <= 20) source_count
          FROM preference_endpoints e
          CROSS JOIN preference_anchor_sources a
          LEFT JOIN preference_source_rank_all r
            ON r.scope_name = 'common'
           AND r.platform_code = e.platform_code
           AND r.source_id = a.source_id
          GROUP BY e.platform_code, e.endpoint_order
        )
        SELECT platform_code, source_count
        FROM counts
        ORDER BY endpoint_order
        """,
    )

    con.execute(
        "CREATE OR REPLACE TEMP TABLE preference_matrix_sources (source_id VARCHAR)"
    )
    con.executemany(
        "INSERT INTO preference_matrix_sources VALUES (?)",
        [(source_id,) for source_id in sorted({item["source_id"] for item in source_preference})],
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE preference_top20_link_sources AS
        SELECT DISTINCT e.product_family, r.source_id
        FROM preference_source_rank_all r
        JOIN preference_endpoints e USING (platform_code)
        WHERE r.rank <= 20
        """
    )

    terminal_pair_summary = rows(
        con,
        """
        WITH selected AS (
          SELECT platform_code, product_family, terminal, family_order
          FROM preference_endpoints
        ), observations AS (
          SELECT s.product_family, s.terminal, s.family_order,
                 o.question_id, o.source_id
          FROM citation_observations o
          JOIN selected s USING (platform_code)
          WHERE o.is_preferred_exact_record
        ), pair_questions AS (
          SELECT product_family, question_id
          FROM observations
          GROUP BY product_family, question_id
          HAVING count(DISTINCT terminal)
                   FILTER (WHERE source_id IS NOT NULL) = 2
        ), presence AS (
          SELECT o.product_family, min(o.family_order) family_order, o.source_id,
                 bool_or(o.terminal = 'web') has_web,
                 bool_or(o.terminal = 'mobile') has_mobile,
                 count(*) citation_count,
                 count(DISTINCT o.question_id) source_question_count
          FROM observations o
          JOIN pair_questions q USING (product_family, question_id)
          WHERE o.source_id IS NOT NULL
          GROUP BY o.product_family, o.source_id
        ), summary AS (
          SELECT product_family, min(family_order) family_order,
                 count(*) FILTER (WHERE has_web) web_sources,
                 count(*) FILTER (WHERE has_mobile) mobile_sources,
                 count(*) FILTER (WHERE has_web AND has_mobile) shared_sources,
                 count(*) FILTER (WHERE has_web AND NOT has_mobile) web_only_sources,
                 count(*) FILTER (WHERE has_mobile AND NOT has_web) mobile_only_sources,
                 count(*) FILTER (
                   WHERE has_web AND citation_count >= 20
                     AND source_question_count >= 5
                 ) qualified_web_sources,
                 count(*) FILTER (
                   WHERE has_mobile AND citation_count >= 20
                     AND source_question_count >= 5
                 ) qualified_mobile_sources,
                 count(*) FILTER (
                   WHERE has_web AND has_mobile AND citation_count >= 20
                     AND source_question_count >= 5
                 ) qualified_shared_sources
          FROM presence
          GROUP BY product_family
        )
        SELECT s.*,
               (SELECT count(*) FROM pair_questions q
                WHERE q.product_family = s.product_family) common_question_count,
               shared_sources::DOUBLE
                 / nullif(web_sources + mobile_sources - shared_sources, 0) source_jaccard,
               qualified_shared_sources::DOUBLE
                 / nullif(qualified_web_sources + qualified_mobile_sources
                          - qualified_shared_sources, 0) qualified_source_jaccard
        FROM summary s
        ORDER BY family_order
        """,
    )

    terminal_tilt = rows(
        con,
        """
        WITH selected AS (
          SELECT platform_code, product_family, terminal
          FROM preference_endpoints
        ), observations AS (
          SELECT s.product_family, s.terminal, o.question_id, o.source_id
          FROM citation_observations o
          JOIN selected s USING (platform_code)
          WHERE o.is_preferred_exact_record
        ), pair_questions AS (
          SELECT product_family, question_id
          FROM observations
          GROUP BY product_family, question_id
          HAVING count(DISTINCT terminal)
                   FILTER (WHERE source_id IS NOT NULL) = 2
        ), source_question AS (
          SELECT o.product_family, o.terminal, o.question_id, o.source_id,
                 count(*) citation_count
          FROM observations o
          JOIN pair_questions p USING (product_family, question_id)
          WHERE o.source_id IS NOT NULL
          GROUP BY o.product_family, o.terminal, o.question_id, o.source_id
        ), question_totals AS (
          SELECT product_family, terminal, question_id,
                 sum(citation_count) total_citations
          FROM source_question
          GROUP BY product_family, terminal, question_id
        ), source_terminal AS (
          SELECT sq.product_family, sq.terminal, sq.source_id,
                 sum(sq.citation_count) citation_count,
                 count(DISTINCT sq.question_id) question_count,
                 sum(sq.citation_count::DOUBLE / qt.total_citations)
                   / (SELECT count(*) FROM pair_questions p
                      WHERE p.product_family = sq.product_family) weighted_share
          FROM source_question sq
          JOIN question_totals qt USING (product_family, terminal, question_id)
          GROUP BY sq.product_family, sq.terminal, sq.source_id
        ), observed_candidates AS (
          SELECT product_family, source_id
          FROM source_question
          GROUP BY product_family, source_id
          HAVING (sum(citation_count) >= 20
              AND count(DISTINCT question_id) >= 5)
              OR source_id IN (SELECT source_id FROM preference_matrix_sources)
        ), candidate_sources AS (
          SELECT product_family, source_id FROM observed_candidates
          UNION
          SELECT product_family, source_id FROM preference_top20_link_sources
        ), qualified AS (
          SELECT c.product_family, c.source_id,
                 coalesce(sum(sq.citation_count), 0) total_citations,
                 count(DISTINCT sq.question_id) source_question_count
          FROM candidate_sources c
          LEFT JOIN source_question sq USING (product_family, source_id)
          GROUP BY c.product_family, c.source_id
        ), aggregate AS (
          SELECT q.product_family, q.source_id, q.total_citations,
                 q.source_question_count,
                 coalesce(max(st.citation_count) FILTER (WHERE st.terminal = 'web'), 0) web_citations,
                 coalesce(max(st.citation_count) FILTER (WHERE st.terminal = 'mobile'), 0) mobile_citations,
                 coalesce(max(st.question_count) FILTER (WHERE st.terminal = 'web'), 0) web_questions,
                 coalesce(max(st.question_count) FILTER (WHERE st.terminal = 'mobile'), 0) mobile_questions,
                 coalesce(max(st.weighted_share) FILTER (WHERE st.terminal = 'web'), 0) web_share,
                 coalesce(max(st.weighted_share) FILTER (WHERE st.terminal = 'mobile'), 0) mobile_share,
                 (SELECT count(*) FROM pair_questions p
                  WHERE p.product_family = q.product_family) common_question_count
          FROM qualified q
          LEFT JOIN source_terminal st USING (product_family, source_id)
          GROUP BY q.product_family, q.source_id, q.total_citations,
                   q.source_question_count
        ), scored AS (
          SELECT *, (mobile_share - web_share) * 100 delta_pp,
                 200 * greatest(web_share, mobile_share)
                   / nullif(web_share + mobile_share, 0) terminal_preference_index,
                 (greatest(web_share, mobile_share) + abs(mobile_share - web_share))
                   * ln(1 + source_question_count) priority_score
          FROM aggregate
        ), ranked AS (
          SELECT *,
                 row_number() OVER (
                   PARTITION BY product_family
                   ORDER BY priority_score DESC, total_citations DESC, source_id
                 ) priority_rank,
                 row_number() OVER (
                   PARTITION BY product_family
                   ORDER BY abs(delta_pp) DESC, source_question_count DESC, source_id
                 ) tilt_rank
          FROM scored
        )
        SELECT r.*, coalesce(s.source_display_name, s.domain, '未知信源') source_name,
               s.domain, coalesce(s.source_type_cn, '未分类') source_type_cn
        FROM ranked r
        JOIN sources s USING (source_id)
        WHERE priority_rank <= 40 OR tilt_rank <= 40
           OR r.source_id IN (SELECT source_id FROM preference_matrix_sources)
           OR EXISTS (
             SELECT 1 FROM preference_top20_link_sources t
             WHERE t.product_family = r.product_family
               AND t.source_id = r.source_id
           )
        ORDER BY product_family, least(priority_rank, tilt_rank), source_id
        """,
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE preference_typed_question AS
        WITH selected AS (
          SELECT platform_code, endpoint_order FROM preference_endpoints
        ), balanced_questions AS (
          SELECT question_id
          FROM citation_observations
          WHERE is_preferred_exact_record
            AND platform_code IN (SELECT platform_code FROM selected)
          GROUP BY question_id
          HAVING count(DISTINCT platform_code)
                   FILTER (WHERE source_id IS NOT NULL)
                 = (SELECT count(*) FROM selected)
        )
        SELECT o.platform_code, o.question_id,
                 CASE
                   WHEN o.source_id IS NULL THEN '信源未规范化'
                   ELSE coalesce(s.source_category_l1_cn, '未分类长尾')
                 END source_category_l1_cn,
                 CASE
                   WHEN o.source_id IS NULL THEN '信源未规范化'
                   ELSE coalesce(s.source_type_cn, '未分类')
                 END source_type_cn,
                 CASE
                   WHEN o.source_id IS NULL THEN 'unnormalized'
                   WHEN coalesce(s.source_category_l1_cn, '未分类长尾') = '未分类长尾'
                     THEN 'unclassified'
                   WHEN s.classification_method IN ('deterministic_rule', 'domain_suffix_rule')
                     THEN 'rule'
                   ELSE 'manual'
                 END classification_bucket,
                 count(*) citation_count
          FROM citation_observations o
          JOIN balanced_questions b USING (question_id)
          LEFT JOIN sources s USING (source_id)
          WHERE o.is_preferred_exact_record
            AND o.platform_code IN (SELECT platform_code FROM selected)
          GROUP BY ALL
        """
    )

    preference_type_detail = rows(
        con,
        """
        WITH question_totals AS (
          SELECT platform_code, question_id, sum(citation_count) total_citations
          FROM preference_typed_question
          GROUP BY platform_code, question_id
        )
        SELECT t.platform_code, a.platform_name_cn, a.product_family, a.terminal,
               sel.endpoint_order, t.source_category_l1_cn, t.source_type_cn,
               sum(t.citation_count) citation_count,
               count(DISTINCT t.question_id) question_count,
               sum(t.citation_count::DOUBLE / q.total_citations)
                 / 334 weighted_share
        FROM preference_typed_question t
        JOIN question_totals q USING (platform_code, question_id)
        JOIN ai_platforms a USING (platform_code)
        JOIN preference_endpoints sel USING (platform_code)
        GROUP BY ALL
        ORDER BY sel.endpoint_order, t.source_category_l1_cn,
                 weighted_share DESC, t.source_type_cn
        """,
    )

    preference_type_mix = rows(
        con,
        """
        WITH question_totals AS (
          SELECT platform_code, question_id, sum(citation_count) total_citations
          FROM preference_typed_question
          GROUP BY platform_code, question_id
        )
        SELECT t.platform_code, a.platform_name_cn, a.product_family, a.terminal,
               e.endpoint_order, t.source_category_l1_cn,
               sum(t.citation_count) citation_count,
               count(DISTINCT t.question_id) question_count,
               sum(t.citation_count::DOUBLE / q.total_citations) / 334 weighted_share
        FROM preference_typed_question t
        JOIN question_totals q USING (platform_code, question_id)
        JOIN ai_platforms a USING (platform_code)
        JOIN preference_endpoints e USING (platform_code)
        GROUP BY ALL
        ORDER BY e.endpoint_order, weighted_share DESC
        """,
    )

    type_detail_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in preference_type_detail:
        key = (item["platform_code"], item["source_category_l1_cn"])
        type_detail_lookup.setdefault(key, []).append(
            {
                "source_type_cn": item["source_type_cn"],
                "citation_count": item["citation_count"],
                "question_count": item["question_count"],
                "weighted_share": item["weighted_share"],
            }
        )
    for item in preference_type_mix:
        item["source_type_breakdown"] = type_detail_lookup[
            (item["platform_code"], item["source_category_l1_cn"])
        ]

    classification_bucket_rows = rows(
        con,
        """
        WITH bucket_question AS (
          SELECT platform_code, question_id, classification_bucket,
                 sum(citation_count) citation_count
          FROM preference_typed_question
          GROUP BY platform_code, question_id, classification_bucket
        ), question_totals AS (
          SELECT platform_code, question_id, sum(citation_count) total_citations
          FROM bucket_question
          GROUP BY platform_code, question_id
        )
        SELECT b.platform_code, e.endpoint_order, b.classification_bucket,
               sum(b.citation_count) citation_count,
               sum(b.citation_count::DOUBLE / q.total_citations) / 334 weighted_share
        FROM bucket_question b
        JOIN question_totals q USING (platform_code, question_id)
        JOIN preference_endpoints e USING (platform_code)
        GROUP BY b.platform_code, e.endpoint_order, b.classification_bucket
        ORDER BY e.endpoint_order, b.classification_bucket
        """,
    )
    classification_coverage: list[dict[str, Any]] = []
    coverage_by_endpoint: dict[str, dict[str, dict[str, float | int]]] = {}
    for item in classification_bucket_rows:
        coverage_by_endpoint.setdefault(item["platform_code"], {})[
            item["classification_bucket"]
        ] = {
            "share": item["weighted_share"],
            "citation_count": item["citation_count"],
        }
    endpoint_labels = {
        item[0]: {"product_family": item[1], "terminal": item[2], "endpoint_order": item[3]}
        for item in PREFERENCE_ENDPOINTS
    }
    bucket_names = ("manual", "rule", "unclassified", "unnormalized")
    for platform_code, label in endpoint_labels.items():
        buckets = coverage_by_endpoint[platform_code]
        shares = {name: float(buckets.get(name, {}).get("share", 0)) for name in bucket_names}
        counts = {
            name: int(buckets.get(name, {}).get("citation_count", 0))
            for name in bucket_names
        }
        recognizable_share = shares["manual"] + shares["rule"] + shares["unclassified"]
        classification_coverage.append(
            {
                "platform_code": platform_code,
                **label,
                **{f"{name}_share": shares[name] for name in bucket_names},
                "classification_coverage": (shares["manual"] + shares["rule"])
                / recognizable_share,
                "recognizable_citation_count": counts["manual"]
                + counts["rule"]
                + counts["unclassified"],
                "classified_citation_count": counts["manual"] + counts["rule"],
                "total_citation_count": sum(counts.values()),
            }
        )
    classification_coverage.append(
        {
            "platform_code": "ALL",
            "product_family": "八端总体",
            "terminal": "all",
            "endpoint_order": 9,
            **{
                f"{name}_share": sum(row[f"{name}_share"] for row in classification_coverage)
                / len(classification_coverage)
                for name in bucket_names
            },
            "recognizable_citation_count": sum(
                row["recognizable_citation_count"] for row in classification_coverage
            ),
            "classified_citation_count": sum(
                row["classified_citation_count"] for row in classification_coverage
            ),
            "total_citation_count": sum(
                row["total_citation_count"] for row in classification_coverage
            ),
        }
    )
    overall_coverage = classification_coverage[-1]
    overall_coverage["classification_coverage"] = (
        overall_coverage["manual_share"] + overall_coverage["rule_share"]
    ) / (1 - overall_coverage["unnormalized_share"])

    overlap = rows(
        con,
        """
        SELECT p.platform_a, aa.platform_name_cn platform_a_name,
               p.platform_b, ab.platform_name_cn platform_b_name,
               p.shared_question_page_count, p.shared_question_count,
               p.platform_a_question_page_count, p.platform_b_question_page_count,
               p.jaccard_similarity
        FROM platform_overlap p
        JOIN ai_platforms aa ON aa.platform_code = p.platform_a
        JOIN ai_platforms ab ON ab.platform_code = p.platform_b
        ORDER BY p.jaccard_similarity DESC
        """,
    )

    shared_unique = rows(
        con,
        """
        WITH qp AS (
          SELECT DISTINCT platform_code, question_id, page_id
          FROM citation_observations
          WHERE is_preferred_exact_record AND page_id IS NOT NULL
        ), freq AS (
          SELECT question_id, page_id, count(*) platform_frequency FROM qp GROUP BY question_id, page_id
        )
        SELECT q.platform_code, a.platform_name_cn,
               count(*) FILTER (WHERE f.platform_frequency = 1) unique_pairs,
               count(*) FILTER (WHERE f.platform_frequency > 1) shared_pairs,
               count(*) total_pairs
        FROM qp q JOIN freq f USING (question_id, page_id)
        JOIN ai_platforms a USING (platform_code)
        GROUP BY q.platform_code, a.platform_name_cn
        ORDER BY shared_pairs::DOUBLE / total_pairs DESC
        """,
    )

    consensus_sources = rows(
        con,
        """
        SELECT coalesce(source_display_name, domain) source_name, domain, source_type_cn,
               platform_count, question_count, page_count,
               unique_citation_count dedup_count,
               question_count * platform_count consensus_score
        FROM sources
        WHERE platform_count >= 2
        ORDER BY consensus_score DESC, unique_citation_count DESC
        LIMIT 24
        """,
    )

    label_stats = rows(
        con,
        """
        WITH base AS (
          SELECT ql.label_dimension, ql.label_value, ql.label_cn,
                 c.question_id, c.platform_code, c.source_id, c.page_id,
                 c.is_preferred_exact_record
          FROM question_labels ql
          JOIN citation_observations c USING (question_id)
        ), overall AS (
          SELECT label_dimension, label_value, label_cn,
                 count(DISTINCT question_id) question_count,
                 count(*) raw_count,
                 count(*) FILTER (WHERE is_preferred_exact_record) dedup_count,
                 count(DISTINCT source_id)
                   FILTER (WHERE is_preferred_exact_record) source_count,
                 count(DISTINCT page_id)
                   FILTER (WHERE is_preferred_exact_record) page_count,
                 count(DISTINCT platform_code)
                   FILTER (WHERE is_preferred_exact_record) platform_count
          FROM base
          GROUP BY label_dimension, label_value, label_cn
        ), per_question AS (
          SELECT label_dimension, label_value, label_cn, question_id,
                 count(*) raw_citations,
                 count(*) FILTER (WHERE is_preferred_exact_record) dedup_citations,
                 count(DISTINCT source_id)
                   FILTER (WHERE is_preferred_exact_record) sources,
                 count(DISTINCT page_id)
                   FILTER (WHERE is_preferred_exact_record) pages,
                 count(DISTINCT platform_code)
                   FILTER (WHERE is_preferred_exact_record) platforms
          FROM base
          GROUP BY label_dimension, label_value, label_cn, question_id
        ), rates AS (
          SELECT label_dimension, label_value, label_cn,
                 avg(raw_citations) avg_raw_citations_per_question,
                 avg(dedup_citations) avg_dedup_citations_per_question,
                 avg(sources) avg_sources_per_question,
                 avg(pages) avg_pages_per_question,
                 avg(platforms) avg_platforms_per_question
          FROM per_question
          GROUP BY label_dimension, label_value, label_cn
        )
        SELECT o.*, r.* EXCLUDE(label_dimension, label_value, label_cn)
        FROM overall o
        JOIN rates r USING (label_dimension, label_value, label_cn)
        ORDER BY o.label_dimension, o.raw_count DESC
        """,
    )

    label_platform = rows(
        con,
        """
        SELECT ql.label_dimension, ql.label_value, ql.label_cn,
               c.platform_code, a.platform_name_cn,
               count(*) raw_count,
               count(*) FILTER (WHERE c.is_preferred_exact_record) dedup_count,
               count(DISTINCT c.source_id) FILTER (WHERE c.is_preferred_exact_record) source_count,
               count(DISTINCT c.page_id) FILTER (WHERE c.is_preferred_exact_record) page_count
        FROM question_labels ql JOIN citation_observations c USING (question_id)
        JOIN ai_platforms a USING (platform_code)
        GROUP BY ALL
        """,
    )

    label_top_sources = rows(
        con,
        """
        WITH agg AS (
          SELECT ql.label_dimension, ql.label_value, ql.label_cn,
                 coalesce(s.source_display_name, s.domain) source_name,
                 count(*) FILTER (WHERE c.is_preferred_exact_record) dedup_count,
                 count(DISTINCT c.question_id) FILTER (WHERE c.is_preferred_exact_record) question_count
          FROM question_labels ql JOIN citation_observations c USING (question_id)
          JOIN sources s USING (source_id)
          GROUP BY ALL
        ), ranked AS (
          SELECT *, row_number() OVER (
            PARTITION BY label_dimension, label_value ORDER BY dedup_count DESC, source_name
          ) rank
          FROM agg
        )
        SELECT * FROM ranked WHERE rank <= 8 ORDER BY label_dimension, label_value, rank
        """,
    )

    feature_prevalence = rows(
        con,
        """
        WITH t AS (SELECT count(*) FILTER (WHERE title_length IS NOT NULL) n FROM page_features)
        SELECT * FROM (
          SELECT '标题含年份' feature, count(*) FILTER (WHERE title_contains_year AND title_length IS NOT NULL)::DOUBLE / n ratio, count(*) FILTER (WHERE title_contains_year AND title_length IS NOT NULL) page_count FROM page_features, t GROUP BY n
          UNION ALL SELECT '榜单/排名信号', count(*) FILTER (WHERE title_contains_ranking AND title_length IS NOT NULL)::DOUBLE / n, count(*) FILTER (WHERE title_contains_ranking AND title_length IS NOT NULL) FROM page_features, t GROUP BY n
          UNION ALL SELECT '指南信号', count(*) FILTER (WHERE title_contains_guide AND title_length IS NOT NULL)::DOUBLE / n, count(*) FILTER (WHERE title_contains_guide AND title_length IS NOT NULL) FROM page_features, t GROUP BY n
          UNION ALL SELECT '对比信号', count(*) FILTER (WHERE title_contains_comparison AND title_length IS NOT NULL)::DOUBLE / n, count(*) FILTER (WHERE title_contains_comparison AND title_length IS NOT NULL) FROM page_features, t GROUP BY n
        ) ORDER BY page_count DESC
        """,
    )

    format_platform = rows(
        con,
        """
        WITH base AS (
          SELECT c.platform_code, pf.content_format_hint,
                 c.page_id, c.question_id, c.is_preferred_exact_record
          FROM citation_observations c JOIN page_features pf USING (page_id)
        ), expanded AS (
          SELECT coalesce(g.platform_code, 'ALL') platform_code, b.* EXCLUDE(platform_code)
          FROM base b JOIN (SELECT platform_code FROM ai_platforms UNION ALL SELECT NULL) g
            ON g.platform_code = b.platform_code OR g.platform_code IS NULL
        )
        SELECT platform_code, content_format_hint,
               count(*) raw_count,
               count(*) FILTER (WHERE is_preferred_exact_record) dedup_count,
               count(DISTINCT page_id) FILTER (WHERE is_preferred_exact_record) page_count,
               count(DISTINCT question_id) FILTER (WHERE is_preferred_exact_record) question_count
        FROM expanded GROUP BY platform_code, content_format_hint
        """,
    )

    length_performance = rows(
        con,
        """
        SELECT '标题长度' metric,
          CASE WHEN pf.title_length IS NULL THEN '标题未提供'
               WHEN pf.title_length <= 20 THEN '≤20'
               WHEN pf.title_length <= 30 THEN '21 至 30'
               WHEN pf.title_length <= 40 THEN '31 至 40'
               WHEN pf.title_length <= 50 THEN '41 至 50'
               WHEN pf.title_length <= 60 THEN '51 至 60' ELSE '61+' END band,
          count(*) page_count, avg(cp.deduplicated_citation_count) avg_citations,
          median(cp.deduplicated_citation_count) median_citations,
          avg(cp.question_count) avg_questions
        FROM page_features pf JOIN content_performance cp USING (page_id)
        GROUP BY band
        UNION ALL
        SELECT '摘要长度',
          CASE WHEN pf.max_snippet_length IS NULL THEN '摘要未提供'
               WHEN pf.max_snippet_length = 0 THEN '0'
               WHEN pf.max_snippet_length <= 50 THEN '1 至 50'
               WHEN pf.max_snippet_length <= 100 THEN '51 至 100'
               WHEN pf.max_snippet_length <= 200 THEN '101 至 200'
               WHEN pf.max_snippet_length <= 300 THEN '201 至 300' ELSE '301+' END,
          count(*), avg(cp.deduplicated_citation_count), median(cp.deduplicated_citation_count), avg(cp.question_count)
        FROM page_features pf JOIN content_performance cp USING (page_id)
        GROUP BY 2
        """,
    )

    feature_platform = rows(
        con,
        """
        WITH base AS (
          SELECT DISTINCT c.platform_code, c.page_id, pf.title_length,
                 pf.title_contains_year, pf.title_contains_ranking,
                 pf.title_contains_comparison, pf.title_contains_guide
          FROM citation_observations c JOIN page_features pf USING (page_id)
          WHERE c.is_preferred_exact_record
        ), totals AS (
          SELECT platform_code, count(*) FILTER (WHERE title_length IS NOT NULL) total_pages FROM base GROUP BY platform_code
        ), long AS (
          SELECT platform_code, page_id, feature, present
          FROM base
          CROSS JOIN LATERAL (VALUES
            ('标题含年份', title_contains_year), ('榜单/排名信号', title_contains_ranking),
            ('对比信号', title_contains_comparison), ('指南信号', title_contains_guide)
          ) AS f(feature, present)
          WHERE title_length IS NOT NULL
        )
        SELECT l.platform_code, a.platform_name_cn, l.feature,
               count(*) FILTER (WHERE present)::DOUBLE / t.total_pages ratio,
               count(*) FILTER (WHERE present) page_count
        FROM long l JOIN totals t USING (platform_code) JOIN ai_platforms a USING (platform_code)
        GROUP BY l.platform_code, a.platform_name_cn, l.feature, t.total_pages
        """,
    )

    top_pages = rows(
        con,
        """
        SELECT cp.page_id, cp.page_title, cp.domain, cp.canonical_url,
               s.source_type_cn, cp.deduplicated_citation_count, cp.question_count,
               cp.platform_count, cp.average_quote_position,
               pf.content_format_hint, pf.title_contains_year,
               pf.title_contains_ranking, pf.title_contains_comparison, pf.title_contains_guide
        FROM content_performance cp
        JOIN page_features pf USING (page_id)
        LEFT JOIN sources s ON cp.source_id = s.source_id
        ORDER BY cp.question_count * cp.platform_count DESC, cp.deduplicated_citation_count DESC
        LIMIT 28
        """,
    )

    feature_combinations = rows(
        con,
        """
        SELECT concat(
          CASE WHEN pf.title_contains_year THEN '年份+' ELSE '' END,
          CASE WHEN pf.title_contains_ranking THEN '榜单+' ELSE '' END,
          CASE WHEN pf.title_contains_comparison THEN '对比+' ELSE '' END,
          CASE WHEN pf.title_contains_guide THEN '指南+' ELSE '' END
        ) combination_raw,
        count(*) page_count,
        avg(cp.deduplicated_citation_count) avg_citations,
        avg(cp.question_count) avg_questions,
        avg(cp.platform_count) avg_platforms
        FROM page_features pf JOIN content_performance cp USING (page_id)
        WHERE pf.title_length IS NOT NULL
        GROUP BY combination_raw
        HAVING count(*) >= 40
        ORDER BY avg_questions DESC
        LIMIT 16
        """,
    )
    for item in feature_combinations:
        item["combination"] = item["combination_raw"].rstrip("+") or "无显式信号"

    year_distribution = rows(
        con,
        """
        SELECT extract(year FROM representative_published_date)::INTEGER publication_year,
               count(*) page_count, sum(unique_citation_count) dedup_count
        FROM pages
        WHERE published_date_status = 'consistent'
        GROUP BY publication_year ORDER BY publication_year
        """,
    )

    freshness = rows(
        con,
        """
        WITH p AS (
          SELECT p.page_id, p.source_id, cp.deduplicated_citation_count,
            CASE WHEN p.published_date_status = 'unknown' THEN '发布时间未知'
                 WHEN p.published_date_status = 'conflicting' THEN '发布时间冲突'
                 WHEN extract(year FROM p.representative_published_date) = 2026 THEN '2026'
                 WHEN extract(year FROM p.representative_published_date) = 2025 THEN '2025'
                 WHEN extract(year FROM p.representative_published_date) BETWEEN 2023 AND 2024 THEN '2023 至 2024'
                 ELSE '2022 及以前' END freshness_band
          FROM pages p JOIN content_performance cp USING (page_id)
        )
        SELECT freshness_band, count(*) page_count, sum(deduplicated_citation_count) dedup_count
        FROM p GROUP BY freshness_band
        """,
    )

    label_freshness = rows(
        con,
        """
        WITH qp AS (
          SELECT DISTINCT question_id, page_id FROM citation_observations
          WHERE is_preferred_exact_record AND page_id IS NOT NULL
        ), base AS (
          SELECT ql.label_dimension, ql.label_value, ql.label_cn,
            CASE WHEN p.published_date_status = 'unknown' THEN '发布时间未知'
                 WHEN p.published_date_status = 'conflicting' THEN '发布时间冲突'
                 WHEN extract(year FROM p.representative_published_date) = 2026 THEN '2026'
                 WHEN extract(year FROM p.representative_published_date) = 2025 THEN '2025'
                 WHEN extract(year FROM p.representative_published_date) BETWEEN 2023 AND 2024 THEN '2023 至 2024'
                 ELSE '2022 及以前' END freshness_band,
            qp.page_id
          FROM qp JOIN question_labels ql USING (question_id) JOIN pages p USING (page_id)
        )
        SELECT label_dimension, label_value, label_cn, freshness_band, count(*) page_links
        FROM base GROUP BY ALL
        """,
    )

    source_type_freshness = rows(
        con,
        """
        SELECT coalesce(s.source_type_cn, '未分类') source_type_cn,
          CASE WHEN p.published_date_status = 'unknown' THEN '发布时间未知'
               WHEN p.published_date_status = 'conflicting' THEN '发布时间冲突'
               WHEN extract(year FROM p.representative_published_date) = 2026 THEN '2026'
               WHEN extract(year FROM p.representative_published_date) = 2025 THEN '2025'
               WHEN extract(year FROM p.representative_published_date) BETWEEN 2023 AND 2024 THEN '2023 至 2024'
               ELSE '2022 及以前' END freshness_band,
          count(*) page_count, sum(cp.deduplicated_citation_count) dedup_count
        FROM pages p JOIN content_performance cp USING (page_id)
        LEFT JOIN sources s ON p.source_id = s.source_id
        GROUP BY source_type_cn, freshness_band
        """,
    )

    title_year_quality = rows(
        con,
        """
        WITH y AS (
          SELECT pf.page_id, p.published_date_status, p.representative_published_date,
                 try_cast(regexp_extract(p.page_title, '(19|20)[0-9]{2}', 0) AS INTEGER) title_year
          FROM page_features pf JOIN pages p USING (page_id)
          WHERE pf.title_contains_year
        )
        SELECT CASE
          WHEN title_year IS NULL THEN '识别失败'
          WHEN published_date_status = 'unknown' THEN '发布时间未知'
          WHEN published_date_status = 'conflicting' THEN '发布时间冲突'
          WHEN title_year = extract(year FROM representative_published_date) THEN '年份一致'
          ELSE '年份不一致' END status,
          count(*) page_count
        FROM y GROUP BY status ORDER BY page_count DESC
        """,
    )

    source_quadrant = rows(
        con,
        """
        SELECT coalesce(source_display_name, domain) source_name, domain, source_type_cn,
               question_count, platform_count, page_count,
               unique_citation_count dedup_count,
               unique_citation_count::DOUBLE / nullif(page_count, 0) citations_per_page
        FROM sources
        WHERE question_count >= 3
        ORDER BY unique_citation_count DESC
        LIMIT 320
        """,
    )

    whitespace_sources = rows(
        con,
        """
        WITH candidates AS (
          SELECT a.platform_code, a.platform_name_cn,
                 s.source_id, coalesce(s.source_display_name, s.domain) source_name,
                 s.source_type_cn, s.platform_count, s.question_count, s.unique_citation_count,
                 row_number() OVER (
                   PARTITION BY a.platform_code
                   ORDER BY s.question_count * (12 - s.platform_count) DESC, s.unique_citation_count DESC
                 ) rank
          FROM ai_platforms a CROSS JOIN sources s
          WHERE s.platform_count BETWEEN 3 AND 10
            AND s.question_count >= 8
            AND NOT EXISTS (
              SELECT 1 FROM source_visibility v
              WHERE v.platform_code = a.platform_code AND v.source_id = s.source_id
            )
        )
        SELECT * FROM candidates WHERE rank <= 8 ORDER BY platform_code, rank
        """,
    )

    expansion_candidates = rows(
        con,
        """
        SELECT coalesce(source_display_name, domain) source_name, domain, source_type_cn,
               platform_count, question_count, unique_citation_count dedup_count,
               (12 - platform_count) potential_platforms,
               question_count * (12 - platform_count) screening_score
        FROM sources
        WHERE platform_count BETWEEN 2 AND 9 AND question_count >= 10
        ORDER BY screening_score DESC, unique_citation_count DESC
        LIMIT 28
        """,
    )

    dictionary = rows(
        con,
        """
        SELECT * FROM data_dictionary ORDER BY table_name
        """,
    )

    dimensions = {
        "industry": "行业维度",
        "prompt_style": "提示风格",
        "query_intent": "提问属性",
        "real_world_scene": "极端与真实场景",
        "time_sensitivity": "时间敏感度",
        "trigger_intensity": "触发强度",
        "legacy_status": "历史状态",
    }

    return {
        "meta": {
            "title": "国内生成式 AI 引用生态全景报告",
            "subtitle": "基于 214,119 条引用观察的 12 平台、9,878 个信源与 107,659 个页面分析",
            "generated_at": dt.datetime.now().astimezone().isoformat(timespec="minutes"),
            "release_date": overview["release_date"],
            "default_scope": "dedup",
            "dimension_names": dimensions,
        },
        "overview": overview,
        "fieldAvailability": field_availability,
        "analysisApplicability": analysis_applicability,
        "platformAvailability": platform_availability,
        "processingStatus": processing_status,
        "platforms": platforms,
        "platformDensity": platform_density,
        "terminalPairs": terminal_pairs,
        "sourceTypes": source_types,
        "sourceFilterRows": source_filter_rows,
        "sourcePareto": source_pareto,
        "sourcePlatform": source_platform,
        "ecosystem": ecosystem,
        "ecosystemUnclassifiedSummary": ecosystem_unclassified_summary,
        "ecosystemUnclassifiedTop": ecosystem_unclassified_top,
        "preferenceMeta": preference_meta,
        "sourcePreference": source_preference,
        "sourceTopRanks": source_top_ranks,
        "anchorSourceMigration": anchor_source_migration,
        "terminalPairSummary": terminal_pair_summary,
        "terminalTilt": terminal_tilt,
        "preferenceTypeMix": preference_type_mix,
        "classificationCoverage": classification_coverage,
        "overlap": overlap,
        "sharedUnique": shared_unique,
        "consensusSources": consensus_sources,
        "labelStats": label_stats,
        "labelPlatform": label_platform,
        "labelTopSources": label_top_sources,
        "featurePrevalence": feature_prevalence,
        "formatPlatform": format_platform,
        "lengthPerformance": length_performance,
        "featurePlatform": feature_platform,
        "topPages": top_pages,
        "featureCombinations": feature_combinations,
        "yearDistribution": year_distribution,
        "freshness": freshness,
        "labelFreshness": label_freshness,
        "sourceTypeFreshness": source_type_freshness,
        "titleYearQuality": title_year_quality,
        "sourceQuadrant": source_quadrant,
        "whitespaceSources": whitespace_sources,
        "expansionCandidates": expansion_candidates,
        "dictionary": dictionary,
    }


def validate_payload(payload: dict[str, Any]) -> None:
    overview = payload["overview"]
    expected = {
        "raw_citations": 214_119,
        "dedup_citations": 189_845,
        "questions": 620,
        "platforms": 12,
        "sources": 9_878,
        "pages": 107_659,
    }
    mismatches = {
        key: (overview.get(key), value)
        for key, value in expected.items()
        if overview.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Dataset contract mismatch: {mismatches}")
    if len(payload["platforms"]) != 12:
        raise RuntimeError("Expected 12 platform rows")
    if not payload["overlap"]:
        raise RuntimeError("Platform overlap data is empty")
    if not payload["ecosystemUnclassifiedTop"]:
        raise RuntimeError("Unclassified ecosystem analysis is empty")
    if not payload["labelStats"]:
        raise RuntimeError("Question taxonomy data is empty")
    global_source_rows = [
        item
        for item in payload["sourceFilterRows"]
        if item["platform_filter"] == "ALL" and item["type_filter"] == "ALL"
    ]
    if not global_source_rows:
        raise RuntimeError("Global source filter rows are empty")
    source_raw_totals = {item["filter_raw_total"] for item in global_source_rows}
    source_dedup_totals = {item["filter_dedup_total"] for item in global_source_rows}
    source_question_totals = {
        item["filter_question_count"] for item in global_source_rows
    }
    if len(source_raw_totals) != 1 or len(source_dedup_totals) != 1:
        raise RuntimeError("Source filter citation denominators are inconsistent")
    if source_question_totals != {overview["questions"]}:
        raise RuntimeError("Global source filter question denominator is inconsistent")
    if any(
        item["question_count"] > item["filter_question_count"]
        for item in payload["sourceFilterRows"]
    ):
        raise RuntimeError("Source question coverage exceeds its filter denominator")
    for item in payload["labelStats"]:
        if not math.isclose(
            item["avg_raw_citations_per_question"],
            item["raw_count"] / item["question_count"],
            abs_tol=1e-9,
        ) or not math.isclose(
            item["avg_dedup_citations_per_question"],
            item["dedup_count"] / item["question_count"],
            abs_tol=1e-9,
        ):
            raise RuntimeError("Label citation rates do not match their question denominator")
        if item["avg_platforms_per_question"] > overview["platforms"]:
            raise RuntimeError("Label platform average exceeds the platform universe")
    preference_meta = payload["preferenceMeta"]
    if preference_meta["balanced_question_count"] != 334:
        raise RuntimeError("Expected 334 source-eligible preference questions")
    if preference_meta["endpoint_count"] != 8 or preference_meta["product_count"] != 4:
        raise RuntimeError("Preference analysis must contain four paired products")
    if preference_meta["qualified_source_count"] != 379:
        raise RuntimeError("Expected 379 sources above the preference screening threshold")
    if preference_meta["matrix_source_count"] != 20:
        raise RuntimeError("Preference matrix must contain 20 sources")
    expected_preference_codes = {item[0] for item in PREFERENCE_ENDPOINTS}
    preference_codes = {
        item["platform_code"] for item in payload["sourcePreference"]
    }
    if preference_codes != expected_preference_codes:
        raise RuntimeError("Preference analysis platform scope is inconsistent")
    if not payload["terminalTilt"] or not payload["preferenceTypeMix"]:
        raise RuntimeError("Preference detail data is empty")
    for key in ("preferenceTypeMix",):
        codes = {item["platform_code"] for item in payload[key]}
        if codes != expected_preference_codes:
            raise RuntimeError(f"{key} platform scope is inconsistent")
    source_top_ranks = payload["sourceTopRanks"]
    if len(source_top_ranks) != 8 * 20 * 2:
        raise RuntimeError("Source rankings must contain two Top 20 lists for eight endpoints")
    for scope_name in ("common", "full"):
        for platform_code in expected_preference_codes:
            endpoint_ranks = [
                item
                for item in source_top_ranks
                if item["scope"] == scope_name
                and item["platform_code"] == platform_code
            ]
            if [item["rank"] for item in endpoint_ranks] != list(range(1, 21)):
                raise RuntimeError(
                    f"Source ranking is incomplete for {scope_name}/{platform_code}"
                )
            shares = [item["share"] for item in endpoint_ranks]
            if shares != sorted(shares, reverse=True):
                raise RuntimeError(
                    f"Source ranking shares are not descending for {scope_name}/{platform_code}"
                )
    if preference_meta["common_scope_question_count"] != 334:
        raise RuntimeError("Common source ranking scope must contain 334 questions")
    anchor_codes = {"DB", "DOUBA", "DP", "DPA"}
    expected_anchor_sources = {
        item["source_id"]
        for item in source_top_ranks
        if item["scope"] == "common"
        and item["platform_code"] in anchor_codes
        and item["rank"] <= 10
    }
    anchor_rows = payload["anchorSourceMigration"]
    anchor_sources = {item["source_id"] for item in anchor_rows}
    if anchor_sources != expected_anchor_sources or len(anchor_sources) != 17:
        raise RuntimeError("Anchor pool must equal the 17-source core Top 10 union")
    if len(anchor_rows) != len(anchor_sources) * len(PREFERENCE_ENDPOINTS):
        raise RuntimeError("Every anchor source must project to all eight endpoints")
    if preference_meta["anchor_pool_size"] != len(anchor_sources):
        raise RuntimeError("Anchor pool metadata is inconsistent")
    product_families = {item[1] for item in PREFERENCE_ENDPOINTS}
    if {item["product_family"] for item in payload["terminalPairSummary"]} != product_families:
        raise RuntimeError("Terminal pair summary product scope is inconsistent")
    if {item["product_family"] for item in payload["terminalTilt"]} != product_families:
        raise RuntimeError("Terminal tilt product scope is inconsistent")
    type_shares: dict[str, float] = {}
    for item in payload["preferenceTypeMix"]:
        code = item["platform_code"]
        type_shares[code] = type_shares.get(code, 0.0) + item["weighted_share"]
    if any(not math.isclose(value, 1.0, abs_tol=1e-9) for value in type_shares.values()):
        raise RuntimeError("Preference type shares must sum to one per endpoint")
    expected_primary_categories = {
        "平台与社区",
        "新闻与媒体",
        "垂直专业内容",
        "商业信息与服务",
        "研究与文档",
        "政府与公共机构",
        "品牌与企业官网",
        "搜索与页面代理",
        "未分类长尾",
    }
    primary_categories = {
        item["source_category_l1_cn"]
        for item in payload["preferenceTypeMix"]
        if item["source_category_l1_cn"] != "信源未规范化"
    }
    if primary_categories != expected_primary_categories:
        raise RuntimeError("Preference type mix primary categories are inconsistent")
    coverage = payload["classificationCoverage"]
    if len(coverage) != 9 or coverage[-1]["platform_code"] != "ALL":
        raise RuntimeError("Classification coverage must contain eight endpoints and a summary")
    linked_pairs = {
        (item["source_id"], item["product_family"])
        for item in payload["terminalTilt"]
    }
    missing_links = {
        (item["source_id"], item["product_family"])
        for item in payload["sourcePreference"]
        if item["weighted_share"] > 0
        and (item["source_id"], item["product_family"]) not in linked_pairs
    }
    if missing_links:
        raise RuntimeError(f"Preference matrix links lack terminal detail: {missing_links}")
    top_twenty_links = {
        (item["source_id"], item["product_family"])
        for item in payload["sourceTopRanks"]
    }
    missing_top_twenty_links = top_twenty_links - linked_pairs
    if missing_top_twenty_links:
        raise RuntimeError(
            f"Top 20 source links lack terminal detail: {missing_top_twenty_links}"
        )
    freshness_counts = {
        item["freshness_band"]: item["page_count"] for item in payload["freshness"]
    }
    if sum(freshness_counts.values()) != overview["pages"]:
        raise RuntimeError("Freshness bands do not partition all pages")
    if freshness_counts.get("发布时间未知") != overview["unknown_date_pages"]:
        raise RuntimeError("Unknown publication-date page count is inconsistent")
    if freshness_counts.get("发布时间冲突") != overview["conflicting_date_pages"]:
        raise RuntimeError("Conflicting publication-date page count is inconsistent")
    applicability = {
        item["analysis"]: item for item in payload["analysisApplicability"]
    }
    if applicability["引用观察统计"]["available_records"] != overview["raw_citations"]:
        raise RuntimeError("Observation analysis must retain every raw record")
    for metric_name in ("标题长度", "摘要长度"):
        metric_total = sum(
            item["page_count"]
            for item in payload["lengthPerformance"]
            if item["metric"] == metric_name
        )
        if metric_total != overview["pages"]:
            raise RuntimeError(f"{metric_name} bands do not partition all pages")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--echarts",
        type=Path,
        help="ECharts 6.0.0 library. Omit to use reports/vendor/echarts-6.0.0.min.js.",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}")
    if not TEMPLATE_PATH.exists():
        raise SystemExit(f"Template not found: {TEMPLATE_PATH}")
    if not RUNTIME_PATH.exists():
        raise SystemExit(f"Runtime not found: {RUNTIME_PATH}")
    echarts_text = resolve_echarts_text(args.echarts)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    payload = build_payload(con)
    validate_payload(payload)
    con.close()
    payload["meta"]["build_fingerprints"] = {
        "builder_sha256": file_sha256(Path(__file__)),
        "template_sha256": file_sha256(TEMPLATE_PATH),
        "runtime_sha256": file_sha256(RUNTIME_PATH),
        "echarts_sha256": file_sha256(args.echarts or ECHARTS_PATH),
    }
    payload["meta"]["echarts_version"] = ECHARTS_VERSION

    report_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    report_json = report_json.replace("</", "<\\/")
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("/*__ECHARTS_LIBRARY__*/", echarts_text)
    html = html.replace("/*__REPORT_DATA__*/", report_json)
    html = html.replace("//__REPORT_RUNTIME__", RUNTIME_PATH.read_text(encoding="utf-8"))

    if "__REPORT_DATA__" in html or "__ECHARTS_LIBRARY__" in html or "__REPORT_RUNTIME__" in html:
        raise RuntimeError("Unresolved build placeholder")
    if args.check:
        print(json.dumps({
            "status": "ok",
            "payload_sections": len(payload),
            "template_bytes": TEMPLATE_PATH.stat().st_size,
            "estimated_output_bytes": len(html.encode("utf-8")),
        }, ensure_ascii=False, indent=2))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(json.dumps({
        "status": "built",
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "raw_citations": payload["overview"]["raw_citations"],
        "dedup_citations": payload["overview"]["dedup_citations"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
