from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import scripts.build_data_warehouse as warehouse_builder  # noqa: E402

from scripts.build_data_warehouse import (  # noqa: E402
    canonicalize_url,
    classify_published_at,
    classify_url,
    DERIVED_DIRS,
    install_outputs,
    normalize_published_at,
    validate_raw_release,
)


CONTROLLED_SOURCE_TYPES = {
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
    "brand_corporate": {
        "brand_corporate": "品牌与企业官网",
    },
    "search_page_proxy": {
        "search_proxy": "搜索与页面代理",
        "map_page_proxy": "地图与页面代理",
        "content_aggregation_proxy": "内容聚合与页面代理",
    },
    "unclassified_long_tail": {
        "unclassified": "未分类",
    },
}
CONTROLLED_SOURCE_CATEGORIES = {
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
REFERENCE_GOVERNANCE_COMBINATIONS = {
    ("legacy_curated", "curated", "high"),
    ("manual_review", "reviewed", "high"),
    ("manual_review", "reviewed", "medium"),
    ("manual_review", "reviewed_unclassified", "low"),
}


class SourceClassificationReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mapping_path = REPO_ROOT / "data" / "reference" / "source_types.csv"
        with mapping_path.open(encoding="utf-8", newline="") as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_mapping_has_unique_normalized_domains(self) -> None:
        domains = [row["domain"].strip().lower() for row in self.rows]
        self.assertEqual(len(domains), len(set(domains)))
        self.assertTrue(all(domain and domain == domain.rstrip(".") for domain in domains))

    def test_mapping_uses_allowed_level_one_categories(self) -> None:
        self.assertGreaterEqual(len(self.rows), 500)
        for row in self.rows:
            with self.subTest(domain=row["domain"]):
                self.assertIn(row["source_category_l1"], CONTROLLED_SOURCE_CATEGORIES)
                if row["source_category_l1"] in CONTROLLED_SOURCE_CATEGORIES:
                    self.assertEqual(
                        row["source_category_l1_cn"],
                        CONTROLLED_SOURCE_CATEGORIES[row["source_category_l1"]],
                    )

    def test_mapping_has_valid_governance_metadata(self) -> None:
        required_fields = {
            "classification_method",
            "classification_confidence",
            "classification_evidence",
            "classification_status",
        }
        self.assertTrue(required_fields <= set(self.rows[0]))
        if not required_fields <= set(self.rows[0]):
            return
        for row in self.rows:
            with self.subTest(domain=row["domain"]):
                governance = (
                    row["classification_method"],
                    row["classification_status"],
                    row["classification_confidence"],
                )
                self.assertIn(governance, REFERENCE_GOVERNANCE_COMBINATIONS)
                self.assertTrue(row["classification_evidence"].strip())
                is_uncertain = row["classification_status"] == "reviewed_unclassified"
                self.assertEqual(
                    row["source_category_l1"] == "unclassified_long_tail",
                    is_uncertain,
                )
                self.assertEqual(row["source_type"] == "unclassified", is_uncertain)

    def test_level_two_taxonomy_is_controlled_and_compatible_with_level_one(self) -> None:
        for row in self.rows:
            with self.subTest(domain=row["domain"]):
                allowed = CONTROLLED_SOURCE_TYPES.get(row["source_category_l1"], {})
                self.assertIn(row["source_type"], allowed)
                if row["source_type"] in allowed:
                    self.assertEqual(row["source_type_cn"], allowed[row["source_type"]])

    def test_required_reviewed_examples_are_evidence_based(self) -> None:
        by_domain = {row["domain"]: row for row in self.rows}
        expected = {
            "anxjm.com": ("商业信息与服务", "商业信息与加盟服务"),
            "cityhui.com": ("商业信息与服务", "本地服务与用户内容"),
            "xtrb.cn": ("新闻与媒体", "地方新闻媒体"),
            "ijia.city": ("商业信息与服务", "商业推荐与榜单内容"),
            "shangyexinzhi.com": ("商业信息与服务", "商业内容平台"),
            "csdn.net": ("平台与社区", "专业技术社区"),
            "99.com.cn": ("垂直专业内容", "医疗健康内容"),
            "ruli.com": ("垂直专业内容", "医疗健康内容"),
            "renrendoc.com": ("研究与文档", "文档平台"),
            "xueqiu.com": ("平台与社区", "财经社区"),
            "weibo.com": ("平台与社区", "社交内容平台"),
            "weibo.cn": ("平台与社区", "社交内容平台"),
            "beijing.gov.cn": ("政府与公共机构", "政府机构"),
        }
        for domain, (category_l1_cn, source_type_cn) in expected.items():
            with self.subTest(domain=domain):
                self.assertIn(domain, by_domain)
                if domain not in by_domain:
                    continue
                self.assertEqual(by_domain[domain]["source_category_l1_cn"], category_l1_cn)
                self.assertEqual(by_domain[domain]["source_type_cn"], source_type_cn)

    def test_audited_edge_cases_have_matching_classification_and_evidence(self) -> None:
        by_domain = {row["domain"]: row for row in self.rows}
        expected = {
            "apple.com": ("brand_corporate", "brand_corporate", "high", "apps.apple.com"),
            "dfcfw.com": (
                "research_documentation",
                "document_platform",
                "high",
                "pdf.dfcfw.com",
            ),
            "3000.cn": (
                "business_services",
                "commercial_recommendation",
                "medium",
                "/brand-column",
            ),
            "yunucms.com": (
                "business_services",
                "business_content_platform",
                "medium",
                "news.yunucms.com/brand",
            ),
            "xudoodoo.com": (
                "business_services",
                "commercial_recommendation",
                "medium",
                "xudoodoo.com/detail",
            ),
            "xinchufang.com": (
                "vertical_professional",
                "consumer_lifestyle_content",
                "medium",
                "xinchufang.com/shipu",
            ),
        }
        for domain, (category, source_type, confidence, evidence_fragment) in expected.items():
            with self.subTest(domain=domain):
                row = by_domain[domain]
                self.assertEqual(row["source_category_l1"], category)
                self.assertEqual(row["source_type"], source_type)
                self.assertEqual(row["classification_confidence"], confidence)
                self.assertIn(evidence_fragment, row["classification_evidence"])

        reviewed_uncertain = {
            row["domain"]
            for row in self.rows
            if row["classification_status"] == "reviewed_unclassified"
        }
        self.assertEqual(reviewed_uncertain, {"cqjiuque.cn", "snqa.com.cn"})

    def test_manual_review_rows_equal_current_top_500_pending_domains(self) -> None:
        manual_domains = {
            row["domain"]
            for row in self.rows
            if row["classification_method"] == "manual_review"
        }
        legacy_domains = {
            row["domain"]
            for row in self.rows
            if row["classification_method"] == "legacy_curated"
        }
        connection = duckdb.connect(
            str(REPO_ROOT / "data" / "catalog" / "cn_geo.duckdb"), read_only=True
        )
        try:
            connection.execute("CREATE TEMP TABLE legacy_domains(domain VARCHAR)")
            connection.executemany(
                "INSERT INTO legacy_domains VALUES (?)", [(domain,) for domain in legacy_domains]
            )
            expected_domains = {
                row[0]
                for row in connection.execute(
                    "SELECT s.domain FROM sources s "
                    "LEFT JOIN legacy_domains l USING (domain) "
                    "WHERE l.domain IS NULL "
                    "ORDER BY s.unique_citation_count DESC, s.domain LIMIT 500"
                ).fetchall()
            }
        finally:
            connection.close()
        self.assertEqual(manual_domains, expected_domains)

    def test_production_preflight_accepts_reference(self) -> None:
        validator = getattr(warehouse_builder, "validate_source_type_reference", None)
        self.assertIsNotNone(validator)
        if validator is None:
            return
        validator(REPO_ROOT / "data" / "reference" / "source_types.csv")


class SourceClassificationPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mapping_path = REPO_ROOT / "data" / "reference" / "source_types.csv"
        with mapping_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            cls.rows = list(reader)
            cls.fieldnames = list(reader.fieldnames or [])

    def assert_invalid_reference(self, mutate, expected_message: str) -> None:
        validator = getattr(warehouse_builder, "validate_source_type_reference", None)
        self.assertIsNotNone(validator)
        if validator is None:
            return
        rows = [dict(row) for row in self.rows]
        mutate(rows)
        with tempfile.TemporaryDirectory() as temp:
            mapping_path = Path(temp) / "source_types.csv"
            with mapping_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(RuntimeError, expected_message):
                validator(mapping_path)

    def test_preflight_rejects_non_normalized_domain(self) -> None:
        self.assert_invalid_reference(
            lambda rows: rows[0].__setitem__("domain", rows[0]["domain"].upper()),
            "domain",
        )

    def test_preflight_rejects_duplicate_normalized_domain(self) -> None:
        def duplicate(rows):
            rows[1]["domain"] = rows[0]["domain"]

        self.assert_invalid_reference(duplicate, "duplicate domain")

    def test_preflight_rejects_invalid_level_one_level_two_pair(self) -> None:
        self.assert_invalid_reference(
            lambda rows: rows[0].__setitem__("source_type", "unclassified"),
            "source_type",
        )

    def test_preflight_rejects_invalid_governance_combination(self) -> None:
        self.assert_invalid_reference(
            lambda rows: rows[0].__setitem__("classification_confidence", "low"),
            "governance",
        )

    def test_preflight_rejects_blank_evidence(self) -> None:
        self.assert_invalid_reference(
            lambda rows: rows[0].__setitem__("classification_evidence", ""),
            "evidence",
        )

    def test_invalid_reference_fails_build_before_quality_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_root = root / "data"
            data_root.mkdir()
            shutil.copy2(REPO_ROOT / "data" / "manifest.json", data_root / "manifest.json")
            shutil.copytree(REPO_ROOT / "data" / "reference", data_root / "reference")
            (data_root / "records").symlink_to(
                REPO_ROOT / "data" / "records", target_is_directory=True
            )
            mapping_path = data_root / "reference" / "source_types.csv"
            rows = [dict(row) for row in self.rows]
            rows[0]["classification_evidence"] = ""
            with mapping_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "build_data_warehouse.py"),
                    "--repo-root",
                    str(root),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("evidence", result.stderr)
            self.assertFalse((data_root / "quality").exists())



class DeterministicBuildTests(unittest.TestCase):
    def build_isolated_derived_tables(self, root: Path) -> dict[str, str]:
        data_root = root / "data"
        data_root.mkdir(parents=True)
        shutil.copy2(REPO_ROOT / "data" / "manifest.json", data_root / "manifest.json")
        shutil.copytree(REPO_ROOT / "data" / "reference", data_root / "reference")
        (data_root / "records").symlink_to(
            REPO_ROOT / "data" / "records", target_is_directory=True
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "build_data_warehouse.py"),
                "--repo-root",
                str(root),
            ],
            check=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        hashes: dict[str, str] = {}
        for directory in ("curated", "features", "marts"):
            for path in sorted((data_root / directory).rglob("*.parquet")):
                relative = path.relative_to(data_root).as_posix()
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes

    def test_two_isolated_builds_have_identical_sources_pages_features_and_marts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = self.build_isolated_derived_tables(root / "first")
            second = self.build_isolated_derived_tables(root / "second")
        self.assertTrue(any("sources" in path for path in first))
        self.assertTrue(any("pages" in path for path in first))
        self.assertTrue(any(path.startswith("features/") for path in first))
        self.assertTrue(any(path.startswith("marts/") for path in first))
        self.assertEqual(first, second)


class CleaningRuleTests(unittest.TestCase):
    def test_url_canonicalization_removes_tracking_and_fragment(self) -> None:
        raw = "HTTPS://Example.COM/path/?utm_source=x&b=2&a=1#section"
        self.assertEqual(canonicalize_url(raw), "https://example.com/path?a=1&b=2")
        self.assertEqual(classify_url(raw), "valid_http")

    def test_url_statuses_preserve_missing_and_expired(self) -> None:
        self.assertEqual(classify_url(""), "missing")
        self.assertEqual(classify_url("expired_url"), "expired")
        self.assertEqual(classify_url("ftp://example.com/a"), "invalid_scheme")

    def test_date_formats_are_explicit(self) -> None:
        self.assertEqual(normalize_published_at("2026年06月05日"), "2026-06-05")
        self.assertEqual(classify_published_at("2026年06月05日"), "parsed_chinese_date")
        self.assertEqual(classify_published_at("946659692"), "parsed_unix_seconds")
        self.assertTrue(normalize_published_at("1780329600").startswith("2026-06-02T00:00:00+08:00"))
        self.assertEqual(normalize_published_at("2021年"), "2021")
        self.assertEqual(classify_published_at("2021年"), "partial_year")
        self.assertEqual(classify_published_at("0"), "placeholder_zero")

    def test_failed_install_restores_previous_derived_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_root = root / "data"
            build_root = data_root / ".build"
            data_root.mkdir()
            build_root.mkdir()
            for name in DERIVED_DIRS:
                old_dir = data_root / name
                new_dir = build_root / name
                old_dir.mkdir()
                new_dir.mkdir()
                (old_dir / "state.txt").write_text("old", encoding="utf-8")
                (new_dir / "state.txt").write_text("new", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "validator failed"):
                install_outputs(
                    data_root,
                    build_root,
                    force=True,
                    validator=lambda: (_ for _ in ()).throw(RuntimeError("validator failed")),
                )

            for name in DERIVED_DIRS:
                self.assertEqual((data_root / name / "state.txt").read_text(encoding="utf-8"), "old")

    def test_raw_preflight_rejects_url_userinfo(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo_root = Path(temp)
            shard_path = repo_root / "data" / "records" / "test" / "part-0001.jsonl"
            shard_path.parent.mkdir(parents=True)
            record = {
                "quote_url": "https://" + "user" + ":" + "credential" + "@example.com/path"
            }
            shard_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            manifest = {
                "categories": [
                    {
                        "files": [
                            {
                                "path": "data/records/test/part-0001.jsonl",
                                "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                            }
                        ]
                    }
                ]
            }

            with self.assertRaisesRegex(RuntimeError, "引用 URL 含用户凭据"):
                validate_raw_release(repo_root, manifest)


class WarehouseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((REPO_ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
        cls.connection = duckdb.connect(str(REPO_ROOT / "data" / "catalog" / "cn_geo.duckdb"), read_only=True)
        fact_path = (
            REPO_ROOT
            / "data"
            / "curated"
            / "citation_observations"
            / "**"
            / "*.parquet"
        ).as_posix()
        cls.connection.execute(
            "CREATE TEMP VIEW citation_observations AS "
            f"SELECT * FROM read_parquet('{fact_path}', hive_partitioning=true)"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.connection.close()

    def scalar(self, query: str):
        return self.connection.execute(query).fetchone()[0]

    def test_sources_include_classification_governance_schema(self) -> None:
        columns = {
            row[0] for row in self.connection.execute("DESCRIBE sources").fetchall()
        }
        self.assertTrue(
            {
                "source_category_l1",
                "source_category_l1_cn",
                "source_type",
                "source_type_cn",
                "classification_status",
                "classification_method",
                "classification_confidence",
                "classification_evidence",
            }
            <= columns
        )

    def test_government_domains_and_proxy_exceptions_are_classified(self) -> None:
        columns = {
            row[0] for row in self.connection.execute("DESCRIBE sources").fetchall()
        }
        if "source_category_l1_cn" not in columns:
            self.skipTest("classification schema has not been generated yet")
        self.assertEqual(
            self.scalar(
                "SELECT count(*) FROM sources "
                "WHERE (domain = 'gov.cn' OR ends_with(domain, '.gov.cn')) "
                "AND source_category_l1_cn <> '政府与公共机构'"
            ),
            0,
        )
        for domain in ("quark.cn", "sm.cn"):
            with self.subTest(domain=domain):
                self.assertEqual(
                    self.scalar(
                        "SELECT source_category_l1_cn FROM sources "
                        f"WHERE domain = '{domain}'"
                    ),
                    "搜索与页面代理",
                )
                self.assertEqual(
                    self.scalar(
                        "SELECT source_display_name FROM sources "
                        f"WHERE domain = '{domain}'"
                    ),
                    domain,
                )

    def test_unclassified_linked_citation_share_is_at_most_eighteen_percent(self) -> None:
        columns = {
            row[0] for row in self.connection.execute("DESCRIBE sources").fetchall()
        }
        if "source_category_l1_cn" not in columns:
            self.skipTest("classification schema has not been generated yet")
        share = self.scalar(
            "SELECT coalesce(sum(unique_citation_count) FILTER "
            "(WHERE source_category_l1_cn = '未分类长尾'), 0)::DOUBLE "
            "/ sum(unique_citation_count) FROM sources"
        )
        self.assertLessEqual(share, 0.18)

    def test_fact_table_preserves_every_raw_observation(self) -> None:
        expected = self.manifest["summary"]["records"]
        self.assertEqual(self.scalar("SELECT count(*) FROM citation_observations"), expected)
        self.assertEqual(self.scalar("SELECT count(DISTINCT citation_id) FROM citation_observations"), expected)

    def test_catalog_is_self_contained_after_relocation(self) -> None:
        catalog_path = REPO_ROOT / "data" / "catalog" / "cn_geo.duckdb"
        with tempfile.TemporaryDirectory() as temp:
            relocated_path = Path(temp) / "relocated.duckdb"
            shutil.copy2(catalog_path, relocated_path)
            connection = duckdb.connect(str(relocated_path), read_only=True)
            try:
                self.assertEqual(
                    connection.execute("SELECT count(*) FROM source_visibility").fetchone()[0],
                    19190,
                )
                view_count = connection.execute(
                    "SELECT count(*) FROM duckdb_views() WHERE internal = false"
                ).fetchone()[0]
                self.assertEqual(view_count, 0)
            finally:
                connection.close()

    def test_duplicate_count_matches_release_manifest(self) -> None:
        actual = self.scalar(
            "SELECT count(*) - count(DISTINCT record_hash) FROM citation_observations"
        )
        self.assertEqual(actual, self.manifest["summary"]["exact_duplicate_records"])

    def test_all_platforms_and_categories_are_mapped(self) -> None:
        self.assertEqual(self.scalar("SELECT count(*) FROM ai_platforms"), 12)
        self.assertEqual(self.scalar("SELECT count(*) FROM questions"), 620)
        self.assertEqual(self.scalar("SELECT count(*) FROM question_labels"), 620)
        self.assertEqual(self.scalar("SELECT count(*) FROM platform_overlap"), 66)

    def test_platform_specific_quote_indices_are_parsed(self) -> None:
        self.assertEqual(
            self.scalar(
                "SELECT count(*) FROM citation_observations "
                "WHERE platform_code = 'KIMI' AND quote_index_parse_status = 'web_search'"
            ),
            4794,
        )
        self.assertEqual(
            self.scalar(
                "SELECT count(*) FROM citation_observations "
                "WHERE quote_index_parse_status = 'unparsed'"
            ),
            0,
        )

    def test_unix_publication_dates_use_china_timezone(self) -> None:
        self.assertGreater(
            self.scalar(
                "SELECT count(*) FROM citation_observations "
                "WHERE published_at_raw = '1780329600.0' "
                "AND published_date = DATE '2026-06-02'"
            ),
            0,
        )

    def test_missing_response_boundaries_are_not_fabricated(self) -> None:
        self.assertEqual(self.scalar("SELECT count(*) FROM responses"), 0)
        self.assertEqual(
            self.scalar("SELECT count(*) FROM citation_observations WHERE response_id IS NOT NULL"),
            0,
        )

    def test_optional_metadata_availability_is_separate_from_quality_flags(self) -> None:
        self.assertEqual(
            self.scalar(
                "SELECT count(*) FROM citation_observations "
                "WHERE availability_flags LIKE '%published_date_unavailable%'"
            ),
            83310,
        )
        self.assertEqual(
            self.scalar(
                "SELECT count(*) FROM citation_observations "
                "WHERE availability_flags LIKE '%published_date_unavailable%' "
                "AND quality_flags = ''"
            ),
            56174,
        )

    def test_page_dates_distinguish_unknown_consistent_and_conflicting_values(self) -> None:
        statuses = dict(
            self.connection.execute(
                "SELECT published_date_status, count(*) FROM pages GROUP BY published_date_status"
            ).fetchall()
        )
        self.assertEqual(statuses, {"unknown": 43675, "consistent": 63314, "conflicting": 670})
        self.assertEqual(
            self.scalar(
                "SELECT count(*) FROM pages WHERE published_date_status = 'conflicting' "
                "AND representative_published_date IS NOT NULL"
            ),
            0,
        )

    def test_quality_report_passed(self) -> None:
        report_path = (
            REPO_ROOT
            / "data"
            / "quality"
            / f"release_date={self.manifest['release_date']}"
            / "quality_report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")
        self.assertTrue(all(report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
