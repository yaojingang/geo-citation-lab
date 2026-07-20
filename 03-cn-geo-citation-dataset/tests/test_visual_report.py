import hashlib
import json
import re
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_visual_report import (  # noqa: E402
    ECHARTS_SHA256,
    build_payload,
    resolve_echarts_text,
)


TEMPLATE_PATH = ROOT / "reports" / "src" / "report_template.html"
RUNTIME_PATH = ROOT / "reports" / "src" / "report_runtime.js"
REPORT_PATH = ROOT / "reports" / "CN-GEO_多维数据分析报告.html"


class ReportMarkupParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.chart_ids: set[str] = set()
        self.remote_assets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            self.ids.append(element_id)
        classes = (attributes.get("class") or "").split()
        if element_id and "chart" in classes:
            self.chart_ids.add(element_id)
        if tag in {"script", "link", "img"}:
            asset = attributes.get("src") or attributes.get("href")
            if asset and asset.startswith(("http://", "https://", "//")):
                self.remote_assets.append(asset)


class VisualReportTests(unittest.TestCase):
    def test_template_ids_are_unique_and_all_charts_are_registered(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        runtime = RUNTIME_PATH.read_text(encoding="utf-8")
        parser = ReportMarkupParser()
        parser.feed(template)

        duplicate_ids = sorted({item for item in parser.ids if parser.ids.count(item) > 1})
        runtime_chart_ids = set(re.findall(r"register\('([^']+)'", runtime))

        self.assertEqual(duplicate_ids, [])
        self.assertEqual(len(parser.chart_ids), 43)
        self.assertEqual(parser.chart_ids, runtime_chart_ids)

    def test_card_descriptions_use_direct_copy(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("<b>目的：</b>", template)
        self.assertNotIn("<b>作用：</b>", template)
        self.assertIn('id="chartUnclassifiedSources"', template)
        self.assertNotIn('id="chartPosition"', template)
        self.assertNotIn("02.6 · Quote position", template)

    def test_ecosystem_shares_use_one_source_table_denominator(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        runtime = RUNTIME_PATH.read_text(encoding="utf-8")

        self.assertNotIn("36.3%", template)
        self.assertIn("未归属生态", template)
        self.assertIn("信源类型分类覆盖率与未分类长尾见 04.5", runtime)
        self.assertNotIn("row.ecosystem!=='未分类信源'", runtime)
        self.assertIn("const ecosystemTotal=REPORT.ecosystem.reduce", runtime)
        self.assertIn("unclassified.dedup_count/ecosystemTotal", runtime)
        self.assertNotIn("unclassified.dedup_count/o.dedup_citations", runtime)

    def test_filter_grid_contains_only_the_six_filter_fields(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        filter_grid = template.split('<div class="filter-grid">', 1)[1].split(
            '<p class="scope-summary">', 1
        )[0]

        self.assertEqual(filter_grid.count("<select"), 6)
        self.assertNotIn('id="resetFilters"', filter_grid)
        self.assertIn('class="filter-head"', template)
        self.assertIn('id="resetFilters"', template)

    def test_filter_arrows_have_consistent_inset_spacing(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        self.assertIn("background-position:right 14px center", template)
        self.assertIn("padding:0 42px 0 11px", template)
        self.assertIn("appearance:none", template)

    def test_processing_status_layout_has_desktop_tablet_and_mobile_columns(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        self.assertIn(".processing-grid{display:grid;grid-template-columns:repeat(4", template)
        self.assertIn(".processing-grid{grid-template-columns:repeat(2,1fr)}", template)
        self.assertIn(".processing-grid{grid-template-columns:1fr}", template)

    def test_classification_coverage_tablet_uses_two_rows_and_a_full_width_final_item(
        self,
    ) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        coverage = template.split(
            '<div class="classification-coverage"', 1
        )[1].split("</div></div><div class=\"chart preference-types\"", 1)[0]
        tablet_css = template.split("@media(max-width:1040px){", 1)[1].split(
            "@media(max-width:760px){", 1
        )[0]

        self.assertEqual(coverage.count('class="classification-item"'), 5)
        self.assertIn(
            ".classification-coverage{display:grid;grid-template-columns:repeat(5",
            template,
        )
        self.assertIn("border-bottom:1px solid var(--border)", template)
        self.assertIn(
            ".classification-item:nth-child(2n){border-right:0}", tablet_css
        )
        self.assertIn(
            ".classification-item:nth-child(-n+4){border-bottom:1px solid var(--border)}",
            tablet_css,
        )
        self.assertIn(
            ".classification-item:nth-child(5){grid-column:1/-1;border-right:0}",
            tablet_css,
        )

    def test_chapter_titles_are_single_line_chinese_headers(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        self.assertEqual(template.count('<header class="chapter-head">'), 10)
        self.assertEqual(template.count('<span class="chapter-no">'), 10)
        self.assertNotIn("chapter-kicker", template)
        self.assertNotIn("Data foundation", template)
        self.assertIn(
            '<h2 class="chapter-title"><span class="chapter-no">01</span><span>数据底座与分析适用性</span></h2>',
            template,
        )

    def test_source_preference_chapter_has_local_filter_and_core_analysis_controls(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        runtime = RUNTIME_PATH.read_text(encoding="utf-8")

        self.assertIn('id="chapter-4"', template)
        self.assertIn('id="preferenceProductFilter"', template)
        self.assertIn('id="chartSourcePreference"', template)
        self.assertIn('id="preferenceSourceSelect"', template)
        self.assertIn(
            "const matrixSources=[...new Map(REPORT.sourcePreference.map",
            runtime,
        )
        self.assertIn(
            "addOptions(preferenceSource,matrixSources",
            runtime,
        )
        self.assertIn('id="chartAnchorMigration"', template)
        self.assertIn('id="preferenceRankScope"', template)
        self.assertIn('id="preferenceTopN"', template)
        self.assertIn('id="preferenceRankTable"', template)
        self.assertIn('id="chartTerminalTilt"', template)
        self.assertIn('id="chartPreferenceTypes"', template)
        self.assertIn('id="preferenceTable"', template)
        self.assertIn('id="pairQualifiedSourceJaccard"', template)
        self.assertNotIn('id="chartPreferenceConcentration"', template)
        self.assertNotIn("chartPreferenceConcentration", runtime)
        self.assertNotIn("preferenceConcentration", runtime)
        self.assertIn("register('chartAnchorMigration'", runtime)
        self.assertIn("REPORT.anchorSourceMigration", runtime)
        self.assertIn("REPORT.sourceTopRanks", runtime)
        self.assertIn("REPORT.classificationCoverage", runtime)
        self.assertIn("state.preferenceProduct=event.target.value", runtime)
        self.assertIn("state.selectedPreferenceSource", runtime)
        self.assertIn("preferenceRankScope:'common'", runtime)
        self.assertIn("preferenceTopN:10", runtime)
        self.assertNotIn("335 个共同问题", runtime)
        self.assertNotIn("380 个达标信源", runtime)

    def test_source_preference_rank_controls_are_accessible_and_scope_is_explained(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")

        rank_section = template.split('id="preferenceRankBoard"', 1)[1].split("</article>", 1)[0]
        self.assertIn('role="group"', rank_section)
        self.assertGreaterEqual(rank_section.count('aria-pressed="'), 4)
        self.assertIn('aria-label="排行口径"', rank_section)
        self.assertIn('aria-label="排行数量"', rank_section)
        self.assertIn('role="table"', rank_section)
        self.assertIn("334 个共同问题", rank_section)
        self.assertIn("各端全部有效信源问题", rank_section)
        self.assertIn("Kimi 不进入本章", template)

    def test_source_preference_module_order_and_governance_copy_are_current(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        runtime = RUNTIME_PATH.read_text(encoding="utf-8")

        self.assertLess(template.index("04.1 · Preference matrix"), template.index("04.2 · Anchor migration"))
        self.assertLess(template.index("04.2 · Anchor migration"), template.index("04.3 · Top source ranks"))
        self.assertLess(template.index("04.3 · Top source ranks"), template.index("04.4 · Terminal tilt"))
        self.assertLess(template.index("04.4 · Terminal tilt"), template.index("04.5 · Source type mix"))
        self.assertLess(template.index("04.5 · Source type mix"), template.index("04.6 · GEO shortlist"))
        self.assertIn('id="classificationCoverage"', template)
        self.assertIn('id="classificationUnclassified"', template)
        self.assertIn('id="classificationUnnormalized"', template)
        self.assertIn('id="classificationRuleShare"', template)
        self.assertIn("Top 500", template)
        self.assertIn(".gov.cn", template)
        self.assertNotIn("Concentration", template)
        self.assertIn("八端共同问题等权口径下，可识别信源分类覆盖率", runtime)
        self.assertIn("anchorHighLabels", runtime)

    def test_rate_views_are_present_and_denominators_are_named(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        runtime = RUNTIME_PATH.read_text(encoding="utf-8")

        self.assertIn('id="rateOverview"', template)
        self.assertIn('data-metric-view="platformScale"', template)
        self.assertIn('data-metric-view="sourceTypes"', template)
        self.assertIn('data-metric-view="labelScale"', template)
        self.assertIn('id="terminalRateGrid"', template)
        self.assertIn('id="anchorCarryover"', template)
        self.assertIn("样本引用份额", runtime)
        self.assertIn("当前筛选问题覆盖率", runtime)
        self.assertIn("锚定端 Top 10 出现率", runtime)
        self.assertIn("共同问题覆盖率", template)
        self.assertIn("平台渗透率", runtime)
        self.assertIn("平台空白率", runtime)
        self.assertIn("问题覆盖率", runtime)

    def test_preference_rank_board_has_mobile_scroll_and_source_buttons(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        runtime = RUNTIME_PATH.read_text(encoding="utf-8")

        self.assertIn(".preference-rank-scroll{overflow-x:auto", template)
        self.assertIn("min-width:1088px", template)
        self.assertNotIn("min-width:1180px", template)
        self.assertIn("position:sticky", template)
        self.assertIn('id="preferenceRankSticky"', template)
        self.assertIn('id="preferenceRankStickyHead"', template)
        self.assertIn("sticky.scrollLeft=scroll.scrollLeft", runtime)
        self.assertIn("rank-source-button", runtime)
        self.assertIn("ariaPressed", runtime)
        self.assertIn("updatePreferenceRankTable", runtime)

        anchor_section = template.split("04.2 · Anchor migration", 1)[1].split(
            "</article>", 1
        )[0]
        self.assertEqual(anchor_section.count('data-preference-group="anchor"'), 1)
        self.assertEqual(anchor_section.count('data-preference-group="control"'), 1)
        self.assertIn('role="list"', anchor_section)
        self.assertIn("cell.dataset.endpoint=item.platform_code", runtime)
        self.assertIn("cell.setAttribute('role','listitem')", runtime)
        self.assertIn("td.dataset.pairEnd=String(index%2===1)", runtime)
        self.assertIn('[data-pair-end="true"]', template)

        anchor_factory = runtime.split("register('chartAnchorMigration'", 1)[1].split(
            "register('chartTerminalTilt'", 1
        )[0]
        self.assertIn("一级类型 ${esc(d.source_category_l1_cn)}", anchor_factory)
        self.assertIn("二级类型 ${esc(d.source_type_cn)}", anchor_factory)

        selection_function = runtime.split("function selectPreferenceSource", 1)[1].split(
            "function draw", 1
        )[0]
        self.assertIn("state.selectedPreferenceSource=", selection_function)
        self.assertIn("drawPreferenceViews()", selection_function)
        self.assertIn("'chartTerminalTilt'", runtime.split("const preferenceChartIds", 1)[1].split(";", 1)[0])
        self.assertIn("validatePreferenceLinkage()", runtime)
        self.assertIn("button.dataset.productFamily=row.product_family", runtime)
        self.assertIn("state.preferenceProduct=productFamily", runtime)
        self.assertIn("document.getElementById('preferenceProductFilter').value=productFamily", runtime)
        self.assertIn("document.getElementById('classificationRuleShare').textContent=pct(coverage.rule_share)", runtime)
        self.assertIn("八端共同问题等权口径", template)
        self.assertNotIn("全局可识别信源引用已完成分类", template)

    def test_preference_interactions_in_system_chrome(self) -> None:
        playwright_paths = sorted(
            Path.home().glob(".npm/_npx/*/node_modules/playwright"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        chrome_path = Path(
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        )
        if not playwright_paths:
            self.skipTest("Playwright is not installed in the local npm cache")
        if not chrome_path.is_file():
            self.skipTest("System Chrome is unavailable")

        browser_script = r"""
const {pathToFileURL}=require('url');
const {chromium}=require(process.argv[2]);
const reportPath=process.argv[3];
const chromePath=process.argv[4];
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:chromePath,args:['--allow-file-access-from-files']});
  const page=await browser.newPage({viewport:{width:1440,height:900}});
  const pageErrors=[];
  page.on('pageerror',error=>pageErrors.push(error.message));
  await page.goto(pathToFileURL(reportPath).href,{waitUntil:'load'});
  await page.waitForFunction(()=>document.querySelectorAll('#preferenceRankBody tr').length===10&&document.getElementById('preferenceSourceSelect').options.length===21,null,{timeout:30000});

  const rankSize=await page.locator('.preference-rank-scroll').evaluate(element=>({clientWidth:element.clientWidth,scrollWidth:element.scrollWidth}));
  const findings=await page.locator('#findingList').innerText();
  await page.locator('[data-metric-view="platformScale"][data-metric-value="share"]').click();
  await page.locator('[data-metric-view="sourceTypes"][data-metric-value="share"]').click();
  await page.locator('[data-metric-view="labelScale"][data-metric-value="rate"]').click();
  await page.waitForFunction(()=>{
    const name=id=>echarts.getInstanceByDom(document.getElementById(id)).getOption().xAxis[0].name;
    return name('chartPlatformScale')==='样本引用份额'&&name('chartSourceTypes')==='样本引用构成率'&&name('chartLabelScale')==='题均引用';
  });
  const rateViews=await page.evaluate(()=>({
    platformAxis:echarts.getInstanceByDom(document.getElementById('chartPlatformScale')).getOption().xAxis[0].name,
    sourceTypeAxis:echarts.getInstanceByDom(document.getElementById('chartSourceTypes')).getOption().xAxis[0].name,
    labelAxis:echarts.getInstanceByDom(document.getElementById('chartLabelScale')).getOption().xAxis[0].name,
    pressed:[...document.querySelectorAll('[data-metric-view][aria-pressed="true"]')].map(button=>button.dataset.metricValue),
    terminalCards:document.querySelectorAll('#terminalRateGrid .rate-detail').length,
    rateOverview:[...document.querySelectorAll('#rateOverview .rate-kpi b')].map(element=>element.textContent)
  }));

  await page.selectOption('#preferenceProductFilter','豆包');
  await page.locator('#preferenceRankBody .rank-source-button[data-product-family="千问"]').first().click();
  await page.waitForFunction(()=>document.getElementById('preferenceProductFilter').value==='千问'&&document.querySelectorAll('#preferenceTable tr.is-selected').length===1);
  const crossProduct={
    product:await page.locator('#preferenceProductFilter').inputValue(),
    selectedRows:await page.locator('#preferenceTable tr.is-selected').count()
  };

  const selector=page.locator('#preferenceSourceSelect');
  const optionCount=await selector.locator('option').count();
  const lastOption=await selector.locator('option').last().evaluate(option=>({value:option.value,label:option.textContent}));
  await selector.selectOption('');
  await page.waitForFunction(()=>document.querySelectorAll('#preferenceTable tr.is-selected').length===0);
  await selector.selectOption(lastOption.value);
  await page.waitForFunction(sourceId=>document.getElementById('preferenceSourceSelect').value===sourceId&&document.querySelectorAll('#preferenceTable tr.is-selected').length===1,lastOption.value);
  const keyboardSelection={
    optionCount,
    lastValue:lastOption.value,
    selectedValue:await selector.inputValue(),
    lastLabel:lastOption.label,
    selectedRowText:await page.locator('#preferenceTable tr.is-selected').innerText()
  };

  await page.setViewportSize({width:390,height:844});
  await page.evaluate(()=>{const table=document.querySelector('.preference-rank-scroll');window.scrollTo(0,table.getBoundingClientRect().top+window.scrollY+100);});
  await page.waitForFunction(()=>Math.abs(document.getElementById('preferenceRankSticky').getBoundingClientRect().top-54)<=2);
  const mobile=await page.evaluate(()=>{
    const main=document.querySelector('.preference-rank-scroll');
    const sticky=document.getElementById('preferenceRankSticky');
    main.scrollLeft=320;
    main.dispatchEvent(new Event('scroll'));
    return {stickyTop:sticky.getBoundingClientRect().top,mainLeft:main.scrollLeft,stickyLeft:sticky.scrollLeft,pageWidth:document.documentElement.scrollWidth,viewportWidth:window.innerWidth};
  });
  await browser.close();
  process.stdout.write(JSON.stringify({rankSize,findings,rateViews,crossProduct,keyboardSelection,mobile,pageErrors}));
})().catch(error=>{console.error(error);process.exit(1);});
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            report_path = temp / "visual-report.html"
            script_path = temp / "browser-check.js"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_visual_report.py"),
                    "--output",
                    str(report_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            script_path.write_text(browser_script, encoding="utf-8")
            result = subprocess.run(
                [
                    "node",
                    str(script_path),
                    str(playwright_paths[0]),
                    str(report_path),
                    str(chrome_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )

        behavior = json.loads(result.stdout)
        self.assertEqual(
            behavior["rankSize"]["clientWidth"],
            behavior["rankSize"]["scrollWidth"],
        )
        self.assertAlmostEqual(behavior["mobile"]["stickyTop"], 54, delta=2)
        self.assertEqual(
            behavior["mobile"]["mainLeft"], behavior["mobile"]["stickyLeft"]
        )
        self.assertEqual(
            behavior["mobile"]["pageWidth"], behavior["mobile"]["viewportWidth"]
        )
        self.assertEqual(
            behavior["rateViews"]["platformAxis"], "样本引用份额"
        )
        self.assertEqual(
            behavior["rateViews"]["sourceTypeAxis"], "样本引用构成率"
        )
        self.assertEqual(behavior["rateViews"]["labelAxis"], "题均引用")
        self.assertEqual(behavior["rateViews"]["pressed"], ["share", "share", "rate"])
        self.assertEqual(behavior["rateViews"]["terminalCards"], 4)
        self.assertTrue(
            all(value.endswith("%") for value in behavior["rateViews"]["rateOverview"])
        )
        self.assertEqual(behavior["crossProduct"], {"product": "千问", "selectedRows": 1})
        self.assertEqual(behavior["keyboardSelection"]["optionCount"], 21)
        self.assertEqual(
            behavior["keyboardSelection"]["lastValue"],
            behavior["keyboardSelection"]["selectedValue"],
        )
        last_source_name = behavior["keyboardSelection"]["lastLabel"].split(" · ", 1)[0]
        self.assertIn(last_source_name, behavior["keyboardSelection"]["selectedRowText"])
        for endpoint in ("DeepSeek电脑端", "DeepSeek移动端", "元宝移动端"):
            self.assertIn(endpoint, behavior["findings"])
        self.assertEqual(behavior["pageErrors"], [])

    def test_availability_language_and_chart_ids_replace_missingness_language(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        runtime = RUNTIME_PATH.read_text(encoding="utf-8")

        self.assertIn('id="chartFieldAvailability"', template)
        self.assertIn('id="chartApplicability"', template)
        self.assertIn('id="chartPlatformAvailability"', template)
        self.assertIn('id="processingStatus"', template)
        self.assertNotIn('id="chartCompleteness"', template)
        self.assertNotIn('id="chartQuality"', template)
        self.assertNotIn('id="chartQualityHeat"', template)
        self.assertIn("REPORT.fieldAvailability", runtime)
        self.assertIn("REPORT.analysisApplicability", runtime)
        self.assertIn("REPORT.platformAvailability", runtime)

    def test_runtime_avoids_dynamic_html_injection(self) -> None:
        runtime = RUNTIME_PATH.read_text(encoding="utf-8")
        self.assertNotIn("innerHTML", runtime)

    def test_built_report_is_self_contained(self) -> None:
        report = REPORT_PATH.read_text(encoding="utf-8")
        parser = ReportMarkupParser()
        parser.feed(report)

        self.assertEqual(parser.remote_assets, [])
        self.assertNotIn("__REPORT_DATA__", report)
        self.assertNotIn("__ECHARTS_LIBRARY__", report)
        self.assertNotIn("__REPORT_RUNTIME__", report)

    def test_built_report_matches_current_builder_template_and_runtime(self) -> None:
        report = REPORT_PATH.read_text(encoding="utf-8")
        source_paths = [ROOT / "scripts" / "build_visual_report.py", TEMPLATE_PATH, RUNTIME_PATH]

        for path in source_paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertIn(digest, report)
        self.assertIn(RUNTIME_PATH.read_text(encoding="utf-8"), report)

    def test_packaged_echarts_copy_matches_pinned_hash(self) -> None:
        echarts_text = resolve_echarts_text(None)

        self.assertEqual(hashlib.sha256(echarts_text.encode("utf-8")).hexdigest(), ECHARTS_SHA256)


class PayloadSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        connection = duckdb.connect(str(ROOT / "data" / "catalog" / "cn_geo.duckdb"), read_only=True)
        try:
            cls.payload = build_payload(connection)
        finally:
            connection.close()

    def test_unknown_title_and_snippet_are_separate_length_bands(self) -> None:
        title_rows = {
            row["band"]: row for row in self.payload["lengthPerformance"] if row["metric"] == "标题长度"
        }
        snippet_rows = {
            row["band"]: row for row in self.payload["lengthPerformance"] if row["metric"] == "摘要长度"
        }

        self.assertEqual(title_rows["标题未提供"]["page_count"], 990)
        self.assertEqual(snippet_rows["摘要未提供"]["page_count"], 2797)

    def test_freshness_partitions_every_page_without_forcing_conflicts_to_a_year(self) -> None:
        rows = {row["freshness_band"]: row for row in self.payload["freshness"]}

        self.assertEqual(sum(row["page_count"] for row in rows.values()), self.payload["overview"]["pages"])
        self.assertEqual(rows["发布时间未知"]["page_count"], 43675)
        self.assertEqual(rows["发布时间冲突"]["page_count"], 670)
        self.assertEqual(
            sum(rows[band]["page_count"] for band in ("2026", "2025", "2023 至 2024", "2022 及以前")),
            63314,
        )

    def test_optional_metadata_does_not_reduce_observation_applicability(self) -> None:
        applicability = {row["analysis"]: row for row in self.payload["analysisApplicability"]}

        self.assertEqual(applicability["引用观察统计"]["available_records"], 214119)
        self.assertEqual(applicability["引用观察统计"]["ratio"], 1.0)
        self.assertLess(applicability["发布时间分析"]["ratio"], 1.0)

    def test_source_filter_rows_carry_complete_filter_denominators(self) -> None:
        all_rows = [
            row
            for row in self.payload["sourceFilterRows"]
            if row["platform_filter"] == "ALL" and row["type_filter"] == "ALL"
        ]

        self.assertTrue(all_rows)
        self.assertEqual({row["filter_raw_total"] for row in all_rows}, {211355})
        self.assertEqual({row["filter_dedup_total"] for row in all_rows}, {187818})
        self.assertEqual({row["filter_question_count"] for row in all_rows}, {620})
        self.assertTrue(
            all(row["question_count"] <= row["filter_question_count"] for row in all_rows)
        )

    def test_label_stats_use_true_per_question_averages(self) -> None:
        required = {
            "avg_raw_citations_per_question",
            "avg_dedup_citations_per_question",
            "avg_sources_per_question",
            "avg_pages_per_question",
            "avg_platforms_per_question",
        }

        self.assertTrue(all(required <= set(row) for row in self.payload["labelStats"]))
        for row in self.payload["labelStats"]:
            self.assertAlmostEqual(
                row["avg_dedup_citations_per_question"],
                row["dedup_count"] / row["question_count"],
                places=8,
            )
            self.assertGreaterEqual(row["avg_sources_per_question"], 0)
            self.assertGreaterEqual(row["avg_pages_per_question"], 0)
            self.assertLessEqual(row["avg_platforms_per_question"], 12)

    def test_rate_denominators_partition_their_declared_universes(self) -> None:
        overview = self.payload["overview"]
        self.assertEqual(
            sum(row["raw_count"] for row in self.payload["platforms"]),
            overview["raw_citations"],
        )
        self.assertEqual(
            sum(row["dedup_count"] for row in self.payload["platforms"]),
            overview["dedup_citations"],
        )

        platform_counts = {
            row["platform_code"]: (row["raw_count"], row["dedup_count"])
            for row in self.payload["platforms"]
        }
        platform_counts["ALL"] = (
            overview["raw_citations"],
            overview["dedup_citations"],
        )
        for platform_code, expected in platform_counts.items():
            rows = [
                row
                for row in self.payload["sourceTypes"]
                if row["platform_code"] == platform_code
            ]
            self.assertEqual(sum(row["raw_count"] for row in rows), expected[0])
            self.assertEqual(sum(row["dedup_count"] for row in rows), expected[1])

        anchor_size = self.payload["preferenceMeta"]["anchor_pool_size"]
        self.assertTrue(
            all(
                0 <= row["source_count"] <= anchor_size
                for row in self.payload["preferenceMeta"]["anchor_top20_carryover"]
            )
        )
        self.assertTrue(
            all(
                row["source_question_count"] <= row["common_question_count"]
                for row in self.payload["terminalTilt"]
            )
        )
        self.assertTrue(
            all(
                row["platform_count"] + row["potential_platforms"]
                == overview["platforms"]
                and row["question_count"] <= overview["questions"]
                for row in self.payload["expansionCandidates"]
            )
        )

    def test_source_preference_uses_balanced_four_product_sample(self) -> None:
        meta = self.payload["preferenceMeta"]
        matrix = self.payload["sourcePreference"]
        codes = {row["platform_code"] for row in matrix}

        self.assertEqual(meta["balanced_question_count"], 334)
        self.assertEqual(meta["endpoint_count"], 8)
        self.assertEqual(meta["product_count"], 4)
        self.assertEqual(meta["qualified_source_count"], 379)
        self.assertEqual(meta["matrix_source_count"], 20)
        self.assertEqual(meta["inferred_endpoints"], ["腾讯元宝手机版", "千问手机版"])
        self.assertEqual(
            codes,
            {"DB", "DOUBA", "DP", "DPA", "TXYB", "TXYBA", "TYQW", "TYQWA"},
        )
        self.assertEqual(len(matrix), 160)
        self.assertFalse(any("KIMI" in code for code in codes))

    def test_source_preference_index_is_centered_at_one_hundred(self) -> None:
        values: dict[str, list[float]] = {}
        for row in self.payload["sourcePreference"]:
            values.setdefault(row["source_id"], []).append(row["preference_index"])

        self.assertEqual(len(values), 20)
        for source_values in values.values():
            self.assertEqual(len(source_values), 8)
            self.assertAlmostEqual(sum(source_values) / len(source_values), 100.0, places=8)

    def test_preference_type_mix_sums_to_one_per_endpoint(self) -> None:
        shares: dict[str, float] = {}
        for row in self.payload["preferenceTypeMix"]:
            shares[row["platform_code"]] = shares.get(row["platform_code"], 0.0) + row["weighted_share"]

        self.assertEqual(len(shares), 8)
        for share in shares.values():
            self.assertAlmostEqual(share, 1.0, places=8)
        self.assertIn(
            "信源未规范化",
            {row["source_category_l1_cn"] for row in self.payload["preferenceTypeMix"]},
        )

    def test_source_top_ranks_cover_two_scopes_and_eight_endpoints(self) -> None:
        endpoint_order = ["DB", "DOUBA", "DP", "DPA", "TXYB", "TXYBA", "TYQW", "TYQWA"]
        top_ranks = self.payload["sourceTopRanks"]

        self.assertEqual(len(top_ranks), 8 * 20 * 2)
        self.assertEqual({row["scope"] for row in top_ranks}, {"common", "full"})
        self.assertFalse(any("KIMI" in row["platform_code"].upper() for row in top_ranks))
        required_fields = {
            "endpoint",
            "platform",
            "terminal",
            "source_id",
            "source_name",
            "domain",
            "source_category_l1_cn",
            "source_type_cn",
            "rank",
            "share",
            "citation_count",
            "question_count",
            "scope",
            "scope_question_count",
        }
        self.assertTrue(all(required_fields <= set(row) for row in top_ranks))
        for scope in ("common", "full"):
            for endpoint in endpoint_order:
                endpoint_rows = [
                    row
                    for row in top_ranks
                    if row["scope"] == scope and row["platform_code"] == endpoint
                ]
                self.assertEqual([row["rank"] for row in endpoint_rows], list(range(1, 21)))
                shares = [row["share"] for row in endpoint_rows]
                self.assertEqual(shares, sorted(shares, reverse=True))
                self.assertTrue(all(row["scope_question_count"] > 0 for row in endpoint_rows))
        common_counts = {
            row["scope_question_count"]
            for row in top_ranks
            if row["scope"] == "common"
        }
        self.assertEqual(common_counts, {334})
        self.assertEqual(
            self.payload["preferenceMeta"]["full_scope_question_counts"],
            [
                {"platform_code": "DB", "question_count": 590},
                {"platform_code": "DOUBA", "question_count": 589},
                {"platform_code": "DP", "question_count": 589},
                {"platform_code": "DPA", "question_count": 587},
                {"platform_code": "TXYB", "question_count": 598},
                {"platform_code": "TXYBA", "question_count": 598},
                {"platform_code": "TYQW", "question_count": 620},
                {"platform_code": "TYQWA", "question_count": 357},
            ],
        )

    def test_complete_source_rankings_sum_to_one_and_define_the_exact_top_twenty(self) -> None:
        connection = duckdb.connect(
            str(ROOT / "data" / "catalog" / "cn_geo.duckdb"), read_only=True
        )
        try:
            payload = build_payload(connection)
            complete_rows = connection.execute(
                """
                SELECT scope_name, platform_code, source_id, rank,
                       weighted_share, citation_count, question_count,
                       scope_question_count
                FROM preference_source_rank_all
                ORDER BY scope_name, platform_code, rank
                """
            ).fetchall()
        finally:
            connection.close()

        complete_by_group: dict[tuple[str, str], list[tuple]] = {}
        for row in complete_rows:
            complete_by_group.setdefault((row[0], row[1]), []).append(row)
        self.assertEqual(len(complete_by_group), 16)
        for (scope, endpoint), group_rows in complete_by_group.items():
            self.assertAlmostEqual(sum(row[4] for row in group_rows), 1.0, places=9)
            self.assertEqual([row[3] for row in group_rows], list(range(1, len(group_rows) + 1)))
            self.assertTrue(all(row[6] <= row[7] for row in group_rows))
            expected_order = sorted(
                group_rows,
                key=lambda row: (-row[4], -row[5], row[2]),
            )
            self.assertEqual(group_rows, expected_order)
            payload_top_twenty = [
                row["source_id"]
                for row in payload["sourceTopRanks"]
                if row["scope"] == scope and row["platform_code"] == endpoint
            ]
            self.assertEqual(payload_top_twenty, [row[2] for row in group_rows[:20]])

    def test_preference_payload_is_deterministic_across_consecutive_builds(self) -> None:
        payload_hashes = []
        for _ in range(3):
            connection = duckdb.connect(
                str(ROOT / "data" / "catalog" / "cn_geo.duckdb"), read_only=True
            )
            try:
                payload = build_payload(connection)
            finally:
                connection.close()
            payload["meta"].pop("generated_at", None)
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            payload_hashes.append(hashlib.sha256(encoded).hexdigest())

        self.assertEqual(len(set(payload_hashes)), 1)

    def test_anchor_sources_are_the_exact_union_of_four_core_top_tens(self) -> None:
        anchor_codes = {"DB", "DOUBA", "DP", "DPA"}
        common_top_ten = {
            row["source_id"]
            for row in self.payload["sourceTopRanks"]
            if row["scope"] == "common"
            and row["platform_code"] in anchor_codes
            and row["rank"] <= 10
        }
        migration = self.payload["anchorSourceMigration"]
        migration_sources = {row["source_id"] for row in migration}

        self.assertEqual(migration_sources, common_top_ten)
        self.assertEqual(len(migration_sources), 17)
        self.assertEqual(len(migration), 17 * 8)
        self.assertEqual(self.payload["preferenceMeta"]["anchor_pool_size"], 17)
        self.assertEqual(self.payload["preferenceMeta"]["common_scope_question_count"], 334)
        self.assertTrue(all(row["scope"] == "common" for row in migration))
        for source_id in migration_sources:
            endpoint_rows = [row for row in migration if row["source_id"] == source_id]
            self.assertEqual(len(endpoint_rows), 8)
            self.assertEqual(
                {row["platform_code"] for row in endpoint_rows},
                {"DB", "DOUBA", "DP", "DPA", "TXYB", "TXYBA", "TYQW", "TYQWA"},
            )
            self.assertEqual(
                {row["anchor_top10_occurrences"] for row in endpoint_rows},
                {
                    sum(
                        row["rank"] is not None and row["rank"] <= 10
                        for row in endpoint_rows
                        if row["platform_code"] in anchor_codes
                    )
                },
            )
            for row in endpoint_rows:
                if row["rank"] is None:
                    self.assertEqual(row["share"], 0)
                    self.assertEqual(row["citation_count"], 0)
                    self.assertEqual(row["question_count"], 0)

        carryover = self.payload["preferenceMeta"]["anchor_top20_carryover"]
        self.assertEqual([row["platform_code"] for row in carryover], [
            "DB", "DOUBA", "DP", "DPA", "TXYB", "TXYBA", "TYQW", "TYQWA"
        ])
        self.assertEqual(
            {row["platform_code"]: row["source_count"] for row in carryover},
            {
                endpoint: sum(
                    row["platform_code"] == endpoint
                    and row["rank"] is not None
                    and row["rank"] <= 20
                    for row in migration
                )
                for endpoint in ("DB", "DOUBA", "DP", "DPA", "TXYB", "TXYBA", "TYQW", "TYQWA")
            },
        )
        source_summaries = {
            row["source_id"]: {
                "anchor_order": row["anchor_order"],
                "occurrences": row["anchor_top10_occurrences"],
                "average_share": row["anchor_average_share"],
            }
            for row in migration
        }
        for source_id, summary in source_summaries.items():
            core_rows = [
                row
                for row in migration
                if row["source_id"] == source_id
                and row["platform_code"] in anchor_codes
            ]
            self.assertAlmostEqual(
                summary["average_share"],
                sum(row["share"] for row in core_rows) / 4,
                places=12,
            )
        expected_anchor_order = sorted(
            source_summaries,
            key=lambda source_id: (
                -source_summaries[source_id]["occurrences"],
                -source_summaries[source_id]["average_share"],
                source_id,
            ),
        )
        self.assertEqual(
            expected_anchor_order,
            [
                source_id
                for source_id, _ in sorted(
                    source_summaries.items(),
                    key=lambda item: item[1]["anchor_order"],
                )
            ],
        )

    def test_anchor_scope_question_count_is_derived_from_common_scope(self) -> None:
        builder = (ROOT / "scripts" / "build_visual_report.py").read_text(encoding="utf-8")

        self.assertNotIn("coalesce(r.scope_question_count, 334)", builder)
        common_count = self.payload["preferenceMeta"]["common_scope_question_count"]
        self.assertTrue(
            all(
                row["scope_question_count"] == common_count
                for row in self.payload["anchorSourceMigration"]
            )
        )

    def test_preference_ui_linkage_contract_covers_groups_summaries_and_top_twenty(self) -> None:
        endpoint_order = ["DB", "DOUBA", "DP", "DPA", "TXYB", "TXYBA", "TYQW", "TYQWA"]
        migration = self.payload["anchorSourceMigration"]
        endpoint_rows = {
            row["platform_code"]: row
            for row in sorted(migration, key=lambda row: row["endpoint_order"])
        }

        self.assertEqual(list(endpoint_rows), endpoint_order)
        self.assertEqual(
            [endpoint_rows[code]["is_anchor_endpoint"] for code in endpoint_order],
            [True, True, True, True, False, False, False, False],
        )
        self.assertEqual(
            [endpoint_rows[code]["terminal"] for code in endpoint_order],
            ["web", "mobile", "web", "mobile", "web", "mobile", "web", "mobile"],
        )
        self.assertEqual(
            [endpoint_rows[code]["product_family"] for code in endpoint_order],
            ["豆包", "豆包", "DeepSeek", "DeepSeek", "腾讯元宝", "腾讯元宝", "千问", "千问"],
        )

        carryover = self.payload["preferenceMeta"]["anchor_top20_carryover"]
        self.assertEqual([row["platform_code"] for row in carryover], endpoint_order)
        self.assertEqual(len(carryover), 8)
        for item in carryover:
            self.assertEqual(
                item["source_count"],
                sum(
                    row["platform_code"] == item["platform_code"]
                    and row["rank"] is not None
                    and row["rank"] <= 20
                    for row in migration
                ),
            )

        self.assertTrue(
            all(row["source_category_l1_cn"] and row["source_type_cn"] for row in migration)
        )
        ranked_product_sources = {
            (row["product_family"], row["source_id"])
            for row in self.payload["sourceTopRanks"]
            if row["rank"] <= 20
        }
        tilt_product_sources = {
            (row["product_family"], row["source_id"])
            for row in self.payload["terminalTilt"]
        }
        self.assertEqual(len(ranked_product_sources), 101)
        self.assertLessEqual(ranked_product_sources, tilt_product_sources)

    def test_retired_concentration_payload_key_does_not_return(self) -> None:
        self.assertNotIn("preferenceConcentration", self.payload)

    def test_preference_type_mix_uses_governed_primary_categories(self) -> None:
        expected_categories = {
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
        mix = self.payload["preferenceTypeMix"]
        observed_categories = {
            row["source_category_l1_cn"]
            for row in mix
            if row["source_category_l1_cn"] != "信源未规范化"
        }

        self.assertEqual(observed_categories, expected_categories)
        self.assertTrue(all(isinstance(row["source_type_breakdown"], list) for row in mix))
        self.assertTrue(all(row["source_type_breakdown"] for row in mix))

        coverage = self.payload["classificationCoverage"]
        self.assertEqual(len(coverage), 9)
        self.assertEqual(coverage[-1]["platform_code"], "ALL")
        self.assertGreater(coverage[-1]["rule_share"], 0)
        for row in coverage:
            self.assertGreaterEqual(row["classification_coverage"], 0)
            self.assertLessEqual(row["classification_coverage"], 1)
            self.assertAlmostEqual(
                row["manual_share"]
                + row["rule_share"]
                + row["unclassified_share"]
                + row["unnormalized_share"],
                1.0,
                places=8,
            )

    def test_terminal_pair_summary_covers_each_product_once(self) -> None:
        rows = self.payload["terminalPairSummary"]
        questions = {row["product_family"]: row["common_question_count"] for row in rows}

        self.assertEqual(len(rows), 4)
        self.assertEqual({row["product_family"] for row in rows}, {"豆包", "DeepSeek", "腾讯元宝", "千问"})
        self.assertEqual(questions, {"豆包": 586, "DeepSeek": 586, "腾讯元宝": 597, "千问": 357})
        self.assertTrue(all(0 <= row["source_jaccard"] <= 1 for row in rows))
        self.assertTrue(all(0 <= row["qualified_source_jaccard"] <= 1 for row in rows))

    def test_positive_matrix_cells_have_terminal_link_details(self) -> None:
        linked = {
            (row["source_id"], row["product_family"])
            for row in self.payload["terminalTilt"]
        }
        missing = {
            (row["source_id"], row["product_family"])
            for row in self.payload["sourcePreference"]
            if row["weighted_share"] > 0
            and (row["source_id"], row["product_family"]) not in linked
        }

        self.assertEqual(missing, set())

    def test_every_top_twenty_source_has_terminal_tilt_link_details(self) -> None:
        top_twenty_links = {
            (row["product_family"], row["source_id"])
            for row in self.payload["sourceTopRanks"]
        }
        terminal_links = {
            (row["product_family"], row["source_id"])
            for row in self.payload["terminalTilt"]
        }

        self.assertEqual(top_twenty_links - terminal_links, set())

if __name__ == "__main__":
    unittest.main()
