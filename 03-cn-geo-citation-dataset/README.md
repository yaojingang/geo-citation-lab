# CN-GEO Citation Dataset

这是一套面向国内生成式 AI 引用研究的公开数据资产，覆盖 12 个 AI 平台、214,119 条原始引用记录，以及清洗后的问题、信源、页面、引用观察、页面特征和分析集市。

- 数据版本：`2.0.1`
- 数据发布日期：`2026-07-14`
- 清洗流水线版本：`1.0.1`

## 快速入口

| 入口 | 用途 |
| --- | --- |
| [多维数据分析报告](./reports/CN-GEO_多维数据分析报告.html) | 浏览平台、信源、页面与内容特征的可视化分析 |
| [数据集中文说明](./data/数据集中文说明.md) | 了解原始数据、字段、平台映射、分类与使用边界 |
| [清洗后数据使用说明](./data/清洗后数据使用说明.md) | 了解标准表、分析集市、DuckDB 查询入口与清洗规则 |
| [数据质量报告](./data/quality/release_date=2026-07-14/quality_report.md) | 查看分片校验、自动验收结果与已知限制 |
| [数据清单](./data/manifest.json) | 复核版本、记录规模、分片路径和 SHA-256 |

GitHub 代码页会把 HTML 文件显示为源码。在线浏览请使用 [GitHub Pages 报告入口](https://yaojingang.github.io/geo-citation-lab/03-cn-geo-citation-dataset/reports/CN-GEO_%E5%A4%9A%E7%BB%B4%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E6%8A%A5%E5%91%8A.html)。

## 数据规模

| 项目 | 数量 |
| --- | ---: |
| 原始引用记录 | 214,119 |
| 原始 JSONL 分片 | 64 |
| 原始 JSONL 体量 | 约 267 MB |
| 完整数据目录体量 | 约 409 MB |
| 规范问题 | 620 |
| AI 平台 | 12 |
| 规范信源 | 9,878 |
| 规范页面 | 107,659 |
| 额外精确重复记录 | 24,274 |

每条原始记录表示某个 AI 平台回答一个问题时出现的一条外部引用。当前数据适合研究来源覆盖、跨平台共识、页面表现和内容特征。品牌推荐率、情感倾向、严格引用排名和趋势分析需要完整回答、回答批次、模型版本与采集时间。

## 目录结构

```text
03-cn-geo-citation-dataset/
├── data/
│   ├── records/       原始 JSONL 分片
│   ├── reference/     平台、分类和信源类型字典
│   ├── contracts/     数据契约与主键规则
│   ├── curated/       清洗后的标准 Parquet 表
│   ├── features/      可复现的页面特征
│   ├── marts/         常用分析集市
│   ├── quality/       发布质量报告
│   └── catalog/       常用表的自包含 DuckDB 查询目录
├── reports/           可视化报告、生成模板和固定版本前端依赖
├── schema/            原始记录 JSON Schema
├── scripts/           数据仓库与报告构建脚本
└── tests/             数据仓库与报告验收测试
```

## 本地重建与验收

项目使用 `uv` 管理固定依赖：

```bash
uv sync
uv run python scripts/build_data_warehouse.py --force
uv run python scripts/build_visual_report.py
uv run python -m unittest discover -s tests -v
```

数据仓库脚本会核对 64 个原始分片的 SHA-256，并在临时目录完成全量构建和验收后替换派生目录。已有派生目录存在时，需要显式提供 `--force`。

公开发布前已移除 25 条引用 URL 中的用户信息，并重算对应记录哈希。构建预检会拒绝任何仍携带 URL 用户信息的原始记录。

## DuckDB 查询示例

```python
import duckdb

con = duckdb.connect("data/catalog/cn_geo.duckdb", read_only=True)
top_sources = con.sql("""
    SELECT domain,
           sum(deduplicated_citation_count) AS citations,
           count(DISTINCT platform_code) AS platforms
    FROM source_visibility
    GROUP BY domain
    ORDER BY citations DESC
    LIMIT 20
""").fetchall()
```

查询目录已将常用维表和分析集市固化在数据库内，复制或移动 `cn_geo.duckdb` 后仍可直接查询。全量引用观察保留在分区 Parquet 中，避免重复存储约 81 MB 的事实表。完整字段口径、可用性规则和分析建议见 [清洗后数据使用说明](./data/清洗后数据使用说明.md)。
