# GEO Citation Lab

GEO Citation Lab 是一个面向 GEO 研究的公开资料仓库，后续定位为：

- `GEO 实验数据报告`：基于 ChatGPT、Google AI Overview / Gemini、Perplexity 的搜索触发、引用来源与页面吸收研究。
- `GEO / AEO / AI Search 论文合集`：持续收纳生成式搜索、AEO、GEO、AI 搜索引用机制与操纵风险相关论文。
- `CN-GEO 引用数据集`：覆盖 12 个国内 AI 平台、214,119 条原始引用记录，以及清洗仓库、分析集市和多维可视化报告。

仓库把可复查的数据、脚本、报告和论文资料集中在同一个入口下，方便做二次分析、引用和后续扩展。

本仓库实验数据、特征抽取与分析管线对应 arXiv 论文：[From Citation Selection to Citation Absorption: A Measurement Framework for Generative Engine Optimization Across AI Search Platforms](https://arxiv.org/abs/2604.25707)，PDF 版见 [arXiv PDF](https://arxiv.org/pdf/2604.25707)。

## Start Here

| 入口 | 路径 | 适合谁 |
| --- | --- | --- |
| 对应 arXiv 论文 | [From Citation Selection to Citation Absorption: A Measurement Framework for Generative Engine Optimization Across AI Search Platforms](https://arxiv.org/abs/2604.25707) / [PDF](https://arxiv.org/pdf/2604.25707) | 想引用或阅读本实验对应的正式论文 |
| CN-GEO 多维分析报告 | [GitHub Pages](https://yaojingang.github.io/geo-citation-lab/03-cn-geo-citation-dataset/reports/CN-GEO_%E5%A4%9A%E7%BB%B4%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E6%8A%A5%E5%91%8A.html) / [`reports/`](./03-cn-geo-citation-dataset/reports/) | 想浏览国内 AI 平台的信源、页面与内容特征分析 |
| CN-GEO 引用数据集 | [`03-cn-geo-citation-dataset/`](./03-cn-geo-citation-dataset/) | 想使用原始 JSONL、标准 Parquet、DuckDB、分析集市和清洗脚本 |
| GEO 实验数据报告 | [`01-geo-experiment-data-report/`](./01-geo-experiment-data-report/) | 想看 AI 搜索平台如何触发搜索、选择信源、吸收引用内容 |
| 论文 HTML 导航 | [GitHub Pages](https://yaojingang.github.io/geo-citation-lab/02-geo-aeo-ai-search-papers/) / [`index.html`](./02-geo-aeo-ai-search-papers/index.html) | 想按主题搜索、筛选和打开 53 篇 GEO / AEO / AI Search 论文 |
| 论文合集目录 | [`02-geo-aeo-ai-search-papers/`](./02-geo-aeo-ai-search-papers/) | 想查看论文 PDF、CSV 清单与校验文件 |
| 长版 HTML 报告 | [`01-geo-experiment-data-report/04-repet/final_report.html`](./01-geo-experiment-data-report/04-repet/final_report.html) | 想快速浏览完整实验报告 |
| 长版 Markdown 报告 | [`01-geo-experiment-data-report/04-repet/final_report.md`](./01-geo-experiment-data-report/04-repet/final_report.md) | 想在 GitHub 里直接按章节阅读正文 |
| PDF 版实验报告 | [`01-geo-experiment-data-report/04-repet/final_report.pdf`](./01-geo-experiment-data-report/04-repet/final_report.pdf) | 想下载、分享或打印实验报告 |
| 3 分钟摘要 | [`01-geo-experiment-data-report/QUICK_REPORT.md`](./01-geo-experiment-data-report/QUICK_REPORT.md) | 想先快速判断这份实验研究讲了什么 |

Live Site: [https://yaojingang.github.io/geo-citation-lab/](https://yaojingang.github.io/geo-citation-lab/)

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| [`01-geo-experiment-data-report/`](./01-geo-experiment-data-report/) | 原有 GEO 引用实验资产，已统一归类到一个大目录下 |
| [`01-geo-experiment-data-report/01-prompt/`](./01-geo-experiment-data-report/01-prompt/) | 602 条实验 Prompt |
| [`01-geo-experiment-data-report/02-data/`](./01-geo-experiment-data-report/02-data/) | 搜索层 CSV 与 72 维 citation-level 特征 CSV |
| [`01-geo-experiment-data-report/03-pipeline/`](./01-geo-experiment-data-report/03-pipeline/) | 解析、抓取、特征提取、统计分析脚本 |
| [`01-geo-experiment-data-report/04-repet/`](./01-geo-experiment-data-report/04-repet/) | 完整研究报告、HTML/PDF 导出与图表 |
| [`01-geo-experiment-data-report/05-kami-report/`](./01-geo-experiment-data-report/05-kami-report/) | 更适合展示/分享的摘要报告 |
| [`02-geo-aeo-ai-search-papers/`](./02-geo-aeo-ai-search-papers/) | 论文合集，按 10 个主题目录收纳 GEO / AEO / AI Search 相关论文 |
| [`03-cn-geo-citation-dataset/`](./03-cn-geo-citation-dataset/) | 国内 AI 引用数据集、数据仓库、质量报告、分析集市与可视化报告 |
| [`03-cn-geo-citation-dataset/data/records/`](./03-cn-geo-citation-dataset/data/records/) | 214,119 条原始引用记录，按 7 个分类层和 32 个分类组合分片 |
| [`03-cn-geo-citation-dataset/data/curated/`](./03-cn-geo-citation-dataset/data/curated/) | 问题、平台、信源、页面和引用观察标准表 |
| [`03-cn-geo-citation-dataset/data/marts/`](./03-cn-geo-citation-dataset/data/marts/) | 信源可见度、平台重合度、页面表现和数据质量分析集市 |

## CN-GEO 引用数据集 Snapshot

| 项目 | 数字 |
| --- | ---: |
| 原始引用记录 | 214,119 |
| 原始 JSONL 分片 | 64 |
| 规范问题 | 620 |
| AI 平台 | 12 |
| 规范信源 | 9,878 |
| 规范页面 | 107,659 |
| 额外精确重复记录 | 24,274 |

这套数据覆盖千问、豆包、腾讯元宝、DeepSeek、百度 AI、Kimi、文心和 AI 抖音等产品及其网页端、手机端形态。清洗层保留原始值与解析状态，分析层提供来源覆盖、跨平台共识、页面表现和确定性内容特征。

建议先打开 [多维数据分析报告](https://yaojingang.github.io/geo-citation-lab/03-cn-geo-citation-dataset/reports/CN-GEO_%E5%A4%9A%E7%BB%B4%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E6%8A%A5%E5%91%8A.html)，再阅读 [`数据集中文说明`](./03-cn-geo-citation-dataset/data/数据集中文说明.md) 和 [`清洗后数据使用说明`](./03-cn-geo-citation-dataset/data/清洗后数据使用说明.md)。完整验收结果见 [`quality_report.md`](./03-cn-geo-citation-dataset/data/quality/release_date=2026-07-14/quality_report.md)。

## 实验数据报告 Snapshot

| 项目 | 数字 |
| --- | ---: |
| 设计 Prompt 总数 | 602 |
| A/B/C/D 四层实验 | 432 / 60 / 60 / 50 |
| 平台数量 | 3 |
| 搜索层有效引用行数 | 21,143 |
| 引用影响力特征行数 | 23,745 |
| 特征维度 | 72 |
| 成功抓取的引用页面 | 18,151 |
| 抓取成功率 | 76.44% |

实验部分主要回答三个问题：

- 什么样的问题最容易触发 AI 去联网搜索？
- AI 搜索最爱选择什么样的来源网站？
- 什么样的页面会被 AI 深度吸收，而不只是“挂名引用”？

普通用户可以先看 [`QUICK_REPORT.md`](./01-geo-experiment-data-report/QUICK_REPORT.md)，想看完整论证再读 [`final_report.md`](./01-geo-experiment-data-report/04-repet/final_report.md) 或 [`final_report.pdf`](./01-geo-experiment-data-report/04-repet/final_report.pdf)。

## 论文合集 Snapshot

论文合集来自 `GEO_AI搜索_AEO_论文合集` 及 2026-06-20 近 30 天新增论文调研，当前按分类合并为 10 个主题目录，共 `53` 篇 PDF：

| 分类 | 主题 | PDF 数量 |
| --- | --- | ---: |
| [`01_GEO基础框架`](./02-geo-aeo-ai-search-papers/01_GEO基础框架/) | GEO 基础框架 | 4 |
| [`02_GEO方法优化`](./02-geo-aeo-ai-search-papers/02_GEO方法优化/) | GEO 方法优化 | 7 |
| [`03_GEO测量评估`](./02-geo-aeo-ai-search-papers/03_GEO测量评估/) | GEO 测量评估 | 8 |
| [`04_AI搜索实证`](./02-geo-aeo-ai-search-papers/04_AI搜索实证/) | AI 搜索实证 | 5 |
| [`05_AEO理论整合`](./02-geo-aeo-ai-search-papers/05_AEO理论整合/) | AEO 理论整合 | 5 |
| [`06_风险操纵`](./02-geo-aeo-ai-search-papers/06_风险操纵/) | 风险、操纵与对抗 | 10 |
| [`07_垂直多模态`](./02-geo-aeo-ai-search-papers/07_垂直多模态/) | 垂直场景与多模态 | 5 |
| [`08_AI搜索架构与AgenticSearch`](./02-geo-aeo-ai-search-papers/08_AI搜索架构与AgenticSearch/) | AI 搜索架构与 Agentic Search | 4 |
| [`09_RAG检索优化`](./02-geo-aeo-ai-search-papers/09_RAG检索优化/) | RAG 检索优化 | 3 |
| [`10_搜索评估治理`](./02-geo-aeo-ai-search-papers/10_搜索评估治理/) | 搜索评估与治理 | 2 |

论文合集可通过 [HTML 导航页](https://yaojingang.github.io/geo-citation-lab/02-geo-aeo-ai-search-papers/) 搜索和筛选，完整 Markdown 清单见 [`02-geo-aeo-ai-search-papers/README.md`](./02-geo-aeo-ai-search-papers/README.md)。源目录中两份 `GEO_AI搜索_AEO_论文整理说明.docx` 内容相同，本仓库按 SHA-256 去重保留一份，并保留 [`论文清单.csv`](./02-geo-aeo-ai-search-papers/00_资料说明/论文清单.csv) 与 [`checksums.sha256`](./02-geo-aeo-ai-search-papers/00_资料说明/checksums.sha256) 方便复核。

## 如何阅读

1. 先读 [`01-geo-experiment-data-report/QUICK_REPORT.md`](./01-geo-experiment-data-report/QUICK_REPORT.md)，快速理解实验结论。
2. 再读 [`01-geo-experiment-data-report/04-repet/final_report.md`](./01-geo-experiment-data-report/04-repet/final_report.md)，查看完整方法、图表和章节论证。
3. 打开 [`01-geo-experiment-data-report/02-data/features_all_platforms_72.csv`](./01-geo-experiment-data-report/02-data/features_all_platforms_72.csv)，筛选你关心的字段。
4. 打开 [论文 HTML 导航](https://yaojingang.github.io/geo-citation-lab/02-geo-aeo-ai-search-papers/) 或阅读 [`02-geo-aeo-ai-search-papers/README.md`](./02-geo-aeo-ai-search-papers/README.md)，按主题进入论文 PDF。

## 公开仓库运行方式

本仓库已将脚本改为从环境变量读取密钥，避免把私钥直接放进 GitHub。

```bash
cd 01-geo-experiment-data-report
cp .env.example .env
```

常见重跑方式：

```bash
cd 01-geo-experiment-data-report/03-pipeline
python3 analyze_influence.py \
  --input ../02-data/features_all_platforms_72.csv \
  --output ../04-repet/citation_influence_report.md
```

```bash
cd 01-geo-experiment-data-report/04-repet
python3 build_self_contained_html.py
```
