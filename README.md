# GEO Citation Lab

> **用可复查的数据，研究 AI 搜索如何选择信源、吸收内容与呈现实体。**
>
> *Open empirical resources for studying how generative search selects sources, absorbs content, and surfaces entities.*

GEO Citation Lab 是一个面向 AI 搜索引用机制的公开实证研究工作台。仓库围绕两条研究主线，整理跨平台引用实验与中文生成式搜索数据，并提供分析脚本、数据仓库、可视化报告、主题论文导航和 GEO 在线能力测试。

`提问 → 搜索触发 → 信源选择 → 内容吸收 → 实体曝光 → 跨平台与跨终端差异`

## 在 AI 客户端一句话安装

把下面这句话发送给具有网络和本地文件权限的 AI 客户端：

> 安装 GEO Citation Lab 的轻量阅读版并打开：
> https://github.com/yaojingang/geo-citation-lab

AI 客户端会下载版本固定的 `v0.1.0` Viewer、验证 SHA-256、解压并打开
`index.html`。Viewer
包含自包含研究报告和可搜索论文目录，体积上限为 15 MiB，无需 Git、Python、Node
或 Docker。论文 PDF 与完整研究数据保留在线入口。

当前 `deploy/manifest.json` 中的 `release_status` 为 `released`，安装器使用
`v0.1.0` 的固定 Release 资产地址。

| 你对 AI 说 | 执行结果 |
| --- | --- |
| “安装并打开 GEO Citation Lab” | 安装轻量 Viewer，并在默认浏览器打开 |
| “把 GEO Citation Lab Viewer 部署到我的 GitHub Pages” | 保留许可与署名文件，确认目标仓库与公开可见性后运行 Pages 发布流程 |
| “下载完整数据并准备分析环境” | 提示约 568 MiB 体积，再克隆仓库并配置 Python 3.11 与 `uv` |

网页聊天客户端通常没有本地文件权限，可以直接使用
[在线首页](https://yaojingang.github.io/geo-citation-lab/)。本地 AI 客户端应遵循
[`AGENTS.md`](./AGENTS.md) 和
[`deploy/manifest.json`](./deploy/manifest.json) 中的安装契约。

| 研究资产 | 当前规模 |
| --- | ---: |
| CN-GEO 原始引用记录 | 214,119 条 |
| 国内 AI 平台与终端代码 | 12 个 |
| 跨平台实验 Prompt | 602 条 |
| GEO / AEO / AI Search 论文 PDF | 54 篇 |
| GEO 在线能力测试 | 30 题、100 分、六维诊断 |

[在线参加 GEO 能力测试](https://ai.laoyao.cn/geo/) · [查看 CN-GEO 分析报告](https://yaojingang.github.io/geo-citation-lab/03-cn-geo-citation-dataset/reports/final/CN-GEO_%E5%A4%9A%E7%BB%B4%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E6%8A%A5%E5%91%8A.html) · [读 3 分钟实验摘要](./01-geo-experiment-data-report/QUICK_REPORT.md) · [浏览论文导航](https://yaojingang.github.io/geo-citation-lab/02-geo-aeo-ai-search-papers/) · [打开在线首页](https://yaojingang.github.io/geo-citation-lab/)

## 这套研究在看什么

AI 搜索中的可见性包含多个环节。一条内容可能进入检索候选、出现在引用列表、参与答案生成，或进一步影响品牌和实体的呈现。仓库用两套实证研究观察这条链路。

- [`01-geo-experiment-data-report/`](./01-geo-experiment-data-report/) 研究 ChatGPT、Google AI Overview / Gemini 和 Perplexity 如何触发搜索、选择信源并吸收页面内容。
- [`03-cn-geo-citation-dataset/`](./03-cn-geo-citation-dataset/) 整理国内 AI 引用记录，并从信源生态、页面特征、实体曝光和 Web / App 差异等角度提供可查询的数据与报告。
- [`02-geo-aeo-ai-search-papers/`](./02-geo-aeo-ai-search-papers/) 为上述问题提供文献背景，覆盖 GEO 方法、测量评估、AI 搜索实证、风险操纵、Agentic Search 和 RAG 等主题。
- [`geo-assessment/`](./geo-assessment/) 将论文、跨平台实验和 CN-GEO 数据中的关键结论转化为 30 道场景题。题目以国内应用为主，面向刚接触 GEO 的用户，并生成六维能力报告。

## 三个研究发现

**引用广度和吸收深度需要分别测量。** 在跨平台实验中，Perplexity 和 Google 平均引用更多信源，ChatGPT 引用较少，但成功抓取页面的平均引用影响力更高。完整口径见论文 [From Citation Selection to Citation Absorption](https://arxiv.org/abs/2604.25707)。

**内容匹配度、结构和证据密度与吸收深度相关。** 高影响力页面通常更长、分段更清楚，也更常包含定义、数字、对比和操作步骤。单独采用 Q&A 格式没有表现出吸收优势。这些结果来自静态样本中的描述性统计和相关性分析。

**同一产品的 Web 与 App 需要分开观察。** 中文生成式搜索研究发现，同一平台不同终端的信源集合存在系统差异，界面类型会影响跨平台比较。完整结果见论文 [What Do Chinese-Language Generative Search Engines Cite and Surface?](https://arxiv.org/abs/2607.15771)。

## 按你的目的进入

| 你想做什么 | 建议入口 |
| --- | --- |
| 了解 GEO 研究结论 | [CN-GEO 在线分析报告](https://yaojingang.github.io/geo-citation-lab/03-cn-geo-citation-dataset/reports/final/CN-GEO_%E5%A4%9A%E7%BB%B4%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E6%8A%A5%E5%91%8A.html) · [跨平台实验摘要](./01-geo-experiment-data-report/QUICK_REPORT.md) |
| 使用数据或继续分析 | [CN-GEO 数据集说明](./03-cn-geo-citation-dataset/) · [清洗后数据使用说明](./03-cn-geo-citation-dataset/data/清洗后数据使用说明.md) |
| 阅读论文或引用研究 | [两篇实证论文](#两条实证研究主线) · [54 篇论文导航](https://yaojingang.github.io/geo-citation-lab/02-geo-aeo-ai-search-papers/) |
| 复查实验与处理方法 | [跨平台实验管线](./01-geo-experiment-data-report/03-pipeline/) · [CN-GEO 构建脚本](./03-cn-geo-citation-dataset/scripts/) |
| 测试 GEO 理解程度 | [在线测试](https://ai.laoyao.cn/geo/) · [系统源码与本地运行](./geo-assessment/) |

只想阅读结论时，可以直接使用 GitHub Pages，无需克隆完整仓库。数据分析与复算方式分别写在两个研究目录的 README 中。

## 两条实证研究主线

### 跨平台引用选择与吸收

这套实验覆盖 `602` 条受控 Prompt、`21,143` 条有效搜索层引用和 `23,745` 条 citation-level 特征记录，并从 `18,151` 个成功抓取页面中提取 `72` 维特征。

研究将生成式搜索拆为两个可观察结果：

- **引用选择**：平台是否触发搜索，以及哪些信源进入引用列表。
- **引用吸收**：引用页面在语言、证据、结构和事实层面对最终答案的参与程度。

对应论文：

> Zhang Kai, He Xinyue, Yao Jingang. [From Citation Selection to Citation Absorption: A Measurement Framework for Generative Engine Optimization Across AI Search Platforms](https://arxiv.org/abs/2604.25707). arXiv:2604.25707, 2026.

研究材料包括 [Prompt 与数据](./01-geo-experiment-data-report/)、[完整 Markdown 报告](./01-geo-experiment-data-report/04-repet/final_report.md) 和 [在线 HTML 报告](https://yaojingang.github.io/geo-citation-lab/01-geo-experiment-data-report/04-repet/final_report.html)。

### 中文生成式搜索与 CN-GEO

CN-GEO 当前发布版包含 `214,119` 条原始引用记录、`64` 个 JSONL 分片、`9,878` 个规范信源和 `107,659` 个规范页面。仓库同时提供标准 Parquet 表、自包含 DuckDB、页面特征、分析集市、质量报告和可视化报告。

对应论文：

> Tao Zhen, Yue Liu, Gege Zhang, Yixuan Niu. [What Do Chinese-Language Generative Search Engines Cite and Surface? A Large-Scale Empirical Study](https://arxiv.org/abs/2607.15771). arXiv:2607.15771, 2026.

论文分析四个中文大模型产品的八个 Web / App 界面，主要研究引用行为、实体曝光和跨界面一致性。

## 论文库

论文库按 10 个主题收录 54 篇 PDF，包含 GEO 基础框架、方法优化、测量评估、AI 搜索实证、AEO 理论、风险与操纵、垂直多模态、Agentic Search、RAG 和搜索评估治理。

- [在线搜索与筛选](https://yaojingang.github.io/geo-citation-lab/02-geo-aeo-ai-search-papers/)
- [完整论文清单](./02-geo-aeo-ai-search-papers/README.md)
- [来源、文件校验与整理说明](./02-geo-aeo-ai-search-papers/00_资料说明/)

## 来源、引用与使用边界

| 资产 | 来源与本仓库角色 |
| --- | --- |
| 跨平台实验 | 本仓库保存实验 Prompt、数据、管线与报告，并作为 arXiv:2604.25707 的公开数据和分析入口。 |
| CN-GEO | 基于 [WENDAOstudy/cn-geo-citation-dataset](https://github.com/WENDAOstudy/cn-geo-citation-dataset)（[CC BY 4.0](https://github.com/WENDAOstudy/cn-geo-citation-dataset/blob/main/LICENSE.md)）整理，提供清洗、查询和可视化分析。 |
| 论文 PDF | 论文著作权归原作者或出版方。仓库提供主题整理、来源链接、文件清单和校验信息。 |

使用数据前请留意这些边界：

- 跨平台实验来自一次静态研究快照，当前没有为每条记录提供统一采集时间戳。
- CN-GEO 原始层缺少完整回答、回答批次、模型版本和采集时间。来源覆盖与跨平台共识可以直接研究，趋势、情感、严格引用排名和品牌推荐率需要更多回答级数据。
- 不同论文、原始数据和清洗仓库采用各自的样本范围与处理口径。引用数字时，以对应论文或数据版本的说明为准。
- 本仓库采用分范围许可：代码和自动化文件使用 MIT，自有报告、文档、可视化和目录元数据使用 CC BY 4.0。论文 PDF、来源数据和其他第三方材料维持原许可。

完整的数据限制和验收结果见 [CN-GEO 数据集中文说明](./03-cn-geo-citation-dataset/data/数据集中文说明.md) 与 [数据质量报告](./03-cn-geo-citation-dataset/data/quality/release_date=2026-07-14/quality_report.md)。

## 发布轻量 Viewer

维护者可以在本地构建并验证发布包：

```bash
python3 scripts/build_viewer.py --package
python3 scripts/verify_distribution.py
```

推送 `main` 后，[`publish.yml`](./.github/workflows/publish.yml) 会部署轻量 Viewer 到
GitHub Pages。推送与 `deploy/manifest.json` 中 `distribution_version` 一致的 `v*`
标签后，同一工作流会发布 Viewer ZIP、SHA-256、manifest 和两个安装器。

准备下一个版本时，维护者先把 `release_status` 设为 `pending`，并同步更新
`distribution_version`、安装器固定 URL 和 README 中的版本。发布提交再将状态设为
`released`，通过验证后创建匹配版本的标签。安装器和 manifest 都使用版本固定的
Release URL，避免 `main` 与 `latest` 在发布窗口内发生版本漂移。

首次启用工作流部署时，需要在仓库 **Settings → Pages → Build and deployment** 中把
Source 设置为 **GitHub Actions**。发布包的第三方材料边界见
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)。

仓库采用分范围许可。代码、安装器、工作流和 Skill 使用
[MIT](./LICENSE-CODE)；本仓库原创报告、文档、可视化和目录元数据使用
[CC BY 4.0](./LICENSE-CONTENT)。用户可以复制、修改和公开部署 Viewer，发布时需要
保留 [`LICENSE`](./LICENSE)、两个分范围许可文件与
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md)，并注明来源和修改情况。论文
PDF、来源数据和其他第三方材料维持各自的原始条款。

## 关注更新与参与

仓库后续更新主要集中在数据版本、分析报告和论文库。可以通过 GitHub 的 **Star** 收藏项目，通过 **Watch** 关注提交与讨论。

欢迎在 [Issues](https://github.com/yaojingang/geo-citation-lab/issues) 中提交：

- 数据字段、清洗规则或报告口径问题；
- 可复现的分析结果与修正建议；
- 值得补充的 GEO / AEO / AI Search 论文。
