# GEO 实验数据报告说明

本目录是 `geo-citation-lab` 中专门存放 GEO 引用机制实验数据、处理脚本与研究报告的资料区。它对应一次面向 `ChatGPT`、`Google AI Overview / Gemini`、`Perplexity` 的数据研究，目标是把 AI 搜索中的 `触发搜索 -> 选择信源 -> 吸收引用内容` 这条链路拆成可以复查的数据、代码和报告。

## 对应论文

- 论文名称：[From Citation Selection to Citation Absorption: A Measurement Framework for Generative Engine Optimization Across AI Search Platforms](https://arxiv.org/abs/2604.25707)
- arXiv 页面：[https://arxiv.org/abs/2604.25707](https://arxiv.org/abs/2604.25707)
- PDF：[https://arxiv.org/pdf/2604.25707](https://arxiv.org/pdf/2604.25707)

## 研究目标

这次数据研究主要回答三个问题：

- 什么样的问题最容易触发 AI 去联网搜索？
- AI 搜索平台会优先选择什么样的来源网站？
- 什么样的页面会被 AI 深度吸收，而不只是被挂名引用？

仓库保留了实验 Prompt、搜索结果 CSV、citation-level 特征表、处理脚本、长版报告和摘要报告，方便后续继续复算、二次分析或引用结论。

## 数据快照

| 项目 | 数字 |
| --- | ---: |
| 设计 Prompt 总数 | 602 |
| A/B/C/D 四层实验 | 432 / 60 / 60 / 50 |
| 平台数量 | 3 |
| 搜索层原始结果行数（清洗后） | 21,181 |
| 搜索层有效引用行数 | 21,143 |
| 引用影响力特征行数 | 23,745 |
| 特征维度 | 72 |
| 成功抓取的引用页面 | 18,151 |
| 抓取成功率 | 76.44% |

## 实验设计

这套实验把 Prompt 分成四层，用来观察不同提问方式、语言、任务强度和真实场景对 AI 搜索行为的影响：

- `A 层`：432 条主实验 Prompt，控制任务类型、触发强度、时效性、行业与子任务。
- `B 层`：60 条风格对照 Prompt，比较自然问法、要求来源、专家角色三种包装方式。
- `C 层`：60 条中英双语对照 Prompt，观察不同语言环境下的搜索强度与信源偏好。
- `D 层`：50 条极端与真实场景 Prompt，覆盖高风险、模糊、多约束和长决策型问题。

## 目录说明

| 路径 | 作用 |
| --- | --- |
| [`01-prompt/`](./01-prompt/) | 602 条实验 Prompt |
| [`02-data/`](./02-data/) | 搜索层 CSV 与 72 维 citation-level 特征 CSV |
| [`03-pipeline/`](./03-pipeline/) | 解析、抓取、特征提取、统计分析脚本 |
| [`04-repet/`](./04-repet/) | 完整研究报告、HTML/PDF 导出与图表 |
| [`05-kami-report/`](./05-kami-report/) | 更适合展示/分享的摘要报告 |
| [`QUICK_REPORT.md`](./QUICK_REPORT.md) | 给普通用户的 3 分钟速读版 |
| [`.env.example`](./.env.example) | 环境变量模板 |

## 阅读顺序

1. 先读 [`QUICK_REPORT.md`](./QUICK_REPORT.md)，快速理解这次数据研究的核心结论。
2. 再读 [`04-repet/final_report.md`](./04-repet/final_report.md)，查看完整方法、图表和章节论证。
3. 打开 [`02-data/features_all_platforms_72.csv`](./02-data/features_all_platforms_72.csv)，筛选你关心的字段。
4. 阅读 [`03-pipeline/citation_features.py`](./03-pipeline/citation_features.py) 和 [`03-pipeline/analyze_influence.py`](./03-pipeline/analyze_influence.py)，复查特征计算和影响力分析逻辑。

## 常见重跑方式

本目录脚本通过环境变量读取密钥。复制环境变量模板后再按需要填入本地密钥：

```bash
cp .env.example .env
```

重跑影响力分析：

```bash
cd 03-pipeline
python3 analyze_influence.py \
  --input ../02-data/features_all_platforms_72.csv \
  --output ../04-repet/citation_influence_report.md
```

重新生成自包含 HTML 报告：

```bash
cd 04-repet
python3 build_self_contained_html.py
```

## 数据说明与已知 caveats

- `chatgpt_results_with_prompt.csv` 原始文件中混入了 `16` 行重复表头，统计时需要先清洗。
- ChatGPT 搜索层的 `A_news`、`A_technology` 在原始文件里命名为 `Anews*`、`Atechnology*`，需要先做命名归一化。
- ChatGPT 搜索层清洗后覆盖 `587` 个 Prompt，仍缺 `15` 个 Prompt 输出。
- `国家(Country)` 和 `语言(Language)` 中存在大量 `unknown` 或 `WW`，因此地区/语言占比最好同时给出“可识别样本口径”。
- `网站类型` 字段里存在少量噪声值，例如 `成功`，这类值更适合在公开版里再做一次标准化。
- 仓库当前没有给每条记录附统一采集时间戳；它更适合作为一次静态研究快照来理解，而不是实时监控数据源。

## 作者与贡献

- 张凯：提出研究想法与需求，定义分析目标与相关规则；微信号：`seermartech`
- 贺欣悦：负责源代码实现、数据采集与清洗、初稿撰写；清华大学本科，清华大学与华盛顿大学 `GIX` 项目的双学位硕士生；GitHub 主页：[shirley-goose](https://github.com/shirley-goose)
- 姚金刚：负责开源整理、二次报告解读、应用场景梳理与论文合集归档
