# CN-GEO Citation Dataset

数据分为原始层、标准层、特征层和分析集市。中文字段说明见 [数据集中文说明](数据集中文说明.md)，清洗结果和使用方法见 [清洗后数据使用说明](清洗后数据使用说明.md)。

## 目录

```text
data/
├── records/       原始 JSONL，保持不变
├── reference/     平台、分类和信源类型字典
├── contracts/     数据契约与主键规则
├── curated/       清洗后的标准 Parquet 表
├── features/      可复现的页面特征
├── marts/         常用分析集市
├── quality/       每次发布的质量报告
└── catalog/       常用表的自包含 DuckDB 查询目录
```

原始发布包含 214,119 条引用记录和 64 个 JSONL 分片。清洗流水线在构建前后核对每个分片的 SHA-256，完整验收结果见 `quality/release_date=2026-07-14/quality_report.md`。

## 重建

项目使用 `uv` 管理固定依赖：

```bash
uv sync
uv run python scripts/build_data_warehouse.py --force
uv run python -m unittest discover -s tests -v
```

省略 `--force` 时，程序发现已有派生目录会直接停止。带上 `--force` 后，程序先在临时目录完成全量构建和验收，再安全替换旧的派生目录。构建锁会阻止两个清洗任务同时替换目录，安装验收失败时会自动恢复上一版。
