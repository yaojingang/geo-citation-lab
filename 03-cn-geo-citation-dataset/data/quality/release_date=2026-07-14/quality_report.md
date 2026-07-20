# 数据质量报告：2026-07-14

- 验收状态：通过
- 原始记录：214,119 条
- 规范问题：620 个
- 规范信源：9,878 个
- 规范页面：107,659 个
- 额外精确重复：24,274 条
- 原始分片校验：64 个全部通过 SHA-256 校验

## 自动验收

- 通过：`source_type_reference_passed_preflight`
- 通过：`raw_shard_checksums_match_manifest`
- 通过：`raw_and_curated_row_counts_match`
- 通过：`citation_ids_are_unique`
- 通过：`exact_duplicate_count_matches_manifest`
- 通过：`all_platform_codes_are_mapped`
- 通过：`all_legacy_category_pairs_are_mapped`
- 通过：`each_legacy_prompt_id_maps_to_one_question`
- 通过：`all_nonempty_publication_values_are_classified`
- 通过：`responses_table_is_intentionally_empty`

## 已知限制

- 旧数据缺少可靠的回答批次边界，responses 当前保留空表结构，citation_observations.response_id 保持为空。
- source_types.csv 保留逐域人工复核证据，政府域名使用确定性后缀规则，其余长尾信源继续标记为未分类。
- 页面特征来自可复现的确定性规则，当前未生成向量、品牌实体和情感特征。

完整机器可读结果见同目录 `quality_report.json`。
