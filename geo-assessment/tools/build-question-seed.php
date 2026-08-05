<?php

declare(strict_types=1);

$appRoot = dirname(__DIR__);
$repoRoot = dirname($appRoot);
$designPath = $repoRoot . '/04-geo-online-assessment-design/02-30道GEO测试题与证据设计.md';
$catalogPath = $repoRoot . '/02-geo-aeo-ai-search-papers/00_资料说明/论文清单.csv';
$outputPath = $appRoot . '/database/seeds/geo-30-v1.1.json';

$markdown = file_get_contents($designPath);
if (!is_string($markdown)) {
    throw new RuntimeException('无法读取题库设计文档。');
}

preg_match_all('/^\| (Q\d{2}) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$/mu', $markdown, $blueprintRows, PREG_SET_ORDER);
$blueprint = [];
foreach ($blueprintRows as $row) {
    $blueprint[$row[1]] = [
        'cognitive_level' => trim($row[2]),
        'misconception_tag' => trim($row[3]),
    ];
}

$sourceRows = [];
$handle = fopen($catalogPath, 'rb');
if ($handle === false) {
    throw new RuntimeException('无法读取论文清单。');
}
$headers = fgetcsv($handle, escape: '');
if (!is_array($headers)) {
    throw new RuntimeException('论文清单缺少表头。');
}
$headers[0] = preg_replace('/^\xEF\xBB\xBF/', '', (string) $headers[0]);
while (($row = fgetcsv($handle, escape: '')) !== false) {
    $sourceRows[] = array_combine($headers, $row);
}
fclose($handle);

if (count($sourceRows) !== 54) {
    throw new RuntimeException('论文清单必须恰好包含 54 篇。');
}

$sources = [];
foreach ($sourceRows as $index => $row) {
    $sources[] = [
        'id' => sprintf('P%02d', $index + 1),
        'title' => $row['论文题名'],
        'kind' => 'paper',
        'citation' => trim($row['年份'] . ' · ' . $row['来源']),
        'url' => $row['URL'] !== '' ? $row['URL'] : null,
    ];
}
$sources[] = [
    'id' => 'D01',
    'title' => '跨平台引用选择与答案吸收实验数据',
    'kind' => 'dataset',
    'citation' => '602 条 Prompt，3 个海外生成式搜索平台，证据冻结于 2026-08-04',
    'url' => 'https://github.com/yaojingang/geo-citation-lab/tree/main/01-geo-experiment-data-report',
];
$sources[] = [
    'id' => 'D02',
    'title' => 'CN-GEO 中文生成式搜索引用数据集',
    'kind' => 'dataset',
    'citation' => 'v2.0.1，214,119 条原始引用记录，证据冻结于 2026-08-04',
    'url' => 'https://github.com/yaojingang/geo-citation-lab/tree/main/03-cn-geo-citation-dataset',
];

preg_match_all('/^### (Q\d{2}) · ([^\n]+)\n\n(.*?)(?=^### Q\d{2} · |^## 7\.)/msu', $markdown, $matches, PREG_SET_ORDER);
$questions = [];
foreach ($matches as $match) {
    $code = $match[1];
    $title = trim($match[2]);
    $block = $match[3];

    if (!preg_match('/\*\*类型\*\*：(单选|多选) \*\*难度\*\*：(基础|进阶|挑战) \*\*分值\*\*：(\d+) \*\*预计\*\*：(\d+) 秒/u', $block, $meta)) {
        throw new RuntimeException("{$code} 的元数据无法解析。");
    }

    $afterMeta = preg_split('/\*\*类型\*\*.*?\n\n/su', $block, 2)[1] ?? '';
    $prompt = trim(preg_split('/^A\. /mu', $afterMeta, 2)[0] ?? '');
    preg_match_all('/^([A-E])\. (.+)$/mu', $block, $choiceMatches, PREG_SET_ORDER);

    $correctText = extractField($block, '正确答案');
    $correctCodes = preg_split('/、/u', $correctText) ?: [];
    $correctCodes = array_values(array_filter(array_map('trim', $correctCodes)));
    $explanation = extractField($block, '解析');
    $principle = extractField($block, '底层原理');
    $distractorRationale = extractField($block, '干扰项说明');
    $sourceRefs = preg_split('/、/u', extractField($block, '来源')) ?: [];
    $sourceRefs = array_values(array_filter(array_map('trim', $sourceRefs)));

    $choices = [];
    foreach ($choiceMatches as $choice) {
        $isCorrect = in_array($choice[1], $correctCodes, true);
        $choices[] = [
            'code' => $choice[1],
            'text' => trim($choice[2]),
            'is_correct' => $isCorrect,
            'rationale' => $isCorrect
                ? '该选项与正确答案集合一致。' . $principle
                : $distractorRationale,
        ];
    }

    $number = (int) substr($code, 1);
    [$dimension, $dimensionLabel] = match (true) {
        $number <= 5 => ['mechanism', '底层机制与范式'],
        $number <= 10 => ['content', '内容与优化'],
        $number <= 15 => ['measurement', '测量与实验推理'],
        $number <= 20 => ['overseas', '海外引用特征'],
        $number <= 25 => ['domestic', '国内引用特征'],
        default => ['governance', '风险治理与多模态'],
    };

    $questions[] = [
        'code' => $code,
        'title' => $title,
        'sort_order' => $number,
        'type' => $meta[1] === '单选' ? 'single' : 'multiple',
        'dimension' => $dimension,
        'dimension_label' => $dimensionLabel,
        'difficulty' => match ($meta[2]) {'基础' => 'basic', '进阶' => 'advanced', default => 'challenge'},
        'difficulty_label' => $meta[2],
        'cognitive_level' => $blueprint[$code]['cognitive_level'] ?? '理解',
        'prompt' => $prompt,
        'choices' => $choices,
        'correct_codes' => $correctCodes,
        'weight' => (int) $meta[3],
        'explanation' => $explanation,
        'principle' => $principle,
        'misconception_tag' => $blueprint[$code]['misconception_tag'] ?? $title,
        'source_refs' => $sourceRefs,
        'expected_seconds' => (int) $meta[4],
    ];
}

if (count($questions) !== 30) {
    throw new RuntimeException('必须从文档生成 30 道题。');
}

$payload = [
    'schema_version' => 'question-set-v1',
    'set' => [
        'version' => 'geo-30-v1.1',
        'title' => 'GEO 在线能力测试',
        'description' => '基于 54 篇论文、海外实验与 CN-GEO 数据集的 30 道研究型评测。',
        'total_points' => 100,
        'time_limit_seconds' => 1800,
        'scoring_version' => 'score-v1.1',
        'evidence_frozen_at' => '2026-08-04',
    ],
    'sources' => $sources,
    'questions' => $questions,
];

$json = json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR) . "\n";
if (file_put_contents($outputPath, $json) === false) {
    throw new RuntimeException('题库种子写入失败。');
}

fwrite(STDOUT, sprintf("已生成 %d 道题、%d 个来源：%s\n", count($questions), count($sources), $outputPath));

function extractField(string $block, string $label): string
{
    if (!preg_match('/\*\*' . preg_quote($label, '/') . '\*\*：([^\n]+)/u', $block, $match)) {
        throw new RuntimeException("题目缺少字段：{$label}");
    }
    return trim($match[1]);
}
