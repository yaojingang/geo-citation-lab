<?php

declare(strict_types=1);

$appRoot = dirname(__DIR__);
$repoRoot = dirname($appRoot);
$catalogPath = $repoRoot . '/02-geo-aeo-ai-search-papers/00_资料说明/论文清单.csv';
$outputPath = $appRoot . '/database/seeds/geo-30-v1.2.json';
$questionPath = __DIR__ . '/question-bank-v1.2.php';

$questions = require $questionPath;

if (!is_array($questions) || count($questions) !== 30) {
    throw new RuntimeException('题库定义必须包含 30 道题');
}

$sourceRows = [];
$handle = fopen($catalogPath, 'rb');
if ($handle === false) {
    throw new RuntimeException('无法读取论文清单');
}
$headers = fgetcsv($handle, 0, ',', '"', '');
if (!is_array($headers)) {
    throw new RuntimeException('论文清单缺少表头');
}
$headers[0] = preg_replace('/^\xEF\xBB\xBF/', '', (string) $headers[0]);
while (($row = fgetcsv($handle, 0, ',', '"', '')) !== false) {
    $sourceRows[] = array_combine($headers, $row);
}
fclose($handle);

if (count($sourceRows) !== 54) {
    throw new RuntimeException('论文清单必须恰好包含 54 篇');
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

$payload = [
    'schema_version' => 'question-set-v1',
    'set' => [
        'version' => 'geo-30-v1.2',
        'title' => 'GEO 在线能力测试',
        'description' => '面向 GEO 初学者的 30 道场景题，以国内应用为主，并保留论文与数据证据',
        'total_points' => 100,
        'time_limit_seconds' => 1800,
        'scoring_version' => 'score-v1.2',
        'validation_profile' => 'beginner-domestic-v1',
        'evidence_frozen_at' => '2026-08-04',
    ],
    'sources' => $sources,
    'questions' => $questions,
];

$json = json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR) . PHP_EOL;
if (file_put_contents($outputPath, $json, LOCK_EX) === false) {
    throw new RuntimeException('题库种子写入失败');
}

fwrite(STDOUT, sprintf("已生成 %d 道题、%d 个来源：%s\n", count($questions), count($payload['sources']), $outputPath));
