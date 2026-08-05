<?php

declare(strict_types=1);

namespace GeoAssessment\Assessment;

use RuntimeException;

final class QuestionSetValidator
{
    private const EXPECTED_DIMENSIONS = [
        'mechanism' => ['count' => 5, 'points' => 17],
        'content' => ['count' => 5, 'points' => 17],
        'measurement' => ['count' => 5, 'points' => 17],
        'overseas' => ['count' => 5, 'points' => 16],
        'domestic' => ['count' => 5, 'points' => 17],
        'governance' => ['count' => 5, 'points' => 16],
    ];

    /** @return array{question_count: int, total_points: int, single_count: int, multiple_count: int, difficulties: array<string, int>, source_count: int, covered_papers: int, expected_seconds: int, content_hash: string} */
    public function validate(array $payload): array
    {
        if (($payload['schema_version'] ?? null) !== 'question-set-v1') {
            throw new RuntimeException('题库 schema_version 必须为 question-set-v1。');
        }
        $set = $payload['set'] ?? [];
        if (!is_array($set) || trim((string) ($set['version'] ?? '')) === '' || (int) ($set['total_points'] ?? 0) !== 100 || (int) ($set['time_limit_seconds'] ?? 0) !== 1800 || trim((string) ($set['scoring_version'] ?? '')) === '') {
            throw new RuntimeException('题集元数据必须包含版本、100 分、1,800 秒和评分版本。');
        }

        $questions = $payload['questions'] ?? null;
        $sources = $payload['sources'] ?? null;
        if (!is_array($questions) || !is_array($sources)) {
            throw new RuntimeException('题库缺少 questions 或 sources。');
        }

        $sourceIds = [];
        foreach ($sources as $source) {
            $id = (string) ($source['id'] ?? '');
            if ($id === '' || isset($sourceIds[$id])) {
                throw new RuntimeException('来源编号不能为空或重复。');
            }
            if (trim((string) ($source['title'] ?? '')) === '' || trim((string) ($source['citation'] ?? '')) === '' || !in_array($source['kind'] ?? null, ['paper', 'dataset'], true)) {
                throw new RuntimeException("来源 {$id} 缺少标题、引用信息或类型无效。");
            }
            $url = $source['url'] ?? null;
            if ($url !== null && (!is_string($url) || filter_var($url, FILTER_VALIDATE_URL) === false || parse_url($url, PHP_URL_SCHEME) !== 'https')) {
                throw new RuntimeException("来源 {$id} 的公开链接必须使用有效 HTTPS URL。");
            }
            $sourceIds[$id] = true;
        }

        $summary = [
            'question_count' => count($questions),
            'total_points' => 0,
            'single_count' => 0,
            'multiple_count' => 0,
            'difficulties' => ['basic' => 0, 'advanced' => 0, 'challenge' => 0],
            'source_count' => count($sources),
            'covered_papers' => 0,
            'expected_seconds' => 0,
            'content_hash' => '',
        ];
        $dimensionStats = [];
        $coveredPapers = [];

        foreach ($questions as $index => $question) {
            $expectedCode = sprintf('Q%02d', $index + 1);
            $code = (string) ($question['code'] ?? '');
            if ($code !== $expectedCode) {
                throw new RuntimeException("题号必须连续，期望 {$expectedCode}，实际 {$code}。");
            }

            $type = (string) ($question['type'] ?? '');
            if (!in_array($type, ['single', 'multiple'], true)) {
                throw new RuntimeException("{$code} 的题型无效。");
            }
            $summary[$type . '_count']++;

            $difficulty = (string) ($question['difficulty'] ?? '');
            if (!array_key_exists($difficulty, $summary['difficulties'])) {
                throw new RuntimeException("{$code} 的难度无效。");
            }
            $summary['difficulties'][$difficulty]++;

            $dimension = (string) ($question['dimension'] ?? '');
            if (!isset(self::EXPECTED_DIMENSIONS[$dimension])) {
                throw new RuntimeException("{$code} 的维度无效。");
            }
            $weight = (int) ($question['weight'] ?? 0);
            if (!in_array($weight, [3, 4], true)) {
                throw new RuntimeException("{$code} 的分值必须为 3 或 4。");
            }
            $summary['total_points'] += $weight;
            if (!isset($dimensionStats[$dimension])) {
                $dimensionStats[$dimension] = ['count' => 0, 'points' => 0];
            }
            $dimensionStats[$dimension]['count']++;
            $dimensionStats[$dimension]['points'] += $weight;

            $choices = $question['choices'] ?? [];
            if (!is_array($choices) || !in_array(count($choices), [4, 5], true)) {
                throw new RuntimeException("{$code} 必须包含 4 或 5 个选项。");
            }
            $choiceCodes = [];
            $flaggedCorrectCodes = [];
            foreach ($choices as $choice) {
                $choiceCode = (string) ($choice['code'] ?? '');
                if ($choiceCode === '' || isset($choiceCodes[$choiceCode])) {
                    throw new RuntimeException("{$code} 包含空白或重复选项码。");
                }
                if (trim((string) ($choice['text'] ?? '')) === '' || trim((string) ($choice['rationale'] ?? '')) === '') {
                    throw new RuntimeException("{$code} 的 {$choiceCode} 选项缺少正文或说明。");
                }
                if (!is_bool($choice['is_correct'] ?? null)) {
                    throw new RuntimeException("{$code} 的 {$choiceCode} 选项缺少布尔型正确标记。");
                }
                if ($choice['is_correct']) {
                    $flaggedCorrectCodes[] = $choiceCode;
                }
                $choiceCodes[$choiceCode] = true;
            }

            $correctCodes = array_values(array_unique($question['correct_codes'] ?? []));
            sort($correctCodes, SORT_STRING);
            sort($flaggedCorrectCodes, SORT_STRING);
            if (($type === 'single' && count($correctCodes) !== 1) || ($type === 'multiple' && !in_array(count($correctCodes), [3, 4], true))) {
                throw new RuntimeException("{$code} 的正确答案数量与题型不匹配。");
            }
            foreach ($correctCodes as $correctCode) {
                if (!isset($choiceCodes[$correctCode])) {
                    throw new RuntimeException("{$code} 的正确答案 {$correctCode} 不在选项中。");
                }
            }
            if ($correctCodes !== $flaggedCorrectCodes) {
                throw new RuntimeException("{$code} 的 correct_codes 与选项正确标记不一致。");
            }

            foreach (['prompt', 'explanation', 'principle', 'misconception_tag', 'cognitive_level'] as $field) {
                $text = trim((string) ($question[$field] ?? ''));
                if ($text === '' || preg_match('/<[^>]+>|\{\{|TBD|TODO/u', $text) === 1) {
                    throw new RuntimeException("{$code} 的 {$field} 为空或包含占位内容。");
                }
            }

            $expectedSeconds = (int) ($question['expected_seconds'] ?? 0);
            if ($expectedSeconds <= 0 || $expectedSeconds > 180) {
                throw new RuntimeException("{$code} 的预计时间越界。");
            }
            $summary['expected_seconds'] += $expectedSeconds;

            $refs = $question['source_refs'] ?? [];
            if (!is_array($refs) || $refs === []) {
                throw new RuntimeException("{$code} 必须有证据来源。");
            }
            foreach ($refs as $ref) {
                if (!isset($sourceIds[$ref])) {
                    throw new RuntimeException("{$code} 引用了未知来源 {$ref}。");
                }
                if (preg_match('/^P\d{2}$/', (string) $ref) === 1) {
                    $coveredPapers[$ref] = true;
                }
            }
        }

        if ($summary['question_count'] !== 30 || $summary['total_points'] !== 100 || $summary['single_count'] !== 20 || $summary['multiple_count'] !== 10) {
            throw new RuntimeException('题库必须满足 30 题、100 分、20 道单选与 10 道多选。');
        }
        if ($summary['difficulties'] !== ['basic' => 10, 'advanced' => 13, 'challenge' => 7]) {
            throw new RuntimeException('难度数量必须为 10 道基础、13 道进阶和 7 道挑战。');
        }
        if ($dimensionStats !== self::EXPECTED_DIMENSIONS) {
            throw new RuntimeException('六维题数或分值与冻结蓝图不一致。');
        }
        if ($summary['expected_seconds'] > 1800) {
            throw new RuntimeException('题库预计作答时间超过 30 分钟。');
        }

        $summary['covered_papers'] = count($coveredPapers);
        if ($summary['covered_papers'] !== 54) {
            throw new RuntimeException('题库必须覆盖 P01 至 P54。');
        }

        $canonical = json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR);
        $summary['content_hash'] = hash('sha256', $canonical);
        return $summary;
    }
}
