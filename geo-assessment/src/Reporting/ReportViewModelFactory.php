<?php

declare(strict_types=1);

namespace GeoAssessment\Reporting;

use DomainException;
use GeoAssessment\Scoring\ScoreBands;
use GeoAssessment\Scoring\ScoringService;
use PDO;

final class ReportViewModelFactory
{
    private const DIMENSIONS = [
        'mechanism' => 'GEO 基础理解',
        'content' => '内容与证据优化',
        'measurement' => '测量与实验判断',
        'overseas' => '平台与来源差异',
        'domestic' => '国内平台与内容生态',
        'governance' => '落地与风险治理',
    ];
    private const DIFFICULTIES = ['basic' => '基础', 'advanced' => '进阶', 'challenge' => '挑战'];

    /** @var PDO */
    private $pdo;

    /** @var ScoringService */
    private $scoring;

    /** @var InsightService */
    private $insights;

    /** @var CohortService */
    private $cohorts;

    public function __construct(
        PDO $pdo,
        ?ScoringService $scoring = null,
        ?InsightService $insights = null,
        ?CohortService $cohorts = null
    ) {
        $this->pdo = $pdo;
        $this->scoring = $scoring ?? new ScoringService();
        $this->insights = $insights ?? new InsightService();
        $this->cohorts = $cohorts ?? new CohortService($pdo);
    }

    /** @return array<string, mixed> */
    public function build(string $attemptId, string $userId): array
    {
        $statement = $this->pdo->prepare('SELECT a.*, u.display_name, qs.version AS set_version, qs.title AS set_title, qs.evidence_frozen_at FROM attempts a JOIN users u ON u.id = a.user_id JOIN question_sets qs ON qs.id = a.set_id WHERE a.id = :id AND a.user_id = :user_id');
        $statement->execute(['id' => $attemptId, 'user_id' => $userId]);
        $attempt = $statement->fetch();
        if (!is_array($attempt) || $attempt['status'] === 'in_progress') {
            throw new DomainException('报告不存在或测试尚未结束。');
        }

        $items = $this->loadItems($attemptId);
        $dimensionLabels = self::DIMENSIONS;
        $snapshotLabels = [];
        foreach ($items as $item) {
            $snapshot = $item['snapshot'];
            $dimension = (string) ($snapshot['dimension'] ?? '');
            $label = trim((string) ($snapshot['dimension_label'] ?? ''));
            if (isset($dimensionLabels[$dimension]) && $label !== '') {
                $snapshotLabels[$dimension][$label] = true;
            }
        }
        foreach ($snapshotLabels as $dimension => $labels) {
            $candidates = array_keys($labels);
            sort($candidates, SORT_STRING);
            $dimensionLabels[$dimension] = $candidates[0];
        }
        $scored = $this->scorePersistedItems($items);
        $scoredByCode = array_column($scored['items'], null, 'code');
        $questions = [];
        $dimensions = [];
        foreach ($dimensionLabels as $key => $label) {
            $dimension = $scored['dimensions'][$key] ?? ['earned' => 0, 'possible' => 0, 'percentage' => 0.0, 'label' => '需补课'];
            $dimensions[$key] = [
                'key' => $key,
                'label' => $label,
                'earned' => $dimension['earned'],
                'possible' => $dimension['possible'],
                'percentage' => $dimension['percentage'],
                'mastery_label' => $dimension['label'],
                'question_codes' => [],
            ];
        }
        $difficulties = [];
        foreach (self::DIFFICULTIES as $key => $label) {
            $difficulties[$key] = ['key' => $key, 'label' => $label, 'correct' => 0, 'incorrect' => 0, 'unanswered' => 0, 'earned' => 0, 'possible' => 0];
        }
        $types = [
            'single' => ['key' => 'single', 'label' => '单选', 'correct' => 0, 'total' => 0],
            'multiple' => ['key' => 'multiple', 'label' => '多选', 'correct' => 0, 'total' => 0],
        ];

        foreach ($items as $item) {
            $snapshot = $item['snapshot'];
            $scoreItem = $scoredByCode[$snapshot['code']];
            $status = !$scoreItem['is_answered'] ? 'unanswered' : ($scoreItem['is_correct'] ? 'correct' : 'incorrect');
            $question = array_merge($snapshot, $scoreItem, [
                'position' => $item['position'],
                'time_spent_seconds' => $item['time_spent_seconds'],
                'change_count' => $item['change_count'],
                'status' => $status,
            ]);
            $questions[] = $question;
            $dimensions[$snapshot['dimension']]['question_codes'][] = $snapshot['code'];
            $difficulties[$snapshot['difficulty']][$status]++;
            $difficulties[$snapshot['difficulty']]['earned'] += $scoreItem['points'];
            $difficulties[$snapshot['difficulty']]['possible'] += $scoreItem['possible_points'];
            $types[$snapshot['type']]['total']++;
            $types[$snapshot['type']]['correct'] += $scoreItem['is_correct'] ? 1 : 0;
        }
        foreach ($difficulties as &$difficulty) {
            $difficulty['percentage'] = $difficulty['possible'] > 0 ? round(($difficulty['earned'] / $difficulty['possible']) * 100, 1) : 0.0;
        }
        unset($difficulty);
        usort($questions, static function (array $left, array $right): int {
            return strcmp((string) $left['code'], (string) $right['code']);
        });
        foreach ($dimensions as &$dimension) {
            sort($dimension['question_codes'], SORT_STRING);
        }
        unset($dimension);

        $history = $this->history((string) $attempt['user_id']);
        $historyForSet = array_values(array_filter($history, static function (array $entry) use ($attempt): bool {
            return $entry['set_version'] === $attempt['set_version'];
        }));
        $trendHistory = array_slice($historyForSet, -5);
        $matrix = array_map(static function (array $question): array {
            return [
                'code' => $question['code'],
                'position' => $question['position'],
                'status' => $question['status'],
                'points' => $question['points'],
                'possible_points' => $question['possible_points'],
                'dimension' => $question['dimension'],
                'title' => $question['title'],
            ];
        }, $questions);

        $score = (int) $attempt['score'];
        $viewCatalog = [
            ['id' => 'score-ring', 'label' => '总分环'],
            ['id' => 'dimension-radar', 'label' => '六维雷达'],
            ['id' => 'dimension-bars', 'label' => '六维得分条'],
            ['id' => 'difficulty-stack', 'label' => '难度表现'],
            ['id' => 'question-time', 'label' => '逐题用时'],
            ['id' => 'question-matrix', 'label' => '30 题矩阵'],
            ['id' => 'score-trend', 'label' => '总分趋势'],
            ['id' => 'dimension-trend', 'label' => '维度趋势'],
        ];

        return [
            'user' => ['id' => $userId, 'display_name' => $attempt['display_name']],
            'attempt' => $attempt,
            'summary' => [
                'score' => $score,
                'stage' => ScoreBands::total($score),
                'correct_count' => (int) $attempt['correct_count'],
                'answered_count' => $scored['answered_count'],
                'duration_seconds' => (int) $attempt['duration_seconds'],
                'attempt_no' => (int) $attempt['attempt_no'],
                'status' => $attempt['status'],
            ],
            'dimensions' => array_values($dimensions),
            'difficulties' => array_values($difficulties),
            'types' => array_values($types),
            'questions' => $questions,
            'matrix' => $matrix,
            'history' => $history,
            'cohort' => $this->cohorts->position((string) $attempt['set_id'], $score),
            'insights' => $this->insights->build($dimensions, $questions),
            'view_catalog' => $viewCatalog,
            'charts' => [
                'score_ring' => ['score' => $score, 'remaining' => 100 - $score],
                'dimension_radar' => ['labels' => array_column($dimensions, 'label'), 'values' => array_column($dimensions, 'percentage')],
                'dimension_bars' => ['labels' => array_column($dimensions, 'label'), 'values' => array_column($dimensions, 'percentage')],
                'difficulty_stack' => ['labels' => array_column($difficulties, 'label'), 'correct' => array_column($difficulties, 'correct'), 'incorrect' => array_column($difficulties, 'incorrect'), 'unanswered' => array_column($difficulties, 'unanswered')],
                'question_time' => ['labels' => array_column($questions, 'code'), 'values' => array_column($questions, 'time_spent_seconds')],
                'question_matrix' => $matrix,
                'trend_ready' => count($trendHistory) >= 2,
                'trend_count' => count($trendHistory),
                'score_trend' => ['labels' => array_map(static function (array $entry): string {
                    return '第' . $entry['attempt_no'] . '次';
                }, $trendHistory), 'scores' => array_column($trendHistory, 'score'), 'attempt_ids' => array_column($trendHistory, 'id')],
                'dimension_trend' => $this->dimensionTrend($trendHistory, $dimensionLabels),
            ],
        ];
    }

    /** @return list<array<string, mixed>> */
    private function loadItems(string $attemptId): array
    {
        $statement = $this->pdo->prepare('SELECT * FROM attempt_items WHERE attempt_id = :attempt_id ORDER BY position');
        $statement->execute(['attempt_id' => $attemptId]);
        return array_map(static function (array $row): array {
            return array_merge($row, [
                'position' => (int) $row['position'],
                'time_spent_seconds' => (int) $row['time_spent_seconds'],
                'change_count' => (int) $row['change_count'],
                'snapshot' => json_decode((string) $row['question_snapshot_json'], true, 512, JSON_THROW_ON_ERROR),
                'selected_codes' => json_decode((string) $row['response_json'], true, 512, JSON_THROW_ON_ERROR),
            ]);
        }, $statement->fetchAll());
    }

    /** @return list<array<string, mixed>> */
    private function history(string $userId): array
    {
        $statement = $this->pdo->prepare("SELECT a.id, a.attempt_no, a.status, a.score, a.correct_count, a.duration_seconds, a.submitted_at, qs.version AS set_version FROM attempts a JOIN question_sets qs ON qs.id = a.set_id WHERE a.user_id = :user_id AND a.status IN ('submitted', 'timed_out') ORDER BY a.attempt_no");
        $statement->execute(['user_id' => $userId]);
        $history = $statement->fetchAll();
        foreach ($history as &$entry) {
            $entry['attempt_no'] = (int) $entry['attempt_no'];
            $entry['score'] = (int) $entry['score'];
            $entry['correct_count'] = (int) $entry['correct_count'];
            $entry['duration_seconds'] = (int) $entry['duration_seconds'];
            $items = $this->loadItems((string) $entry['id']);
            $scored = $this->scorePersistedItems($items);
            $entry['dimensions'] = $scored['dimensions'];
        }
        unset($entry);
        return $history;
    }

    /**
     * The stored item points are the scoring-version result captured at submission.
     * Current scoring is used only for stable answer diagnostics and presentation.
     *
     * @param list<array<string, mixed>> $items
     * @return array<string, mixed>
     */
    private function scorePersistedItems(array $items): array
    {
        $scored = $this->scoring->score(array_map(static function (array $item): array {
            return ['snapshot' => $item['snapshot'], 'selected_codes' => $item['selected_codes']];
        }, $items));
        $dimensions = [];
        $score = 0;
        $correctCount = 0;

        foreach ($items as $index => $item) {
            $result = $scored['items'][$index];
            $points = $item['points'] === null ? (int) $result['points'] : (int) $item['points'];
            $possible = (int) $result['possible_points'];
            $dimension = (string) $item['snapshot']['dimension'];
            $isCorrect = $points === $possible;
            $scored['items'][$index]['points'] = $points;
            $scored['items'][$index]['is_correct'] = $isCorrect;
            $score += $points;
            $correctCount += $isCorrect ? 1 : 0;
            if (!isset($dimensions[$dimension])) {
                $dimensions[$dimension] = ['earned' => 0, 'possible' => 0, 'percentage' => 0.0, 'label' => ''];
            }
            $dimensions[$dimension]['earned'] += $points;
            $dimensions[$dimension]['possible'] += $possible;
        }

        foreach ($dimensions as &$dimension) {
            $dimension['percentage'] = $dimension['possible'] > 0
                ? round(($dimension['earned'] / $dimension['possible']) * 100, 1)
                : 0.0;
            $dimension['label'] = ScoreBands::dimension($dimension['percentage']);
        }
        unset($dimension);

        $scored['score'] = $score;
        $scored['correct_count'] = $correctCount;
        $scored['dimensions'] = $dimensions;
        return $scored;
    }

    /** @return array{labels: list<string>, series: list<array{name: string, key: string, values: list<float>}>} */
    private function dimensionTrend(array $history, array $dimensionLabels): array
    {
        $series = [];
        foreach ($dimensionLabels as $key => $label) {
            $series[] = [
                'name' => $label,
                'key' => $key,
                'values' => array_map(static function (array $entry) use ($key): float {
                    return (float) ($entry['dimensions'][$key]['percentage'] ?? 0);
                }, $history),
            ];
        }
        return [
            'labels' => array_map(static function (array $entry): string {
                return '第' . $entry['attempt_no'] . '次';
            }, $history),
            'series' => $series,
        ];
    }
}
