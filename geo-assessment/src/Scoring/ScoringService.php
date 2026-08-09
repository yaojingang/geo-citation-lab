<?php

declare(strict_types=1);

namespace GeoAssessment\Scoring;

final class ScoringService
{
    /** @var SelectionDiagnostics */
    private $diagnostics;

    public function __construct(?SelectionDiagnostics $diagnostics = null)
    {
        $this->diagnostics = $diagnostics ?? new SelectionDiagnostics();
    }

    /**
     * @param list<array{snapshot: array<string, mixed>, selected_codes: array}> $attemptItems
     * @return array{score: int, correct_count: int, answered_count: int, dimensions: array<string, array{earned: int, possible: int, percentage: float, label: string}>, items: list<array<string, mixed>>}
     */
    public function score(array $attemptItems): array
    {
        $score = 0;
        $correctCount = 0;
        $answeredCount = 0;
        $dimensions = [];
        $results = [];

        foreach ($attemptItems as $item) {
            $snapshot = $item['snapshot'];
            $selected = $this->diagnostics->canonical($item['selected_codes'] ?? []);
            $correct = $this->diagnostics->canonical($snapshot['correct_codes'] ?? []);
            $weight = (int) $snapshot['weight'];
            $dimension = (string) $snapshot['dimension'];
            $isCorrect = $selected === $correct;
            $points = $isCorrect ? $weight : 0;
            $diagnosis = $this->diagnostics->diagnose($correct, $selected);

            $score += $points;
            $correctCount += $isCorrect ? 1 : 0;
            $answeredCount += $selected !== [] ? 1 : 0;
            if (!isset($dimensions[$dimension])) {
                $dimensions[$dimension] = ['earned' => 0, 'possible' => 0, 'percentage' => 0.0, 'label' => ''];
            }
            $dimensions[$dimension]['earned'] += $points;
            $dimensions[$dimension]['possible'] += $weight;

            $results[] = array_merge([
                'code' => (string) $snapshot['code'],
                'selected_codes' => $selected,
                'correct_codes' => $correct,
                'is_correct' => $isCorrect,
                'is_answered' => $selected !== [],
                'points' => $points,
                'possible_points' => $weight,
            ], $diagnosis);
        }

        foreach ($dimensions as &$dimension) {
            $dimension['percentage'] = $dimension['possible'] > 0
                ? round(($dimension['earned'] / $dimension['possible']) * 100, 1)
                : 0.0;
            $dimension['label'] = ScoreBands::dimension($dimension['percentage']);
        }
        unset($dimension);

        return [
            'score' => $score,
            'correct_count' => $correctCount,
            'answered_count' => $answeredCount,
            'dimensions' => $dimensions,
            'items' => $results,
        ];
    }
}
