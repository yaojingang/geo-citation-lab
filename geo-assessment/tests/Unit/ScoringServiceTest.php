<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Scoring\ScoringService;
use PHPUnit\Framework\TestCase;

final class ScoringServiceTest extends TestCase
{
    public function test_it_requires_an_exact_set_match_for_single_and_multiple_choice_questions(): void
    {
        $items = [
            $this->item('Q01', 'mechanism', 3, ['B'], ['B']),
            $this->item('Q02', 'mechanism', 4, ['A', 'B', 'D'], ['A', 'B']),
            $this->item('Q03', 'content', 4, ['A', 'C'], ['A', 'C', 'D']),
            $this->item('Q04', 'content', 3, ['C'], []),
        ];

        $result = (new ScoringService())->score($items);

        self::assertSame(3, $result['score']);
        self::assertSame(1, $result['correct_count']);
        self::assertSame(3, $result['dimensions']['mechanism']['earned']);
        self::assertSame(0, $result['dimensions']['content']['earned']);
        self::assertSame(['D'], $result['items'][1]['missing_codes']);
        self::assertSame(['D'], $result['items'][2]['extra_codes']);
    }

    public function test_it_can_produce_a_perfect_score(): void
    {
        $items = [];
        for ($number = 1; $number <= 30; $number++) {
            $weight = $number <= 20 ? 3 : 4;
            $items[] = $this->item(sprintf('Q%02d', $number), 'dimension', $weight, ['A'], ['A']);
        }

        $result = (new ScoringService())->score($items);

        self::assertSame(100, $result['score']);
        self::assertSame(30, $result['correct_count']);
    }

    private function item(string $code, string $dimension, int $weight, array $correct, array $selected): array
    {
        return [
            'snapshot' => [
                'code' => $code,
                'dimension' => $dimension,
                'weight' => $weight,
                'correct_codes' => $correct,
            ],
            'selected_codes' => $selected,
        ];
    }
}
