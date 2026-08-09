<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Scoring\ScoreBands;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

final class ScoreBandsTest extends TestCase
{
    #[DataProvider('totalBands')]
    public function test_it_maps_total_score_boundaries(int $score, string $label): void
    {
        self::assertSame($label, ScoreBands::total($score)['label']);
    }

    public static function totalBands(): array
    {
        return [
            [0, '探索期'], [39, '探索期'], [40, '基础期'], [59, '基础期'],
            [60, '实践期'], [74, '实践期'], [75, '系统期'], [89, '系统期'],
            [90, '研究策略期'], [100, '研究策略期'],
        ];
    }

    #[DataProvider('dimensionBands')]
    public function test_it_maps_dimension_percentage_boundaries(float $percentage, string $label): void
    {
        self::assertSame($label, ScoreBands::dimension($percentage));
    }

    public static function dimensionBands(): array
    {
        return [[0, '需补课'], [39.9, '需补课'], [40, '基础理解'], [59.9, '基础理解'], [60, '可实践'], [79.9, '可实践'], [80, '系统掌握']];
    }
}
