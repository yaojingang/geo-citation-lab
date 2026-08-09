<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Reporting\CertificateAward;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

final class CertificateAwardTest extends TestCase
{
    #[DataProvider('scoreBoundaries')]
    public function test_certificate_titles_follow_the_approved_score_boundaries(int $score, string $title): void
    {
        self::assertSame($title, CertificateAward::forScore($score)['title']);
    }

    public static function scoreBoundaries(): array
    {
        return [
            [0, ''],
            [69, ''],
            [70, 'GEO 精英'],
            [79, 'GEO 精英'],
            [80, 'GEO 专家'],
            [89, 'GEO 专家'],
            [90, 'GEO 大师'],
            [100, 'GEO 大师'],
        ];
    }

    public function test_certificate_copy_is_positive_and_has_no_terminal_period(): void
    {
        foreach ([35, 75, 85, 95] as $score) {
            $award = CertificateAward::forScore($score);

            self::assertNotSame('', $award['encouragement']);
            self::assertDoesNotMatchRegularExpression('/[。.]\z/u', $award['encouragement']);
            self::assertDoesNotMatchRegularExpression('/[。.]\z/u', $award['highlight']);
        }
    }
}
