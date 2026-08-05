<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Identity\NameNormalizer;
use InvalidArgumentException;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

final class NameNormalizerTest extends TestCase
{
    #[DataProvider('validNames')]
    public function test_it_normalizes_valid_display_names(string $input, string $display, string $normalized): void
    {
        $result = (new NameNormalizer())->normalize($input);

        self::assertSame($display, $result['display_name']);
        self::assertSame($normalized, $result['normalized_name']);
    }

    public static function validNames(): array
    {
        return [
            ['  张　 三  ', '张 三', '张 三'],
            ['Kai Zhang', 'Kai Zhang', 'kai zhang'],
            ['李小龙', '李小龙', '李小龙'],
        ];
    }

    #[DataProvider('invalidNames')]
    public function test_it_rejects_invalid_names(string $input): void
    {
        $this->expectException(InvalidArgumentException::class);
        (new NameNormalizer())->normalize($input);
    }

    public static function invalidNames(): array
    {
        return [
            [''],
            ['A'],
            [str_repeat('长', 41)],
            ["A\0B"],
            ["A\u{202E}B"],
        ];
    }
}
