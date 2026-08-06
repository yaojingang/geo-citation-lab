<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Reporting\QrCodeSvg;
use PHPUnit\Framework\TestCase;

final class QrCodeSvgTest extends TestCase
{
    public function test_it_generates_a_self_contained_square_svg_and_data_uri(): void
    {
        $renderer = new QrCodeSvg();
        $svg = $renderer->render('https://ai.laoyao.cn/geo/certificates/123e4567-e89b-12d3-a456-426614174000');

        self::assertStringStartsWith('<svg ', $svg);
        self::assertStringContainsString('viewBox="0 0 ', $svg);
        self::assertStringContainsString('shape-rendering="crispEdges"', $svg);
        self::assertStringContainsString('fill="#1b365d"', $svg);
        self::assertStringNotContainsString('<script', $svg);
        self::assertSame('data:image/svg+xml;base64,' . base64_encode($svg), $renderer->dataUri('https://ai.laoyao.cn/geo/certificates/123e4567-e89b-12d3-a456-426614174000'));
    }
}
