<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Http\SecurityHeaders;
use PHPUnit\Framework\TestCase;

final class SecurityHeadersTest extends TestCase
{
    public function test_personalized_pages_cannot_be_cached(): void
    {
        $headers = SecurityHeaders::all();

        self::assertSame('no-store, private', $headers['Cache-Control']);
        self::assertSame('no-cache', $headers['Pragma']);
    }

    public function test_baidu_analytics_inline_loader_has_an_exact_csp_hash(): void
    {
        $policy = SecurityHeaders::all()['Content-Security-Policy'];
        $template = (string) file_get_contents(dirname(__DIR__, 2) . '/templates/layout.php');

        self::assertSame(1, preg_match('/<script>(var _hmt.*?\(\);)<\/script>/s', $template, $matches));
        $hash = base64_encode(hash('sha256', $matches[1], true));

        self::assertStringContainsString("script-src 'self' 'sha256-{$hash}' https://hm.baidu.com", $policy);
        self::assertStringContainsString("img-src 'self' data: https://hm.baidu.com", $policy);
        self::assertStringContainsString("connect-src 'self' https://hm.baidu.com", $policy);
        self::assertStringNotContainsString("'unsafe-inline'", $policy);
    }
}
