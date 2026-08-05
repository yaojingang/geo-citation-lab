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

    public function test_default_policy_does_not_allow_baidu_analytics(): void
    {
        $policy = SecurityHeaders::all()['Content-Security-Policy'];

        self::assertStringContainsString("script-src 'self'", $policy);
        self::assertStringNotContainsString('hm.baidu.com', $policy);
        self::assertStringNotContainsString("'unsafe-inline'", $policy);
    }

    public function test_enabled_baidu_analytics_loader_has_an_exact_csp_hash(): void
    {
        $id = '0123456789abcdef0123456789abcdef';
        $policy = SecurityHeaders::all($id)['Content-Security-Policy'];
        $hash = \GeoAssessment\Support\BaiduAnalytics::cspHash($id);

        self::assertStringContainsString("script-src 'self' 'sha256-{$hash}' https://hm.baidu.com", $policy);
        self::assertStringContainsString("img-src 'self' data: https://hm.baidu.com", $policy);
        self::assertStringContainsString("connect-src 'self' https://hm.baidu.com", $policy);
        self::assertStringNotContainsString("'unsafe-inline'", $policy);
    }
}
