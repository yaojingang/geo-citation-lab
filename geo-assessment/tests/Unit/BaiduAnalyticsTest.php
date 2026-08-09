<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Support\BaiduAnalytics;
use PHPUnit\Framework\TestCase;

final class BaiduAnalyticsTest extends TestCase
{
    public function test_invalid_ids_fail_closed(): void
    {
        self::assertSame('', BaiduAnalytics::normalize(''));
        self::assertSame('', BaiduAnalytics::normalize('not-an-id'));
        self::assertSame('', BaiduAnalytics::inlineLoader('not-an-id'));
    }

    public function test_loader_is_complete_and_uses_the_configured_id(): void
    {
        $id = '0123456789abcdef0123456789abcdef';
        $loader = BaiduAnalytics::inlineLoader($id);

        self::assertStringStartsWith('var _hmt = _hmt || [];', $loader);
        self::assertStringContainsString('document.createElement("script")', $loader);
        self::assertStringContainsString("https://hm.baidu.com/hm.js?{$id}", $loader);
        self::assertStringContainsString('s.parentNode.insertBefore(hm, s);', $loader);
        self::assertSame(base64_encode(hash('sha256', $loader, true)), BaiduAnalytics::cspHash($id));
    }
}
