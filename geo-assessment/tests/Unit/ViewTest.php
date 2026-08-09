<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Support\BaiduAnalytics;
use GeoAssessment\Support\View;
use PHPUnit\Framework\TestCase;

final class ViewTest extends TestCase
{
    public function test_terminal_period_is_removed_without_changing_internal_punctuation(): void
    {
        self::assertSame('一句中文描述', View::trimTerminalPeriod('一句中文描述。'));
        self::assertSame('Version 1.0', View::trimTerminalPeriod('Version 1.0.'));
        self::assertSame('第一句。第二句', View::trimTerminalPeriod('第一句。第二句。'));
        self::assertSame('为什么？', View::trimTerminalPeriod('为什么？'));
    }

    public function test_layout_footer_uses_the_attempt_question_set_version(): void
    {
        $view = new View(dirname(__DIR__, 2) . '/templates');
        $response = $view->render('error', [
            'view' => $view,
            'statusCode' => 200,
            'heading' => '历史报告',
            'message' => '版本显示校验',
            'attempt' => ['set_version' => 'geo-30-v1.1'],
        ], '历史报告', 'page-error');

        self::assertStringContainsString('题集 geo-30-v1.1', $response->body);
        self::assertStringNotContainsString('题集 geo-30-v1.2', $response->body);
    }

    public function test_analytics_loader_requires_an_explicit_page_allowance(): void
    {
        $analyticsId = '0123456789abcdef0123456789abcdef';
        $view = new View(dirname(__DIR__, 2) . '/templates', '', $analyticsId);
        $loader = '<script>' . BaiduAnalytics::inlineLoader($analyticsId) . '</script>';
        $privateResponse = $view->render('error', [
            'view' => $view,
            'statusCode' => 200,
            'heading' => '统计代码',
            'message' => '页面头部校验',
        ], '统计代码', 'page-error');
        $publicResponse = $view->render('error', [
            'view' => $view,
            'statusCode' => 200,
            'heading' => '统计代码',
            'message' => '页面头部校验',
        ], '统计代码', 'page-error', 200, true);

        self::assertStringNotContainsString($loader, $privateResponse->body);
        self::assertFalse($privateResponse->analyticsAllowed);
        $headEnd = strpos($publicResponse->body, '</head>');
        $loaderPosition = strpos($publicResponse->body, $loader);
        self::assertIsInt($headEnd);
        self::assertIsInt($loaderPosition);
        self::assertLessThan($headEnd, $loaderPosition);
        self::assertSame(1, substr_count($publicResponse->body, $loader));
        self::assertTrue($publicResponse->analyticsAllowed);
    }

    public function test_layout_contains_the_complete_literal_analytics_script(): void
    {
        $layout = (string) file_get_contents(dirname(__DIR__, 2) . '/templates/layout.php');

        self::assertStringContainsString('<script>var _hmt = _hmt || [];', $layout);
        self::assertStringContainsString('(function() {', $layout);
        self::assertStringContainsString('var hm = document.createElement("script");', $layout);
        self::assertStringContainsString('hm.src = "https://hm.baidu.com/hm.js?', $layout);
        self::assertStringContainsString('var s = document.getElementsByTagName("script")[0];', $layout);
        self::assertStringContainsString('s.parentNode.insertBefore(hm, s);', $layout);
    }
}
