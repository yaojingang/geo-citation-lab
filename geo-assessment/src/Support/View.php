<?php

declare(strict_types=1);

namespace GeoAssessment\Support;

use GeoAssessment\Http\Response;

final class View
{
    /** @var string */
    private $templatesPath;

    /** @var string */
    private $basePath;

    /** @var string */
    private $baiduAnalyticsId;

    public function __construct(string $templatesPath, string $basePath = '', string $baiduAnalyticsId = '')
    {
        $this->templatesPath = $templatesPath;
        $this->basePath = $basePath;
        $this->baiduAnalyticsId = BaiduAnalytics::normalize($baiduAnalyticsId);
    }

    public function render(string $template, array $data = [], string $title = 'GEO 在线能力测试', string $pageClass = '', int $status = 200, bool $allowAnalytics = false): Response
    {
        $templatePath = $this->templatesPath . '/' . $template . '.php';
        if (!is_file($templatePath)) {
            throw new \RuntimeException("模板不存在：{$template}");
        }
        $basePath = $this->basePath;
        $baiduAnalyticsId = $allowAnalytics ? $this->baiduAnalyticsId : '';
        $templatesPath = $this->templatesPath;
        extract($data, EXTR_SKIP);
        ob_start();
        require $templatePath;
        $content = (string) ob_get_clean();
        ob_start();
        require $this->templatesPath . '/layout.php';
        return Response::html((string) ob_get_clean(), $status)
            ->withAnalyticsAllowed(BaiduAnalytics::enabled($baiduAnalyticsId));
    }

    public function url(string $path): string
    {
        if ($path === '') {
            $path = '/';
        }
        return $this->basePath . $path;
    }

    public function analyticsEnabled(): bool
    {
        return BaiduAnalytics::enabled($this->baiduAnalyticsId);
    }

    /** @param mixed $value */
    public static function e($value): string
    {
        return htmlspecialchars((string) $value, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
    }

    /** @param mixed $value */
    public static function trimTerminalPeriod($value): string
    {
        $text = (string) $value;

        return preg_replace('/[。.]\z/u', '', $text) ?? $text;
    }

    /** @param mixed $value */
    public static function jsonAttribute($value): string
    {
        return self::e(json_encode($value, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_HEX_TAG | JSON_HEX_AMP | JSON_HEX_APOS | JSON_HEX_QUOT | JSON_THROW_ON_ERROR));
    }

    public static function duration(int $seconds): string
    {
        return sprintf('%02d:%02d', intdiv($seconds, 60), $seconds % 60);
    }
}
