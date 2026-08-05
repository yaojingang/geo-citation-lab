<?php

declare(strict_types=1);

namespace GeoAssessment\Support;

final class BaiduAnalytics
{
    public static function normalize($value): string
    {
        $id = strtolower(trim((string) $value));

        return preg_match('/\A[a-f0-9]{32}\z/D', $id) === 1 ? $id : '';
    }

    public static function enabled(string $id): bool
    {
        return self::normalize($id) !== '';
    }

    public static function inlineLoader(string $id): string
    {
        $id = self::normalize($id);
        if ($id === '') {
            return '';
        }

        return 'var _hmt = _hmt || [];' . "\n"
            . '(function() {' . "\n"
            . '  var hm = document.createElement("script");' . "\n"
            . '  hm.src = "https://hm.baidu.com/hm.js?' . $id . '";' . "\n"
            . '  var s = document.getElementsByTagName("script")[0];' . "\n"
            . '  s.parentNode.insertBefore(hm, s);' . "\n"
            . '})();';
    }

    public static function cspHash(string $id): string
    {
        $loader = self::inlineLoader($id);

        return $loader === '' ? '' : base64_encode(hash('sha256', $loader, true));
    }
}
