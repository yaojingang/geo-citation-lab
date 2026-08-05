<?php

declare(strict_types=1);

namespace GeoAssessment\Http;

use GeoAssessment\Support\BaiduAnalytics;

final class SecurityHeaders
{
    /** @return array<string, string> */
    public static function all(string $baiduAnalyticsId = ''): array
    {
        $scriptSource = "script-src 'self'";
        $imageSource = "img-src 'self' data:";
        $connectSource = "connect-src 'self'";
        if (BaiduAnalytics::enabled($baiduAnalyticsId)) {
            $scriptSource .= " 'sha256-" . BaiduAnalytics::cspHash($baiduAnalyticsId) . "' https://hm.baidu.com";
            $imageSource .= ' https://hm.baidu.com';
            $connectSource .= ' https://hm.baidu.com';
        }

        return [
            'Content-Security-Policy' => "default-src 'self'; {$scriptSource}; style-src 'self'; {$imageSource}; font-src 'self'; {$connectSource}; form-action 'self'; base-uri 'none'; frame-ancestors 'none'; object-src 'none'",
            'X-Content-Type-Options' => 'nosniff',
            'Referrer-Policy' => 'no-referrer',
            'Permissions-Policy' => 'camera=(), microphone=(), geolocation=()',
            'X-Frame-Options' => 'DENY',
            'Cache-Control' => 'no-store, private',
            'Pragma' => 'no-cache',
        ];
    }
}
