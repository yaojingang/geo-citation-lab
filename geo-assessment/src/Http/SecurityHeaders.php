<?php

declare(strict_types=1);

namespace GeoAssessment\Http;

final class SecurityHeaders
{
    /** @return array<string, string> */
    public static function all(): array
    {
        return [
            'Content-Security-Policy' => "default-src 'self'; script-src 'self' 'sha256-SlfJ22vr7F+kLSfZOKuC+k2k+KUmsd/2r6xhQB8BW+U=' https://hm.baidu.com; style-src 'self'; img-src 'self' data: https://hm.baidu.com; font-src 'self'; connect-src 'self' https://hm.baidu.com; form-action 'self'; base-uri 'none'; frame-ancestors 'none'; object-src 'none'",
            'X-Content-Type-Options' => 'nosniff',
            'Referrer-Policy' => 'no-referrer',
            'Permissions-Policy' => 'camera=(), microphone=(), geolocation=()',
            'X-Frame-Options' => 'DENY',
            'Cache-Control' => 'no-store, private',
            'Pragma' => 'no-cache',
        ];
    }
}
