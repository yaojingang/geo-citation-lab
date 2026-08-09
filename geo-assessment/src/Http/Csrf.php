<?php

declare(strict_types=1);

namespace GeoAssessment\Http;

final class Csrf
{
    public const COOKIE = 'geo_assessment_csrf';
    /** @var string|null */
    private $issuedSecret;

    /** @var string */
    private $appKey;

    public function __construct(string $appKey)
    {
        $this->appKey = $appKey;
    }

    public function token(Request $request): string
    {
        $secret = $request->cookie(self::COOKIE);
        if ($secret === null || !preg_match('/^[A-Za-z0-9_-]{43}$/', $secret)) {
            $secret = rtrim(strtr(base64_encode(random_bytes(32)), '+/', '-_'), '=');
            $this->issuedSecret = $secret;
        }
        return hash_hmac('sha256', 'geo-assessment-form', $secret . $this->appKey);
    }

    public function validate(Request $request): bool
    {
        $provided = (string) ($request->input('_csrf') ?? $request->header('x-csrf-token') ?? '');
        return $provided !== '' && hash_equals($this->token($request), $provided);
    }

    public function issuedSecret(): ?string
    {
        return $this->issuedSecret;
    }
}
