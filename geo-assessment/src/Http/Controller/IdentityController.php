<?php

declare(strict_types=1);

namespace GeoAssessment\Http\Controller;

use GeoAssessment\Assessment\AttemptService;
use GeoAssessment\Http\RateLimiter;
use GeoAssessment\Http\Request;
use GeoAssessment\Http\Response;
use GeoAssessment\Identity\IdentityService;
use GeoAssessment\Support\View;
use InvalidArgumentException;
use Throwable;

final class IdentityController
{
    /** @var View */
    private $view;

    /** @var HomeController */
    private $home;

    /** @var IdentityService */
    private $identities;

    /** @var AttemptService */
    private $attempts;

    /** @var RateLimiter */
    private $limiter;

    /** @var bool */
    private $trustProxy;

    /** @var bool */
    private $secureCookie;

    public function __construct(
        View $view,
        HomeController $home,
        IdentityService $identities,
        AttemptService $attempts,
        RateLimiter $limiter,
        bool $trustProxy,
        bool $secureCookie
    ) {
        $this->view = $view;
        $this->home = $home;
        $this->identities = $identities;
        $this->attempts = $attempts;
        $this->limiter = $limiter;
        $this->trustProxy = $trustProxy;
        $this->secureCookie = $secureCookie;
    }

    public function create(Request $request): Response
    {
        if (!$this->limiter->allow($request->remoteAddress($this->trustProxy), 'identity', 20, 3600)) {
            return $this->view->render('error', ['view' => $this->view, 'statusCode' => 429, 'heading' => '请稍后再试', 'message' => '当前设备创建测试者过于频繁。'], '请稍后再试', 'page-error', 429)->withHeader('Retry-After', '3600');
        }
        try {
            $identity = $this->identities->create((string) $request->input('name', ''));
        } catch (InvalidArgumentException $error) {
            return $this->home->index($request, $error->getMessage(), 422);
        }
        try {
            $attempt = $this->attempts->start((string) $identity['user']['id']);
        } catch (Throwable $error) {
            $this->identities->deleteUser((string) $identity['user']['id']);
            throw $error;
        }
        return Response::redirect($this->view->url('/attempts/' . $attempt['id'] . '/questions/1'))
            ->withHeader('Set-Cookie', $this->sessionCookie($identity['token'], 315360000));
    }

    public function switch(Request $request): Response
    {
        if ((string) $request->input('confirm_switch', '') !== '1') {
            return $this->home->index($request, '请先确认切换。切换后，当前浏览器将无法再打开这份报告。', 422);
        }
        $token = $request->cookie('geo_assessment_session');
        if ($token !== null) {
            $this->identities->revoke($token);
        }
        return Response::redirect($this->view->url('/'))->withHeader('Set-Cookie', $this->sessionCookie('', 0));
    }

    public function delete(Request $request): Response
    {
        $token = $request->cookie('geo_assessment_session');
        $user = $this->identities->resolve($token);
        if ($user === null) {
            return Response::redirect($this->view->url('/'));
        }
        $confirmation = trim((string) $request->input('confirmation_name', ''));
        if (!hash_equals((string) $user['display_name'], $confirmation)) {
            return $this->home->index($request, '请输入完整姓名以确认删除。', 422);
        }
        $this->identities->deleteUser((string) $user['id']);
        return Response::redirect($this->view->url('/'))->withHeader('Set-Cookie', $this->sessionCookie('', 0));
    }

    private function sessionCookie(string $value, int $maxAge): string
    {
        $path = $this->view->url('/') ?: '/';
        $parts = [
            'geo_assessment_session=' . rawurlencode($value),
            'Max-Age=' . $maxAge,
            'Path=' . $path,
            'HttpOnly',
            'SameSite=Lax',
        ];
        if ($this->secureCookie) {
            $parts[] = 'Secure';
        }
        return implode('; ', $parts);
    }
}
