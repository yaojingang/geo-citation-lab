<?php

declare(strict_types=1);

namespace GeoAssessment\Http\Controller;

use DomainException;
use GeoAssessment\Assessment\AttemptService;
use GeoAssessment\Http\Csrf;
use GeoAssessment\Http\RateLimiter;
use GeoAssessment\Http\Request;
use GeoAssessment\Http\Response;
use GeoAssessment\Identity\IdentityService;
use GeoAssessment\Support\View;

final class AttemptController
{
    /** @var View */
    private $view;

    /** @var IdentityService */
    private $identities;

    /** @var AttemptService */
    private $attempts;

    /** @var Csrf */
    private $csrf;

    /** @var RateLimiter */
    private $limiter;

    public function __construct(
        View $view,
        IdentityService $identities,
        AttemptService $attempts,
        Csrf $csrf,
        RateLimiter $limiter
    ) {
        $this->view = $view;
        $this->identities = $identities;
        $this->attempts = $attempts;
        $this->csrf = $csrf;
        $this->limiter = $limiter;
    }

    public function start(Request $request): Response
    {
        [$user, $token] = $this->requireUser($request);
        if (!$this->limiter->allow($token, 'attempt', 20, 60)) {
            return $this->failure($request, '开始测试的请求过于频繁。', 429);
        }
        try {
            $attempt = $this->attempts->start((string) $user['id']);
        } catch (DomainException $error) {
            return $this->failure($request, $error->getMessage(), 409);
        }
        $position = 1;
        foreach ($this->attempts->items((string) $attempt['id'], (string) $user['id']) as $item) {
            if ($item['selected_codes'] === []) {
                $position = $item['position'];
                break;
            }
        }
        return Response::redirect($this->view->url('/attempts/' . $attempt['id'] . '/questions/' . $position));
    }

    public function question(Request $request, array $params): Response
    {
        [$user] = $this->requireUser($request);
        $attemptId = (string) $params['id'];
        $position = max(1, min(30, (int) $params['position']));
        try {
            $attempt = $this->attempts->getAttempt($attemptId, (string) $user['id']);
            if ($attempt['status'] !== 'in_progress') {
                return Response::redirect($this->view->url('/reports/' . $attemptId));
            }
            $item = $this->attempts->itemAt($attemptId, (string) $user['id'], $position);
            $items = $this->attempts->items($attemptId, (string) $user['id']);
        } catch (DomainException $error) {
            if (strpos($error->getMessage(), '时间已到') !== false) {
                return Response::redirect($this->view->url('/reports/' . $attemptId));
            }
            return $this->failure($request, '该测试不存在或不属于当前测试者。', 404);
        }

        $answered = count(array_filter($items, static function (array $row): bool {
            return $row['selected_codes'] !== [];
        }));
        return $this->view->render('question', [
            'view' => $this->view,
            'csrfToken' => $this->csrf->token($request),
            'user' => $user,
            'attempt' => $attempt,
            'item' => $item,
            'items' => $items,
            'answered' => $answered,
        ], $item['snapshot']['prompt'], 'page-question');
    }

    public function save(Request $request, array $params): Response
    {
        [$user, $token] = $this->requireUser($request);
        if (!$this->limiter->allow($token, 'answer', 120, 60)) {
            return $this->failure($request, '保存请求过于频繁。', 429);
        }
        $selected = $request->input('selected_codes', []);
        $selected = is_array($selected) ? $selected : [$selected];
        try {
            $result = $this->attempts->saveAnswer(
                (string) $params['id'],
                (string) $user['id'],
                (string) $request->input('question_code', ''),
                $selected,
                (int) $request->input('activity_seq', 1),
                (int) $request->input('active_seconds_delta', 0)
            );
        } catch (DomainException $error) {
            if (strpos($error->getMessage(), '时间已到') !== false || strpos($error->getMessage(), '已结束') !== false) {
                return $request->expectsJson()
                    ? Response::json(['redirect' => $this->view->url('/reports/' . $params['id']), 'message' => $error->getMessage()], 409)
                    : Response::redirect($this->view->url('/reports/' . $params['id']));
            }
            return $this->failure($request, $error->getMessage(), 422);
        }
        $navigateTo = $request->input('navigate_to');
        if ($navigateTo !== null && $navigateTo !== '') {
            return Response::redirect($this->view->url('/attempts/' . $params['id'] . '/questions/' . max(1, min(30, (int) $navigateTo))));
        }
        return Response::json($result);
    }

    public function submit(Request $request, array $params): Response
    {
        [$user, $token] = $this->requireUser($request);
        if (!$this->limiter->allow($token, 'submit', 10, 60)) {
            return $this->failure($request, '交卷请求过于频繁。', 429);
        }
        $attemptId = (string) $params['id'];
        $questionCode = (string) $request->input('question_code', '');
        if ($questionCode !== '') {
            $selected = $request->input('selected_codes', []);
            $selected = is_array($selected) ? $selected : [$selected];
            try {
                $this->attempts->saveAnswer($attemptId, (string) $user['id'], $questionCode, $selected, (int) $request->input('activity_seq', 1), (int) $request->input('active_seconds_delta', 0));
            } catch (DomainException $error) {
                if (strpos($error->getMessage(), '时间已到') === false && strpos($error->getMessage(), '已结束') === false) {
                    return $this->failure($request, $error->getMessage(), 422);
                }
            }
        }
        try {
            $this->attempts->submit($attemptId, (string) $user['id']);
        } catch (DomainException $error) {
            return $this->failure($request, $error->getMessage(), 404);
        }
        return Response::redirect($this->view->url('/reports/' . $attemptId));
    }

    /** @return array{0: array<string, mixed>, 1: string} */
    private function requireUser(Request $request): array
    {
        $token = $request->cookie('geo_assessment_session');
        $user = $this->identities->resolve($token);
        if ($user === null || $token === null) {
            throw new DomainException('请先在首页输入姓名。');
        }
        return [$user, $token];
    }

    private function failure(Request $request, string $message, int $status): Response
    {
        if ($request->expectsJson()) {
            return Response::json(['error' => $message], $status);
        }
        return $this->view->render('error', ['view' => $this->view, 'statusCode' => $status, 'heading' => $status === 404 ? '页面未找到' : '请求未完成', 'message' => $message], '请求未完成', 'page-error', $status);
    }
}
