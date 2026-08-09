<?php

declare(strict_types=1);

namespace GeoAssessment\Http\Controller;

use GeoAssessment\Assessment\AttemptService;
use GeoAssessment\Http\Csrf;
use GeoAssessment\Http\Request;
use GeoAssessment\Http\Response;
use GeoAssessment\Identity\IdentityService;
use GeoAssessment\Support\View;

final class HomeController
{
    /** @var View */
    private $view;

    /** @var IdentityService */
    private $identities;

    /** @var AttemptService */
    private $attempts;

    /** @var Csrf */
    private $csrf;

    public function __construct(
        View $view,
        IdentityService $identities,
        AttemptService $attempts,
        Csrf $csrf
    ) {
        $this->view = $view;
        $this->identities = $identities;
        $this->attempts = $attempts;
        $this->csrf = $csrf;
    }

    public function index(Request $request, ?string $error = null, int $status = 200): Response
    {
        $token = $request->cookie('geo_assessment_session');
        $user = $this->identities->resolve($token);
        $history = [];
        $current = null;
        $currentPosition = 1;
        if ($user !== null) {
            $current = $this->attempts->current((string) $user['id']);
            $history = $this->attempts->history((string) $user['id']);
            if ($current !== null) {
                foreach ($this->attempts->items((string) $current['id'], (string) $user['id']) as $item) {
                    if ($item['selected_codes'] === []) {
                        $currentPosition = $item['position'];
                        break;
                    }
                }
            }
        }

        return $this->view->render('home', [
            'view' => $this->view,
            'csrfToken' => $this->csrf->token($request),
            'user' => $user,
            'history' => $history,
            'current' => $current,
            'currentPosition' => $currentPosition,
            'remaining' => max(0, 10 - count($history)),
            'error' => $error,
        ], 'GEO 在线能力测试', 'page-home', $status, $user === null && $error === null && $status === 200);
    }
}
