<?php

declare(strict_types=1);

namespace GeoAssessment\Http\Controller;

use DomainException;
use GeoAssessment\Http\Request;
use GeoAssessment\Http\Response;
use GeoAssessment\Identity\IdentityService;
use GeoAssessment\Reporting\ReportViewModelFactory;
use GeoAssessment\Support\View;

final class ReportController
{
    /** @var View */
    private $view;

    /** @var IdentityService */
    private $identities;

    /** @var ReportViewModelFactory */
    private $reports;

    public function __construct(View $view, IdentityService $identities, ReportViewModelFactory $reports)
    {
        $this->view = $view;
        $this->identities = $identities;
        $this->reports = $reports;
    }

    public function show(Request $request, array $params): Response
    {
        $user = $this->identities->resolve($request->cookie('geo_assessment_session'));
        if ($user === null) {
            return Response::redirect($this->view->url('/'));
        }
        try {
            $report = $this->reports->build((string) $params['id'], (string) $user['id']);
        } catch (DomainException $error) {
            return $this->view->render('error', ['view' => $this->view, 'statusCode' => 404, 'heading' => '报告未找到', 'message' => '该报告不存在、尚未生成，或不属于当前测试者。'], '报告未找到', 'page-error', 404);
        }
        return $this->view->render('report', ['view' => $this->view, 'report' => $report], 'GEO 能力报告 · ' . $report['summary']['score'] . ' 分', 'page-report')
            ->withHeader('X-Robots-Tag', 'noindex, noarchive');
    }
}
