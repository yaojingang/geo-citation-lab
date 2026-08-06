<?php

declare(strict_types=1);

namespace GeoAssessment\Http\Controller;

use DomainException;
use GeoAssessment\Http\Request;
use GeoAssessment\Http\Response;
use GeoAssessment\Reporting\CertificateViewModelFactory;
use GeoAssessment\Reporting\QrCodeSvg;
use GeoAssessment\Support\View;

final class CertificateController
{
    /** @var View */
    private $view;

    /** @var CertificateViewModelFactory */
    private $certificates;

    /** @var QrCodeSvg */
    private $qrCodes;

    /** @var bool */
    private $trustProxy;

    /** @var string */
    private $publicUrl;

    public function __construct(
        View $view,
        CertificateViewModelFactory $certificates,
        ?QrCodeSvg $qrCodes = null,
        bool $trustProxy = false,
        string $publicUrl = ''
    ) {
        $this->view = $view;
        $this->certificates = $certificates;
        $this->qrCodes = $qrCodes ?? new QrCodeSvg();
        $this->trustProxy = $trustProxy;
        $this->publicUrl = $this->normalizePublicUrl($publicUrl);
    }

    public function show(Request $request, array $params): Response
    {
        try {
            $certificate = $this->certificates->build((string) $params['id']);
        } catch (DomainException $error) {
            return $this->view->render('error', [
                'view' => $this->view,
                'statusCode' => 404,
                'heading' => '证书未找到',
                'message' => '该证书不存在，或对应测试尚未完成',
            ], '证书未找到', 'page-error', 404);
        }

        $certificatePath = $this->view->url('/certificates/' . $certificate['attempt_id']);
        $certificateUrl = $this->publicUrl !== ''
            ? $this->publicUrl . '/certificates/' . rawurlencode((string) $certificate['attempt_id'])
            : $request->absoluteUrl($certificatePath, $this->trustProxy);
        $certificate['verification_url'] = $certificateUrl;
        $certificate['qr_data_uri'] = $this->qrCodes->dataUri($certificateUrl);

        return $this->view->render('certificate', [
            'view' => $this->view,
            'certificate' => $certificate,
        ], $certificate['title'] . ' · ' . $certificate['recipient_name'], 'page-certificate')
            ->withHeader('X-Robots-Tag', 'noindex, noarchive, noimageindex');
    }

    private function normalizePublicUrl(string $url): string
    {
        $url = rtrim(trim($url), '/');
        $parts = $url === '' ? false : parse_url($url);
        if (!is_array($parts) || !isset($parts['scheme'], $parts['host'])) {
            return '';
        }
        if (!in_array(strtolower((string) $parts['scheme']), ['http', 'https'], true) || isset($parts['user'], $parts['pass'], $parts['query'], $parts['fragment'])) {
            return '';
        }

        return $url;
    }
}
