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
            $certificate = $this->certificates->build((string) $params['token']);
        } catch (DomainException $error) {
            return $this->view->render('error', [
                'view' => $this->view,
                'statusCode' => 404,
                'heading' => '证书未找到',
                'message' => '该证书不存在，或对应测试尚未完成',
            ], '证书未找到', 'page-error', 404);
        }

        $verificationToken = (string) $certificate['verification_token'];
        $certificatePath = $this->view->url('/certificates/' . rawurlencode($verificationToken));
        $certificateUrl = $this->publicUrl !== ''
            ? $this->publicUrl . '/certificates/' . rawurlencode($verificationToken)
            : $this->localVerificationUrl($request, $certificatePath);
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
        if ($url === '') {
            return '';
        }
        if (!is_array($parts) || !isset($parts['scheme'], $parts['host'])) {
            throw new \InvalidArgumentException('GEO_PUBLIC_URL 必须是完整的 HTTP(S) 地址。');
        }
        if (!in_array(strtolower((string) $parts['scheme']), ['http', 'https'], true) || isset($parts['user']) || isset($parts['pass']) || isset($parts['query']) || isset($parts['fragment'])) {
            throw new \InvalidArgumentException('GEO_PUBLIC_URL 格式无效。');
        }

        $host = strtolower((string) $parts['host']);
        if (strtolower((string) $parts['scheme']) !== 'https' && !$this->isLoopbackHost($host)) {
            throw new \InvalidArgumentException('公开证书地址必须使用 HTTPS。');
        }

        return $url;
    }

    private function localVerificationUrl(Request $request, string $path): string
    {
        $hostHeader = trim((string) ($request->header('host') ?? 'localhost'));
        $parsed = parse_url('http://' . $hostHeader);
        $host = is_array($parsed) && isset($parsed['host']) ? strtolower((string) $parsed['host']) : '';
        if (!$this->isLoopbackHost($host)) {
            throw new \RuntimeException('生成公开证书需要配置 GEO_PUBLIC_URL。');
        }

        return $request->absoluteUrl($path, $this->trustProxy);
    }

    private function isLoopbackHost(string $host): bool
    {
        return in_array(trim($host, '[]'), ['localhost', '127.0.0.1', '::1'], true);
    }
}
