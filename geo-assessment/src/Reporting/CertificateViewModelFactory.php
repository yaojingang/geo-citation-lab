<?php

declare(strict_types=1);

namespace GeoAssessment\Reporting;

use DomainException;
use PDO;

final class CertificateViewModelFactory
{
    private const DIMENSION_ORDER = ['mechanism', 'content', 'domestic', 'overseas', 'measurement', 'governance'];

    private const SHORT_LABELS = [
        'mechanism' => '基础理解',
        'content' => '内容证据',
        'domestic' => '国内生态',
        'overseas' => '平台来源',
        'measurement' => '测量实验',
        'governance' => '落地治理',
    ];

    /** @var PDO */
    private $pdo;

    /** @var ReportViewModelFactory */
    private $reports;

    public function __construct(PDO $pdo, ?ReportViewModelFactory $reports = null)
    {
        $this->pdo = $pdo;
        $this->reports = $reports ?? new ReportViewModelFactory($pdo);
    }

    /** @return array<string, mixed> */
    public function build(string $verificationToken): array
    {
        $statement = $this->pdo->prepare("SELECT id, user_id FROM attempts WHERE certificate_token = :token AND status IN ('submitted', 'timed_out')");
        $statement->execute(['token' => $verificationToken]);
        $attempt = $statement->fetch();
        if (!is_array($attempt)) {
            throw new DomainException('证书不存在或测试尚未结束');
        }

        return $this->fromReport($this->reports->build((string) $attempt['id'], (string) $attempt['user_id']));
    }

    /** @param array<string, mixed> $report @return array<string, mixed> */
    public function fromReport(array $report): array
    {
        $attemptId = (string) $report['attempt']['id'];
        $score = (int) $report['summary']['score'];
        $submittedAt = new \DateTimeImmutable((string) $report['attempt']['submitted_at']);
        $submittedAt = $submittedAt->setTimezone(new \DateTimeZone(date_default_timezone_get()));
        $dimensionsByKey = [];
        foreach ($report['dimensions'] as $dimension) {
            $dimensionsByKey[(string) $dimension['key']] = $dimension;
        }
        $dimensions = [];
        foreach (self::DIMENSION_ORDER as $key) {
            if (!isset($dimensionsByKey[$key])) {
                continue;
            }
            $dimension = $dimensionsByKey[$key];
            $dimensions[] = [
                'key' => $key,
                'label' => (string) $dimension['label'],
                'short_label' => self::SHORT_LABELS[$key],
                'score' => (float) $dimension['percentage'],
            ];
        }

        return [
            'title' => 'GEO专业能力测试评估证书',
            'attempt_id' => $attemptId,
            'verification_token' => $this->verificationToken($attemptId),
            'recipient_name' => (string) $report['user']['display_name'],
            'score' => $score,
            'award' => CertificateAward::forScore($score),
            'dimensions' => $dimensions,
            'number' => 'GEO-' . $submittedAt->format('ymd') . '-' . strtoupper(substr(hash('sha256', $attemptId), 0, 6)),
            'issued_on' => $submittedAt->format('Y.m.d'),
            'issuer' => 'GEO Citation Lab',
        ];
    }

    private function verificationToken(string $attemptId): string
    {
        $statement = $this->pdo->prepare('SELECT certificate_token FROM attempts WHERE id = :id');
        $statement->execute(['id' => $attemptId]);
        $token = $statement->fetchColumn();
        if (is_string($token) && $token !== '') {
            return $token;
        }

        for ($attempt = 0; $attempt < 3; $attempt++) {
            $token = rtrim(strtr(base64_encode(random_bytes(32)), '+/', '-_'), '=');
            try {
                $update = $this->pdo->prepare('UPDATE attempts SET certificate_token = :token WHERE id = :id AND certificate_token IS NULL');
                $update->execute(['token' => $token, 'id' => $attemptId]);
                if ($update->rowCount() === 1) {
                    return $token;
                }
            } catch (\PDOException $error) {
                if ($attempt === 2) {
                    throw $error;
                }
                continue;
            }

            $statement->execute(['id' => $attemptId]);
            $existing = $statement->fetchColumn();
            if (is_string($existing) && $existing !== '') {
                return $existing;
            }
        }

        throw new DomainException('证书验证标识生成失败');
    }
}
