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
    public function build(string $attemptId): array
    {
        $statement = $this->pdo->prepare("SELECT user_id FROM attempts WHERE id = :id AND status IN ('submitted', 'timed_out')");
        $statement->execute(['id' => $attemptId]);
        $userId = $statement->fetchColumn();
        if (!is_string($userId) || $userId === '') {
            throw new DomainException('证书不存在或测试尚未结束');
        }

        return $this->fromReport($this->reports->build($attemptId, $userId));
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
            'recipient_name' => (string) $report['user']['display_name'],
            'score' => $score,
            'award' => CertificateAward::forScore($score),
            'dimensions' => $dimensions,
            'number' => 'GEO-' . $submittedAt->format('ymd') . '-' . strtoupper(substr(hash('sha256', $attemptId), 0, 6)),
            'issued_on' => $submittedAt->format('Y.m.d'),
            'issuer' => 'GEO Citation Lab',
        ];
    }
}
