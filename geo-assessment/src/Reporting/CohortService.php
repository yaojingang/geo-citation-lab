<?php

declare(strict_types=1);

namespace GeoAssessment\Reporting;

use PDO;

final class CohortService
{
    /** @var PDO */
    private $pdo;

    public function __construct(PDO $pdo)
    {
        $this->pdo = $pdo;
    }

    /** @return array<string, mixed> */
    public function position(string $setId, int $score): array
    {
        $statement = $this->pdo->prepare("SELECT current.score
            FROM attempts current
            WHERE current.set_id = :set_id
              AND current.status IN ('submitted', 'timed_out')
              AND current.score IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM attempts later
                  WHERE later.user_id = current.user_id
                    AND later.set_id = current.set_id
                    AND later.status IN ('submitted', 'timed_out')
                    AND later.score IS NOT NULL
                    AND (
                        (later.submitted_at IS NOT NULL AND current.submitted_at IS NULL)
                        OR later.submitted_at > current.submitted_at
                        OR (
                            (
                                later.submitted_at = current.submitted_at
                                OR (later.submitted_at IS NULL AND current.submitted_at IS NULL)
                            )
                            AND later.attempt_no > current.attempt_no
                        )
                    )
              )");
        $statement->execute(['set_id' => $setId]);
        $scores = array_map('intval', array_column($statement->fetchAll(), 'score'));
        $sampleSize = count($scores);
        if ($sampleSize < 50) {
            return ['visible' => false, 'sample_size' => $sampleSize, 'reason' => '同版本完成用户少于 50 人'];
        }

        $atOrBelow = count(array_filter($scores, static function (int $value) use ($score): bool {
            return $value <= $score;
        }));
        $percentile = round(($atOrBelow / $sampleSize) * 100, 1);
        $bands = $sampleSize >= 200 ? 10 : 4;
        $band = max(1, min($bands, (int) ceil(($percentile / 100) * $bands)));

        return [
            'visible' => true,
            'sample_size' => $sampleSize,
            'percentile' => $percentile,
            'band_count' => $bands,
            'band' => $band,
            'label' => $bands === 10 ? "第 {$band} 十分位组" : "第 {$band} 四分位组",
            'calculated_at' => gmdate('Y-m-d\TH:i:s\Z'),
        ];
    }
}
