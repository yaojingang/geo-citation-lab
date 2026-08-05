<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Integration;

use GeoAssessment\Reporting\CohortService;
use PDO;
use PHPUnit\Framework\TestCase;

final class CohortServiceTest extends TestCase
{
    public function test_latest_completed_attempt_is_selected_when_legacy_timestamp_is_null(): void
    {
        $pdo = new PDO('sqlite::memory:', null, null, [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]);
        $pdo->exec('CREATE TABLE attempts (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            set_id TEXT NOT NULL,
            attempt_no INTEGER NOT NULL,
            status TEXT NOT NULL,
            submitted_at TEXT,
            score INTEGER
        )');
        $pdo->exec("INSERT INTO attempts VALUES
            ('legacy', 'user-1', 'set-1', 1, 'submitted', NULL, 10),
            ('current', 'user-1', 'set-1', 2, 'submitted', '2026-08-05T06:00:00Z', 20)");

        $position = (new CohortService($pdo))->position('set-1', 20);

        self::assertFalse($position['visible']);
        self::assertSame(1, $position['sample_size']);
    }
}
