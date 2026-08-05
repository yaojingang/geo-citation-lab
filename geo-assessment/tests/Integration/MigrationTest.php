<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Integration;

use GeoAssessment\Assessment\QuestionImporter;
use GeoAssessment\Support\Database;
use GeoAssessment\Support\MigrationRunner;
use PDO;
use PHPUnit\Framework\TestCase;

final class MigrationTest extends TestCase
{
    public function test_migrations_and_question_import_are_idempotent(): void
    {
        $path = sys_get_temp_dir() . '/geo-assessment-' . bin2hex(random_bytes(8)) . '.sqlite';

        try {
            $pdo = Database::connect($path);
            $migrations = dirname(__DIR__, 2) . '/database/migrations';
            $runner = new MigrationRunner($pdo, $migrations);
            self::assertSame(1, $runner->migrate());
            self::assertSame(0, $runner->migrate());

            $seed = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.1.json';
            $importer = new QuestionImporter($pdo);
            self::assertSame('imported', $importer->import($seed));
            self::assertSame('unchanged', $importer->import($seed));

            self::assertSame(30, (int) $pdo->query('SELECT COUNT(*) FROM questions')->fetchColumn());
            self::assertSame(125, (int) $pdo->query('SELECT COUNT(*) FROM choices')->fetchColumn());
            self::assertSame(1, (int) $pdo->query('SELECT COUNT(*) FROM question_sets WHERE active = 1')->fetchColumn());
            self::assertSame('ok', (string) $pdo->query('PRAGMA integrity_check')->fetchColumn());
        } finally {
            unset($pdo);
            @unlink($path);
            @unlink($path . '-shm');
            @unlink($path . '-wal');
        }
    }

    public function test_database_enforces_one_in_progress_attempt_per_user(): void
    {
        $path = sys_get_temp_dir() . '/geo-assessment-' . bin2hex(random_bytes(8)) . '.sqlite';
        try {
            $pdo = Database::connect($path);
            (new MigrationRunner($pdo, dirname(__DIR__, 2) . '/database/migrations'))->migrate();
            $pdo->exec("INSERT INTO users (id, display_name, normalized_name, created_at, last_seen_at) VALUES ('u1', '测试者', '测试者', '2026-08-04T00:00:00Z', '2026-08-04T00:00:00Z')");
            $pdo->exec("INSERT INTO question_sets (id, version, title, total_points, time_limit_seconds, scoring_version, evidence_frozen_at, content_hash, sources_json, active, created_at) VALUES ('s1', 'v1', '测试', 100, 1800, 's1', '2026-08-04', 'hash', '[]', 1, '2026-08-04T00:00:00Z')");
            $pdo->exec("INSERT INTO attempts (id, user_id, set_id, attempt_no, status, started_at, deadline_at, scoring_version, created_at, updated_at) VALUES ('a1', 'u1', 's1', 1, 'in_progress', '2026-08-04T00:00:00Z', '2026-08-04T00:30:00Z', 's1', '2026-08-04T00:00:00Z', '2026-08-04T00:00:00Z')");

            $this->expectException(\PDOException::class);
            $pdo->exec("INSERT INTO attempts (id, user_id, set_id, attempt_no, status, started_at, deadline_at, scoring_version, created_at, updated_at) VALUES ('a2', 'u1', 's1', 2, 'in_progress', '2026-08-04T00:00:00Z', '2026-08-04T00:30:00Z', 's1', '2026-08-04T00:00:00Z', '2026-08-04T00:00:00Z')");
        } finally {
            unset($pdo);
            @unlink($path);
            @unlink($path . '-shm');
            @unlink($path . '-wal');
        }
    }
}
