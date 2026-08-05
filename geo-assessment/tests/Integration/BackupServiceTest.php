<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Integration;

use GeoAssessment\Assessment\QuestionImporter;
use GeoAssessment\Support\BackupService;
use GeoAssessment\Support\Database;
use GeoAssessment\Support\MigrationRunner;
use PDO;
use PHPUnit\Framework\TestCase;

final class BackupServiceTest extends TestCase
{
    public function testCreatesAndVerifiesConsistentBackup(): void
    {
        $directory = sys_get_temp_dir() . '/geo-backup-' . bin2hex(random_bytes(5));
        mkdir($directory, 0770, true);
        $databasePath = $directory . '/app.sqlite';
        $pdo = Database::connect($databasePath);
        (new MigrationRunner($pdo, dirname(__DIR__, 2) . '/database/migrations'))->migrate();
        (new QuestionImporter($pdo))->import(dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.1.json');

        $service = new BackupService($databasePath, $directory . '/backups');
        $backup = $service->create();

        self::assertFileExists($backup);
        self::assertFileExists($backup . '.sha256');
        self::assertSame('ok', $service->verify($backup)['integrity']);
    }

    public function testLegacySqliteCopyPathCreatesConsistentBackupAndRestoresWal(): void
    {
        $directory = sys_get_temp_dir() . '/geo-backup-legacy-' . bin2hex(random_bytes(5));
        mkdir($directory, 0770, true);
        $databasePath = $directory . '/app.sqlite';
        $pdo = Database::connect($databasePath);
        (new MigrationRunner($pdo, dirname(__DIR__, 2) . '/database/migrations'))->migrate();
        (new QuestionImporter($pdo))->import(dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.1.json');

        $target = $directory . '/legacy-copy.sqlite';
        $service = new BackupService($databasePath, $directory . '/backups');
        $method = new \ReflectionMethod($service, 'copyConsistently');
        $method->setAccessible(true);
        $method->invoke($service, $pdo, $target);

        self::assertFileExists($target);
        self::assertSame('wal', strtolower((string) $pdo->query('PRAGMA journal_mode')->fetchColumn()));
        $backup = new PDO('sqlite:' . $target);
        self::assertSame('ok', $backup->query('PRAGMA integrity_check')->fetchColumn());
        self::assertSame(30, (int) $backup->query('SELECT COUNT(*) FROM questions')->fetchColumn());
    }
}
