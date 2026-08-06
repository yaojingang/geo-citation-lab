<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Integration;

use GeoAssessment\Assessment\QuestionImporter;
use GeoAssessment\Support\BackupService;
use GeoAssessment\Support\Database;
use GeoAssessment\Support\MigrationRunner;
use PDO;
use PDOStatement;
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
        (new QuestionImporter($pdo))->import(dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json');

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
        (new QuestionImporter($pdo))->import(dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json');

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

    public function testLegacyBackupReusesTheMigrationConnection(): void
    {
        $directory = sys_get_temp_dir() . '/geo-backup-shared-connection-' . bin2hex(random_bytes(5));
        mkdir($directory, 0770, true);
        $databasePath = $directory . '/app.sqlite';
        $setup = Database::connect($databasePath);
        (new MigrationRunner($setup, dirname(__DIR__, 2) . '/database/migrations'))->migrate();
        (new QuestionImporter($setup))->import(dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json');
        unset($setup);

        $pdo = new LegacyVersionPdo($databasePath);
        $service = new BackupService($databasePath, $directory . '/backups');
        $backup = $service->create($pdo);

        self::assertTrue($pdo->deleteModeRequested);
        self::assertFileExists($backup);
        self::assertSame('ok', $service->verify($backup)['integrity']);
    }
}

final class LegacyVersionPdo extends PDO
{
    public bool $deleteModeRequested = false;

    public function __construct(string $path)
    {
        parent::__construct('sqlite:' . $path, null, null, [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]);
        $this->exec('PRAGMA busy_timeout = 5000');
        $this->exec('PRAGMA journal_mode = WAL');
    }

    public function query(string $query, ?int $fetchMode = null, mixed ...$fetchModeArgs): PDOStatement|false
    {
        if ($query === 'SELECT sqlite_version()') {
            return new ScalarStatement('3.24.0');
        }
        if ($query === 'PRAGMA journal_mode = DELETE') {
            $this->deleteModeRequested = true;
        }

        return parent::query($query);
    }
}

final class ScalarStatement extends PDOStatement
{
    private string $value;

    public function __construct(string $value)
    {
        $this->value = $value;
    }

    public function fetchColumn(int $column = 0): mixed
    {
        return $this->value;
    }
}
