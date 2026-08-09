<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Support\MigrationRunner;
use PDO;
use PDOException;
use PDOStatement;
use PHPUnit\Framework\TestCase;

final class MigrationRunnerTest extends TestCase
{
    public function test_read_cursor_is_closed_before_schema_migration_begins(): void
    {
        $directory = sys_get_temp_dir() . '/geo-migration-' . bin2hex(random_bytes(8));
        mkdir($directory, 0700);
        file_put_contents($directory . '/001_example.sql', 'ALTER TABLE questions ADD COLUMN region_scope TEXT;');

        try {
            $pdo = new LockSensitivePdo();

            self::assertSame(1, (new MigrationRunner($pdo, $directory))->migrate());
            self::assertFalse($pdo->cursorOpen);
            self::assertContains('ALTER TABLE questions ADD COLUMN region_scope TEXT;', $pdo->executedSql);
        } finally {
            @unlink($directory . '/001_example.sql');
            @rmdir($directory);
        }
    }
}

final class LockSensitivePdo extends PDO
{
    public bool $cursorOpen = false;

    /** @var list<string> */
    public array $executedSql = [];

    public function __construct()
    {
    }

    public function exec(string $statement): int|false
    {
        $this->executedSql[] = $statement;

        return 0;
    }

    public function prepare(string $query, array $options = []): PDOStatement|false
    {
        $tracksCursor = strpos($query, 'SELECT checksum') === 0;
        if ($tracksCursor) {
            $this->cursorOpen = true;
        }

        return new CursorTrackingStatement($this, $tracksCursor);
    }

    public function beginTransaction(): bool
    {
        if ($this->cursorOpen) {
            throw new PDOException('database is locked by an open read cursor');
        }

        return true;
    }

    public function commit(): bool
    {
        return true;
    }
}

final class CursorTrackingStatement extends PDOStatement
{
    private LockSensitivePdo $pdo;

    private bool $tracksCursor;

    public function __construct(LockSensitivePdo $pdo, bool $tracksCursor)
    {
        $this->pdo = $pdo;
        $this->tracksCursor = $tracksCursor;
    }

    public function execute(?array $params = null): bool
    {
        return true;
    }

    public function fetchColumn(int $column = 0): mixed
    {
        return false;
    }

    public function closeCursor(): bool
    {
        if ($this->tracksCursor) {
            $this->pdo->cursorOpen = false;
        }

        return true;
    }
}
