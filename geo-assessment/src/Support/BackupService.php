<?php

declare(strict_types=1);

namespace GeoAssessment\Support;

use PDO;
use RuntimeException;

final class BackupService
{
    /** @var string */
    private $databasePath;

    /** @var string */
    private $backupDirectory;

    public function __construct(string $databasePath, string $backupDirectory)
    {
        $this->databasePath = $databasePath;
        $this->backupDirectory = $backupDirectory;
    }

    public function create(?PDO $pdo = null): string
    {
        if (!is_file($this->databasePath)) {
            throw new RuntimeException('数据库尚未创建。');
        }
        $this->ensureDirectory($this->backupDirectory);
        $name = 'geo-assessment-' . gmdate('Ymd-His') . '-' . bin2hex(random_bytes(3)) . '.sqlite';
        $target = rtrim($this->backupDirectory, '/') . '/' . $name;
        $pdo = $pdo ?? Database::connect($this->databasePath);
        $sqliteVersion = (string) $pdo->query('SELECT sqlite_version()')->fetchColumn();
        if (version_compare($sqliteVersion, '3.27.0', '>=')) {
            $pdo->exec('PRAGMA wal_checkpoint(FULL)');
            $pdo->exec('VACUUM INTO ' . $pdo->quote($target));
        } else {
            $this->copyConsistently($pdo, $target);
        }
        chmod($target, 0600);
        $hash = hash_file('sha256', $target);
        if (!is_string($hash)) {
            throw new RuntimeException('无法计算备份校验和。');
        }
        file_put_contents($target . '.sha256', $hash . '  ' . basename($target) . PHP_EOL, LOCK_EX);
        chmod($target . '.sha256', 0600);
        $this->verify($target);
        $this->prune(7);
        return $target;
    }

    /** @return array{path: string, checksum: string, integrity: string, tables: int} */
    public function verify(string $backupPath): array
    {
        if (!is_file($backupPath) || !is_file($backupPath . '.sha256')) {
            throw new RuntimeException('备份或校验和文件不存在。');
        }
        chmod($backupPath, 0600);
        chmod($backupPath . '.sha256', 0600);
        $expected = substr(trim((string) file_get_contents($backupPath . '.sha256')), 0, 64);
        $actual = hash_file('sha256', $backupPath);
        if (!is_string($actual) || !hash_equals($expected, $actual)) {
            throw new RuntimeException('备份校验和不匹配。');
        }
        $pdo = new PDO('sqlite:' . $backupPath, null, null, [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION, PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC]);
        $integrity = (string) $pdo->query('PRAGMA integrity_check')->fetchColumn();
        if ($integrity !== 'ok') {
            throw new RuntimeException('备份完整性检查失败：' . $integrity);
        }
        $tables = (int) $pdo->query("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")->fetchColumn();
        if ($tables < 9) {
            throw new RuntimeException('备份缺少必需数据表。');
        }
        $activeSets = (int) $pdo->query('SELECT COUNT(*) FROM question_sets WHERE active = 1')->fetchColumn();
        $questions = (int) $pdo->query('SELECT COUNT(*) FROM questions q JOIN question_sets qs ON qs.id = q.set_id WHERE qs.active = 1')->fetchColumn();
        $points = (int) $pdo->query('SELECT COALESCE(SUM(q.weight), 0) FROM questions q JOIN question_sets qs ON qs.id = q.set_id WHERE qs.active = 1')->fetchColumn();
        if ($activeSets !== 1 || $questions !== 30 || $points !== 100) {
            throw new RuntimeException('备份的活动题集业务约束失效。');
        }
        return ['path' => $backupPath, 'checksum' => $actual, 'integrity' => $integrity, 'tables' => $tables];
    }

    private function prune(int $keep): void
    {
        $files = glob(rtrim($this->backupDirectory, '/') . '/*.sqlite') ?: [];
        rsort($files, SORT_STRING);
        foreach (array_slice($files, $keep) as $file) {
            if (is_file($file)) {
                unlink($file);
            }
            if (is_file($file . '.sha256')) {
                unlink($file . '.sha256');
            }
        }
    }

    private function ensureDirectory(string $directory): void
    {
        if (!is_dir($directory) && !mkdir($directory, 0770, true) && !is_dir($directory)) {
            throw new RuntimeException("无法创建目录：{$directory}");
        }
    }

    private function copyConsistently(PDO $pdo, string $target): void
    {
        $journalMode = strtolower((string) $pdo->query('PRAGMA journal_mode')->fetchColumn());
        if ($journalMode === 'wal') {
            $pdo->exec('PRAGMA wal_checkpoint(FULL)');
        }
        $copyMode = strtolower((string) $pdo->query('PRAGMA journal_mode = DELETE')->fetchColumn());
        if ($copyMode !== 'delete') {
            throw new RuntimeException('无法为 SQLite 3.24–3.26 获取一致备份锁。');
        }

        $transactionStarted = false;
        $failure = null;
        try {
            $pdo->exec('BEGIN IMMEDIATE');
            $transactionStarted = true;
            if (!copy($this->databasePath, $target)) {
                throw new RuntimeException('无法创建数据库备份。');
            }
            $pdo->exec('COMMIT');
            $transactionStarted = false;
        } catch (\Throwable $error) {
            if ($transactionStarted) {
                $pdo->exec('ROLLBACK');
            }
            if (is_file($target)) {
                unlink($target);
            }
            $failure = $error;
        }

        if ($journalMode === 'wal') {
            try {
                $restoredMode = strtolower((string) $pdo->query('PRAGMA journal_mode = WAL')->fetchColumn());
                if ($restoredMode !== 'wal' && $failure === null) {
                    $failure = new RuntimeException('备份完成，但无法恢复 SQLite WAL 日志模式。');
                }
            } catch (\Throwable $restoreError) {
                if ($failure === null) {
                    $failure = $restoreError;
                }
            }
            if ($failure !== null) {
                if (is_file($target)) {
                    unlink($target);
                }
            }
        }

        if ($failure !== null) {
            throw $failure;
        }
    }

}
