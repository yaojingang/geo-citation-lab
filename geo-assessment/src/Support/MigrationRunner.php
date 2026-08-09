<?php

declare(strict_types=1);

namespace GeoAssessment\Support;

use PDO;
use RuntimeException;

final class MigrationRunner
{
    /** @var PDO */
    private $pdo;

    /** @var string */
    private $directory;

    public function __construct(PDO $pdo, string $directory)
    {
        $this->pdo = $pdo;
        $this->directory = $directory;
    }

    public function migrate(): int
    {
        $this->pdo->exec('CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, checksum TEXT NOT NULL, applied_at TEXT NOT NULL)');
        $files = glob(rtrim($this->directory, '/') . '/*.sql') ?: [];
        sort($files, SORT_STRING);
        $applied = 0;

        foreach ($files as $file) {
            $version = pathinfo($file, PATHINFO_FILENAME);
            $sql = file_get_contents($file);
            if (!is_string($sql)) {
                throw new RuntimeException("无法读取迁移：{$file}");
            }
            $checksum = hash('sha256', $sql);
            $statement = $this->pdo->prepare('SELECT checksum FROM schema_migrations WHERE version = :version');
            $statement->execute(['version' => $version]);
            $knownChecksum = $statement->fetchColumn();
            $statement->closeCursor();
            if (is_string($knownChecksum)) {
                if (!hash_equals($knownChecksum, $checksum)) {
                    throw new RuntimeException("迁移 {$version} 的校验和已变更。");
                }
                continue;
            }

            $this->pdo->beginTransaction();
            try {
                $this->pdo->exec($sql);
                $insert = $this->pdo->prepare('INSERT INTO schema_migrations (version, checksum, applied_at) VALUES (:version, :checksum, :applied_at)');
                $insert->execute(['version' => $version, 'checksum' => $checksum, 'applied_at' => gmdate('Y-m-d\TH:i:s\Z')]);
                $this->pdo->commit();
                $applied++;
            } catch (\Throwable $error) {
                if ($this->pdo->inTransaction()) {
                    $this->pdo->rollBack();
                }
                throw $error;
            }
        }

        return $applied;
    }
}
