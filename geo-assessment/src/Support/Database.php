<?php

declare(strict_types=1);

namespace GeoAssessment\Support;

use PDO;

final class Database
{
    public static function connect(string $path): PDO
    {
        $isNew = $path !== ':memory:' && !is_file($path);
        if ($path !== ':memory:') {
            $directory = dirname($path);
            if (!is_dir($directory) && !mkdir($directory, 0770, true) && !is_dir($directory)) {
                throw new \RuntimeException("无法创建数据库目录：{$directory}");
            }
        }

        $pdo = new PDO('sqlite:' . $path, null, null, [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
            PDO::ATTR_EMULATE_PREPARES => false,
        ]);
        $pdo->exec('PRAGMA foreign_keys = ON');
        $pdo->exec('PRAGMA busy_timeout = 5000');
        if ($path !== ':memory:') {
            $pdo->exec('PRAGMA journal_mode = WAL');
            if ($isNew) {
                chmod($path, 0660);
            }
        }

        return $pdo;
    }
}
