<?php

declare(strict_types=1);

namespace GeoAssessment\Http;

use PDO;

final class RateLimiter
{
    /** @var PDO */
    private $pdo;

    /** @var string */
    private $appKey;

    public function __construct(PDO $pdo, string $appKey)
    {
        $this->pdo = $pdo;
        $this->appKey = $appKey;
    }

    public function allow(string $identity, string $action, int $limit, int $windowSeconds): bool
    {
        $now = time();
        $windowStart = intdiv($now, $windowSeconds) * $windowSeconds;
        $windowText = gmdate('Y-m-d\TH:i:s\Z', $windowStart);
        $expires = gmdate('Y-m-d\TH:i:s\Z', $windowStart + $windowSeconds);
        $keyHash = hash_hmac('sha256', $identity, $this->appKey);
        $parameters = ['key_hash' => $keyHash, 'action' => $action, 'window_started_at' => $windowText];

        $this->pdo->exec('BEGIN IMMEDIATE');
        $transactionStarted = true;
        try {
            $this->pdo->prepare('DELETE FROM rate_limits WHERE expires_at < :now')->execute(['now' => gmdate('Y-m-d\TH:i:s\Z', $now)]);
            $update = $this->pdo->prepare('UPDATE rate_limits SET count = count + 1, expires_at = :expires_at WHERE key_hash = :key_hash AND action = :action AND window_started_at = :window_started_at');
            $update->execute($parameters + ['expires_at' => $expires]);
            if ($update->rowCount() === 0) {
                $insert = $this->pdo->prepare('INSERT INTO rate_limits (key_hash, action, window_started_at, count, expires_at) VALUES (:key_hash, :action, :window_started_at, 1, :expires_at)');
                $insert->execute($parameters + ['expires_at' => $expires]);
            }
            $statement = $this->pdo->prepare('SELECT count FROM rate_limits WHERE key_hash = :key_hash AND action = :action AND window_started_at = :window_started_at');
            $statement->execute($parameters);
            $count = (int) $statement->fetchColumn();
            $this->pdo->exec('COMMIT');
            $transactionStarted = false;
        } catch (\Throwable $error) {
            if ($transactionStarted) {
                $this->pdo->exec('ROLLBACK');
            }
            throw $error;
        }

        return $count <= $limit;
    }
}
