<?php

declare(strict_types=1);

namespace GeoAssessment\Identity;

use GeoAssessment\Support\Clock;
use GeoAssessment\Support\SystemClock;
use GeoAssessment\Support\Uuid;
use PDO;

final class IdentityService
{
    /** @var PDO */
    private $pdo;

    /** @var NameNormalizer */
    private $normalizer;

    /** @var Clock */
    private $clock;

    public function __construct(
        PDO $pdo,
        ?NameNormalizer $normalizer = null,
        ?Clock $clock = null
    ) {
        $this->pdo = $pdo;
        $this->normalizer = $normalizer ?? new NameNormalizer();
        $this->clock = $clock ?? new SystemClock();
    }

    /** @return array{user: array<string, mixed>, token: string, expires_at: string} */
    public function create(string $name): array
    {
        $normalized = $this->normalizer->normalize($name);
        $userId = Uuid::v4();
        $sessionId = Uuid::v4();
        $token = self::token();
        $now = $this->clock->now();
        $nowText = $this->format($now);
        $expiresAt = $this->format($now->modify('+10 years'));

        $this->pdo->beginTransaction();
        try {
            $user = $this->pdo->prepare('INSERT INTO users (id, display_name, normalized_name, created_at, last_seen_at) VALUES (:id, :display_name, :normalized_name, :created_at, :last_seen_at)');
            $user->execute([
                'id' => $userId,
                'display_name' => $normalized['display_name'],
                'normalized_name' => $normalized['normalized_name'],
                'created_at' => $nowText,
                'last_seen_at' => $nowText,
            ]);
            $session = $this->pdo->prepare('INSERT INTO sessions (id, user_id, token_hash, created_at, last_seen_at, expires_at) VALUES (:id, :user_id, :token_hash, :created_at, :last_seen_at, :expires_at)');
            $session->execute([
                'id' => $sessionId,
                'user_id' => $userId,
                'token_hash' => hash('sha256', $token),
                'created_at' => $nowText,
                'last_seen_at' => $nowText,
                'expires_at' => $expiresAt,
            ]);
            $this->pdo->commit();
        } catch (\Throwable $error) {
            if ($this->pdo->inTransaction()) {
                $this->pdo->rollBack();
            }
            throw $error;
        }

        return [
            'user' => array_merge(['id' => $userId], $normalized, ['created_at' => $nowText]),
            'token' => $token,
            'expires_at' => $expiresAt,
        ];
    }

    /** @return array<string, mixed>|null */
    public function resolve(?string $token): ?array
    {
        if ($token === null || $token === '') {
            return null;
        }
        $statement = $this->pdo->prepare('SELECT u.*, s.id AS session_id, s.last_seen_at AS session_seen_at FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = :hash AND s.expires_at > :now');
        $statement->execute(['hash' => hash('sha256', $token), 'now' => $this->format($this->clock->now())]);
        $user = $statement->fetch();
        if (!is_array($user)) {
            return null;
        }

        $lastSeen = new \DateTimeImmutable((string) $user['session_seen_at']);
        if ($lastSeen < $this->clock->now()->modify('-24 hours')) {
            $now = $this->format($this->clock->now());
            $expires = $this->format($this->clock->now()->modify('+10 years'));
            $update = $this->pdo->prepare('UPDATE sessions SET last_seen_at = :now, expires_at = :expires WHERE id = :id');
            $update->execute(['now' => $now, 'expires' => $expires, 'id' => $user['session_id']]);
            $this->pdo->prepare('UPDATE users SET last_seen_at = :now WHERE id = :id')->execute(['now' => $now, 'id' => $user['id']]);
        }

        unset($user['session_seen_at']);
        return $user;
    }

    public function revoke(string $token): void
    {
        $statement = $this->pdo->prepare('DELETE FROM sessions WHERE token_hash = :hash');
        $statement->execute(['hash' => hash('sha256', $token)]);
    }

    public function deleteUser(string $userId): void
    {
        $statement = $this->pdo->prepare('DELETE FROM users WHERE id = :id');
        $statement->execute(['id' => $userId]);
    }

    private static function token(): string
    {
        return rtrim(strtr(base64_encode(random_bytes(32)), '+/', '-_'), '=');
    }

    private function format(\DateTimeImmutable $date): string
    {
        return $date->setTimezone(new \DateTimeZone('UTC'))->format('Y-m-d\TH:i:s\Z');
    }
}
