<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Integration;

use DomainException;
use GeoAssessment\Assessment\AttemptService;
use GeoAssessment\Assessment\QuestionImporter;
use GeoAssessment\Identity\IdentityService;
use GeoAssessment\Reporting\ReportViewModelFactory;
use GeoAssessment\Support\Database;
use GeoAssessment\Support\MigrationRunner;
use PDO;
use PHPUnit\Framework\TestCase;

final class PrivacyAndOwnershipTest extends TestCase
{
    private string $path;
    private PDO $pdo;

    protected function setUp(): void
    {
        $this->path = sys_get_temp_dir() . '/geo-privacy-' . bin2hex(random_bytes(8)) . '.sqlite';
        $this->pdo = Database::connect($this->path);
        (new MigrationRunner($this->pdo, dirname(__DIR__, 2) . '/database/migrations'))->migrate();
        (new QuestionImporter($this->pdo))->import(dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.1.json');
    }

    protected function tearDown(): void
    {
        unset($this->pdo);
        @unlink($this->path);
        @unlink($this->path . '-shm');
        @unlink($this->path . '-wal');
    }

    public function testAnotherIdentityCannotReadAttemptOrReport(): void
    {
        $identities = new IdentityService($this->pdo);
        $owner = $identities->create('报告所有者');
        $stranger = $identities->create('其他测试者');
        $attempts = new AttemptService($this->pdo);
        $attempt = $attempts->start($owner['user']['id']);
        $attempts->submit($attempt['id'], $owner['user']['id']);

        try {
            $attempts->getAttempt($attempt['id'], $stranger['user']['id']);
            self::fail('其他身份不应读取测试。');
        } catch (DomainException) {
            self::assertTrue(true);
        }

        $this->expectException(DomainException::class);
        (new ReportViewModelFactory($this->pdo))->build($attempt['id'], $stranger['user']['id']);
    }

    public function testDeletingUserCascadesSessionsAttemptsAndAnswers(): void
    {
        $identities = new IdentityService($this->pdo);
        $identity = $identities->create('删除验收用户');
        $attempt = (new AttemptService($this->pdo))->start($identity['user']['id']);

        $identities->deleteUser($identity['user']['id']);

        self::assertSame(0, $this->countRows('SELECT COUNT(*) FROM users WHERE id = :id', $identity['user']['id']));
        self::assertSame(0, $this->countRows('SELECT COUNT(*) FROM sessions WHERE user_id = :id', $identity['user']['id']));
        self::assertSame(0, $this->countRows('SELECT COUNT(*) FROM attempts WHERE id = :id', $attempt['id']));
        self::assertSame(0, $this->countRows('SELECT COUNT(*) FROM attempt_items WHERE attempt_id = :id', $attempt['id']));
    }

    private function countRows(string $sql, string $id): int
    {
        $statement = $this->pdo->prepare($sql);
        $statement->execute(['id' => $id]);
        return (int) $statement->fetchColumn();
    }
}
