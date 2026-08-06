<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Integration;

use DomainException;
use GeoAssessment\Assessment\AttemptService;
use GeoAssessment\Assessment\QuestionImporter;
use GeoAssessment\Identity\IdentityService;
use GeoAssessment\Reporting\CertificateViewModelFactory;
use GeoAssessment\Support\Database;
use GeoAssessment\Support\MigrationRunner;
use PDO;
use PHPUnit\Framework\TestCase;

final class CertificateViewModelTest extends TestCase
{
    /** @var string */
    private $path;

    /** @var PDO */
    private $pdo;

    protected function setUp(): void
    {
        $this->path = sys_get_temp_dir() . '/geo-certificate-' . bin2hex(random_bytes(8)) . '.sqlite';
        $this->pdo = Database::connect($this->path);
        (new MigrationRunner($this->pdo, dirname(__DIR__, 2) . '/database/migrations'))->migrate();
        (new QuestionImporter($this->pdo))->import(dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json');
    }

    protected function tearDown(): void
    {
        unset($this->pdo);
        @unlink($this->path);
        @unlink($this->path . '-shm');
        @unlink($this->path . '-wal');
    }

    public function test_completed_attempt_builds_a_public_safe_certificate_view_model(): void
    {
        $identity = (new IdentityService($this->pdo))->create('姚金刚');
        $attempts = new AttemptService($this->pdo);
        $attempt = $attempts->start($identity['user']['id']);
        foreach ($attempts->items($attempt['id'], $identity['user']['id']) as $item) {
            $attempts->saveAnswer($attempt['id'], $identity['user']['id'], $item['snapshot']['code'], $item['snapshot']['correct_codes'], 10, 1);
        }
        $attempts->submit($attempt['id'], $identity['user']['id']);

        $certificate = (new CertificateViewModelFactory($this->pdo))->build($attempt['id']);

        self::assertSame('GEO专业能力测试评估证书', $certificate['title']);
        self::assertSame('姚金刚', $certificate['recipient_name']);
        self::assertSame(100, $certificate['score']);
        self::assertSame('GEO 大师', $certificate['award']['title']);
        self::assertMatchesRegularExpression('/\AGEO-\d{6}-[A-F0-9]{6}\z/', $certificate['number']);
        self::assertCount(6, $certificate['dimensions']);
        self::assertSame('GEO 基础理解', $certificate['dimensions'][0]['label']);
        self::assertArrayNotHasKey('questions', $certificate);
        self::assertArrayNotHasKey('history', $certificate);
        self::assertArrayNotHasKey('answers', $certificate);
    }

    public function test_in_progress_attempt_has_no_public_certificate(): void
    {
        $identity = (new IdentityService($this->pdo))->create('未交卷用户');
        $attempt = (new AttemptService($this->pdo))->start($identity['user']['id']);

        $this->expectException(DomainException::class);
        (new CertificateViewModelFactory($this->pdo))->build($attempt['id']);
    }
}
