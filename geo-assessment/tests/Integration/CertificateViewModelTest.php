<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Integration;

use DomainException;
use GeoAssessment\Assessment\AttemptService;
use GeoAssessment\Assessment\QuestionImporter;
use GeoAssessment\Identity\IdentityService;
use GeoAssessment\Http\Controller\CertificateController;
use GeoAssessment\Http\Request;
use GeoAssessment\Reporting\CertificateViewModelFactory;
use GeoAssessment\Reporting\ReportViewModelFactory;
use GeoAssessment\Support\Database;
use GeoAssessment\Support\MigrationRunner;
use GeoAssessment\Support\View;
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

        $certificate = (new CertificateViewModelFactory($this->pdo))->build($attempt['certificate_token']);

        self::assertSame('GEO专业能力测试评估证书', $certificate['title']);
        self::assertSame('姚金刚', $certificate['recipient_name']);
        self::assertSame(100, $certificate['score']);
        self::assertSame('GEO 大师', $certificate['award']['title']);
        self::assertMatchesRegularExpression('/\AGEO-\d{6}-[A-F0-9]{6}\z/', $certificate['number']);
        self::assertCount(6, $certificate['dimensions']);
        self::assertSame('GEO 基础理解', $certificate['dimensions'][0]['label']);
        self::assertSame($attempt['certificate_token'], $certificate['verification_token']);
        self::assertNotSame($certificate['attempt_id'], $certificate['verification_token']);
        self::assertArrayNotHasKey('questions', $certificate);
        self::assertArrayNotHasKey('history', $certificate);
        self::assertArrayNotHasKey('answers', $certificate);
    }

    public function test_in_progress_attempt_has_no_public_certificate(): void
    {
        $identity = (new IdentityService($this->pdo))->create('未交卷用户');
        $attempt = (new AttemptService($this->pdo))->start($identity['user']['id']);

        $this->expectException(DomainException::class);
        (new CertificateViewModelFactory($this->pdo))->build($attempt['certificate_token']);
    }

    public function test_an_existing_completed_attempt_receives_a_verification_token_when_its_report_opens(): void
    {
        $identity = (new IdentityService($this->pdo))->create('旧版证书用户');
        $attempts = new AttemptService($this->pdo);
        $attempt = $attempts->start($identity['user']['id']);
        $attempts->submit($attempt['id'], $identity['user']['id']);
        $clearToken = $this->pdo->prepare('UPDATE attempts SET certificate_token = NULL WHERE id = :id');
        $clearToken->execute(['id' => $attempt['id']]);

        $report = (new ReportViewModelFactory($this->pdo))->build($attempt['id'], $identity['user']['id']);
        $factory = new CertificateViewModelFactory($this->pdo);
        $certificate = $factory->fromReport($report);

        self::assertMatchesRegularExpression('/\A[A-Za-z0-9_-]{43}\z/', $certificate['verification_token']);
        self::assertSame($attempt['id'], $factory->build($certificate['verification_token'])['attempt_id']);
    }

    public function test_certificate_url_requires_a_canonical_public_url_for_non_loopback_hosts(): void
    {
        $identity = (new IdentityService($this->pdo))->create('证书地址测试');
        $attempts = new AttemptService($this->pdo);
        $attempt = $attempts->start($identity['user']['id']);
        $attempts->submit($attempt['id'], $identity['user']['id']);
        $controller = new CertificateController(
            new View(dirname(__DIR__, 2) . '/templates'),
            new CertificateViewModelFactory($this->pdo)
        );

        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('GEO_PUBLIC_URL');
        $controller->show(
            new Request('GET', '/certificates/' . $attempt['certificate_token'], [], [], ['host' => 'evil.example']),
            ['token' => $attempt['certificate_token']]
        );
    }

    public function test_configured_certificate_url_ignores_the_request_host(): void
    {
        $identity = (new IdentityService($this->pdo))->create('正式证书地址');
        $attempts = new AttemptService($this->pdo);
        $attempt = $attempts->start($identity['user']['id']);
        $attempts->submit($attempt['id'], $identity['user']['id']);
        $controller = new CertificateController(
            new View(dirname(__DIR__, 2) . '/templates'),
            new CertificateViewModelFactory($this->pdo),
            null,
            false,
            'https://geo.example.com'
        );

        $response = $controller->show(
            new Request('GET', '/certificates/' . $attempt['certificate_token'], [], [], ['host' => 'evil.example']),
            ['token' => $attempt['certificate_token']]
        );

        self::assertSame(200, $response->status);
        self::assertStringNotContainsString('evil.example', $response->body);
    }

    public function test_public_url_rejects_query_parameters(): void
    {
        $this->expectException(\InvalidArgumentException::class);
        new CertificateController(
            new View(dirname(__DIR__, 2) . '/templates'),
            new CertificateViewModelFactory($this->pdo),
            null,
            false,
            'https://geo.example.com?redirect=evil.example'
        );
    }
}
