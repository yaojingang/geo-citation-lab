<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Integration;

use DomainException;
use GeoAssessment\Assessment\AttemptService;
use GeoAssessment\Assessment\QuestionImporter;
use GeoAssessment\Identity\IdentityService;
use GeoAssessment\Support\Clock;
use GeoAssessment\Support\Database;
use GeoAssessment\Support\MigrationRunner;
use DateTimeImmutable;
use PDO;
use PHPUnit\Framework\TestCase;

final class AttemptLifecycleTest extends TestCase
{
    private string $path;
    private PDO $pdo;
    private IdentityService $identity;
    private AttemptService $attempts;

    protected function setUp(): void
    {
        $this->path = sys_get_temp_dir() . '/geo-assessment-' . bin2hex(random_bytes(8)) . '.sqlite';
        $this->pdo = Database::connect($this->path);
        (new MigrationRunner($this->pdo, dirname(__DIR__, 2) . '/database/migrations'))->migrate();
        (new QuestionImporter($this->pdo))->import(dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json');
        $this->identity = new IdentityService($this->pdo);
        $this->attempts = new AttemptService($this->pdo);
    }

    protected function tearDown(): void
    {
        unset($this->pdo);
        @unlink($this->path);
        @unlink($this->path . '-shm');
        @unlink($this->path . '-wal');
    }

    public function test_same_names_create_isolated_browser_identities(): void
    {
        $first = $this->identity->create('王小明');
        $second = $this->identity->create('王小明');

        self::assertNotSame($first['user']['id'], $second['user']['id']);
        self::assertNotSame($first['token'], $second['token']);
        self::assertSame($first['user']['id'], $this->identity->resolve($first['token'])['id']);
        self::assertSame($second['user']['id'], $this->identity->resolve($second['token'])['id']);
    }

    public function test_attempt_snapshot_autosave_and_scoring_form_a_complete_flow(): void
    {
        $identity = $this->identity->create('林之谦');
        $attempt = $this->attempts->start($identity['user']['id']);

        self::assertSame(1, $attempt['attempt_no']);
        self::assertCount(30, $this->attempts->items($attempt['id'], $identity['user']['id']));
        self::assertSame($attempt['id'], $this->attempts->start($identity['user']['id'])['id']);

        $firstItem = $this->attempts->itemAt($attempt['id'], $identity['user']['id'], 1);
        $snapshot = $firstItem['snapshot'];
        self::assertContains($snapshot['region_scope'], ['domestic', 'general', 'overseas']);
        self::assertSame(range('A', chr(64 + count($snapshot['choices']))), array_column($snapshot['choices'], 'code'));
        $saved = $this->attempts->saveAnswer($attempt['id'], $identity['user']['id'], $snapshot['code'], [$snapshot['correct_codes'][0]], 2, 8);
        self::assertFalse($saved['stale']);
        $stale = $this->attempts->saveAnswer($attempt['id'], $identity['user']['id'], $snapshot['code'], [], 1, 5);
        self::assertTrue($stale['stale']);
        self::assertSame([$snapshot['correct_codes'][0]], $stale['selected_codes']);

        foreach ($this->attempts->items($attempt['id'], $identity['user']['id']) as $item) {
            $this->attempts->saveAnswer($attempt['id'], $identity['user']['id'], $item['snapshot']['code'], $item['snapshot']['correct_codes'], 10 + $item['position'], 1);
        }
        $result = $this->attempts->submit($attempt['id'], $identity['user']['id']);

        self::assertSame('submitted', $result['status']);
        self::assertSame(100, $result['score']);
        self::assertSame(30, $result['correct_count']);
        self::assertSame($result, $this->attempts->submit($attempt['id'], $identity['user']['id']));
    }

    public function test_activity_time_is_clamped_and_unchanged_answers_do_not_inflate_change_count(): void
    {
        $identity = $this->identity->create('活跃用时样本');
        $attempt = $this->attempts->start($identity['user']['id']);
        $item = $this->attempts->itemAt($attempt['id'], $identity['user']['id'], 1);
        $answer = [$item['snapshot']['choices'][0]['code']];

        $this->attempts->saveAnswer($attempt['id'], $identity['user']['id'], $item['snapshot']['code'], $answer, 2, 100);
        $afterFirstSave = $this->attempts->itemAt($attempt['id'], $identity['user']['id'], 1);
        self::assertSame(30, $afterFirstSave['time_spent_seconds']);
        self::assertSame(1, (int) $afterFirstSave['change_count']);

        $this->attempts->saveAnswer($attempt['id'], $identity['user']['id'], $item['snapshot']['code'], $answer, 3, 100);
        $afterSecondSave = $this->attempts->itemAt($attempt['id'], $identity['user']['id'], 1);
        self::assertSame(60, $afterSecondSave['time_spent_seconds']);
        self::assertSame(1, (int) $afterSecondSave['change_count']);
    }

    public function test_delayed_save_cannot_mutate_an_attempt_that_finishes_during_the_request(): void
    {
        $identity = $this->identity->create('并发保存样本');
        $attempt = $this->attempts->start($identity['user']['id']);
        $item = $this->attempts->itemAt($attempt['id'], $identity['user']['id'], 1);
        $clock = new class($this->pdo, $attempt['id']) implements Clock {
            private bool $triggered = false;

            public function __construct(private readonly PDO $pdo, private readonly string $attemptId)
            {
            }

            public function now(): DateTimeImmutable
            {
                if (!$this->triggered) {
                    $this->triggered = true;
                    $statement = $this->pdo->prepare("UPDATE attempts SET status = 'submitted', score = 0, correct_count = 0, duration_seconds = 0, submitted_at = :now, updated_at = :now WHERE id = :id");
                    $statement->execute(['now' => '2026-08-04 10:00:00', 'id' => $this->attemptId]);
                }
                return new DateTimeImmutable('2026-08-04 10:00:01+00:00');
            }
        };
        $service = new AttemptService($this->pdo, clock: $clock);

        try {
            $service->saveAnswer($attempt['id'], $identity['user']['id'], $item['snapshot']['code'], [$item['snapshot']['choices'][0]['code']], 2, 10);
            self::fail('预期已结束的测试拒绝延迟保存。');
        } catch (DomainException $error) {
            self::assertSame('该测试已结束。', $error->getMessage());
        }

        $stored = $this->attempts->itemAt($attempt['id'], $identity['user']['id'], 1);
        self::assertSame([], $stored['selected_codes']);
        self::assertSame(0, (int) $stored['activity_seq']);
    }

    public function test_each_identity_is_limited_to_ten_attempts(): void
    {
        $identity = $this->identity->create('周素琴');
        for ($number = 1; $number <= 10; $number++) {
            $attempt = $this->attempts->start($identity['user']['id']);
            self::assertSame($number, $attempt['attempt_no']);
            $this->attempts->submit($attempt['id'], $identity['user']['id']);
        }

        $this->expectException(DomainException::class);
        $this->attempts->start($identity['user']['id']);
    }
}
