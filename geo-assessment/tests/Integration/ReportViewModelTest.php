<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Integration;

use GeoAssessment\Assessment\AttemptService;
use GeoAssessment\Assessment\QuestionImporter;
use GeoAssessment\Identity\IdentityService;
use GeoAssessment\Reporting\ReportViewModelFactory;
use GeoAssessment\Support\Database;
use GeoAssessment\Support\MigrationRunner;
use PDO;
use PHPUnit\Framework\TestCase;

final class ReportViewModelTest extends TestCase
{
    public function test_report_contains_all_diagnostic_and_visualization_views(): void
    {
        $path = sys_get_temp_dir() . '/geo-assessment-' . bin2hex(random_bytes(8)) . '.sqlite';
        try {
            $pdo = Database::connect($path);
            (new MigrationRunner($pdo, dirname(__DIR__, 2) . '/database/migrations'))->migrate();
            (new QuestionImporter($pdo))->import(dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json');
            $identity = (new IdentityService($pdo))->create('陈星河');
            $attempts = new AttemptService($pdo);
            $attempt = $attempts->start($identity['user']['id']);
            $items = $attempts->items($attempt['id'], $identity['user']['id']);

            foreach (array_slice($items, 0, 15) as $item) {
                $attempts->saveAnswer($attempt['id'], $identity['user']['id'], $item['snapshot']['code'], $item['snapshot']['correct_codes'], 100 + $item['position'], 12);
            }
            foreach (array_slice($items, 15, 5) as $item) {
                $attempts->saveAnswer($attempt['id'], $identity['user']['id'], $item['snapshot']['code'], [$item['snapshot']['choices'][0]['code']], 100 + $item['position'], 20);
            }
            $attempts->submit($attempt['id'], $identity['user']['id']);

            $report = (new ReportViewModelFactory($pdo))->build($attempt['id'], $identity['user']['id']);

            self::assertSame('陈星河', $report['user']['display_name']);
            self::assertCount(6, $report['dimensions']);
            self::assertSame('GEO 基础理解', $report['dimensions'][0]['label']);
            self::assertSame('平台与来源差异', $report['dimensions'][3]['label']);
            self::assertCount(3, $report['difficulties']);
            self::assertCount(30, $report['questions']);
            self::assertSame('Q01', $report['questions'][0]['code']);
            self::assertSame('Q30', $report['questions'][29]['code']);
            self::assertSame(array_map(static fn (int $number): string => sprintf('Q%02d', $number), range(1, 30)), array_column($report['matrix'], 'code'));
            self::assertCount(30, $report['charts']['question_time']['labels']);
            self::assertCount(1, $report['charts']['score_trend']['scores']);
            self::assertFalse($report['charts']['trend_ready']);
            self::assertSame(1, $report['charts']['trend_count']);
            self::assertCount(6, $report['charts']['dimension_trend']['series']);
            self::assertCount(30, $report['matrix']);
            self::assertSame([20, 10], array_column($report['types'], 'total'));
            self::assertArrayHasKey('time_strategy', $report['insights']);
            self::assertArrayHasKey('change_count', $report['questions'][0]);
            self::assertCount(3, $report['insights']['learning_path']);
            self::assertFalse($report['cohort']['visible']);
            self::assertSame(8, count($report['view_catalog']));
        } finally {
            unset($pdo);
            @unlink($path);
            @unlink($path . '-shm');
            @unlink($path . '-wal');
        }
    }

    public function test_report_keeps_the_dimension_labels_saved_with_an_older_question_set(): void
    {
        $path = sys_get_temp_dir() . '/geo-assessment-' . bin2hex(random_bytes(8)) . '.sqlite';
        try {
            $pdo = Database::connect($path);
            (new MigrationRunner($pdo, dirname(__DIR__, 2) . '/database/migrations'))->migrate();
            (new QuestionImporter($pdo))->import(dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.1.json');
            $identity = (new IdentityService($pdo))->create('旧版测试者');
            $attempts = new AttemptService($pdo);
            $attempt = $attempts->start($identity['user']['id']);
            $attempts->submit($attempt['id'], $identity['user']['id']);

            $report = (new ReportViewModelFactory($pdo))->build($attempt['id'], $identity['user']['id']);

            self::assertSame('底层机制与范式', $report['dimensions'][0]['label']);
            self::assertSame('海外引用特征', $report['dimensions'][3]['label']);
        } finally {
            unset($pdo);
            @unlink($path);
            @unlink($path . '-shm');
            @unlink($path . '-wal');
        }
    }
}
