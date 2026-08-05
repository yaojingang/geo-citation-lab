<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Integration;

use GeoAssessment\Assessment\QuestionImporter;
use GeoAssessment\Support\Config;
use GeoAssessment\Support\Database;
use GeoAssessment\Support\HealthCheck;
use GeoAssessment\Support\MigrationRunner;
use PHPUnit\Framework\TestCase;

final class HealthCheckTest extends TestCase
{
    public function testInstalledApplicationPassesEveryRequiredCheck(): void
    {
        $directory = sys_get_temp_dir() . '/geo-health-' . bin2hex(random_bytes(5));
        mkdir($directory . '/storage/logs', 0770, true);
        mkdir($directory . '/storage/backups', 0770, true);
        mkdir($directory . '/public/assets/vendor', 0770, true);
        file_put_contents($directory . '/public/assets/vendor/chart.umd.min.js', '/*! Chart.js v4.5.1 */');
        $databasePath = $directory . '/storage/app.sqlite';
        file_put_contents($directory . '/storage/app.key', str_repeat('a', 64));
        $pdo = Database::connect($databasePath);
        (new MigrationRunner($pdo, dirname(__DIR__, 2) . '/database/migrations'))->migrate();
        (new QuestionImporter($pdo))->import(dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.1.json');
        $config = new Config([
            'root' => $directory,
            'db_path' => $databasePath,
            'log_dir' => $directory . '/storage/logs',
            'backup_dir' => $directory . '/storage/backups',
            'app_key' => null,
        ]);

        $result = (new HealthCheck($config))->run();

        self::assertTrue($result['ok']);
        self::assertNotEmpty($result['checks']);
        self::assertNotContains('fail', array_column($result['checks'], 'status'));
    }
}
