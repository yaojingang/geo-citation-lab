<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Support\Config;
use PHPUnit\Framework\TestCase;

final class ConfigurationTest extends TestCase
{
    public function test_data_directory_controls_all_default_persistent_paths(): void
    {
        $variables = ['GEO_DATA_DIR', 'GEO_DB_PATH', 'GEO_LOG_DIR', 'GEO_BACKUP_DIR'];
        $original = [];
        foreach ($variables as $variable) {
            $original[$variable] = getenv($variable);
            putenv($variable);
        }
        $dataDirectory = sys_get_temp_dir() . '/geo-shared-' . bin2hex(random_bytes(5));

        try {
            putenv('GEO_DATA_DIR=' . $dataDirectory);
            $config = Config::load(dirname(__DIR__, 2) . '/config/app.php');

            self::assertSame($dataDirectory, $config->get('data_dir'));
            self::assertSame($dataDirectory . '/app.sqlite', $config->get('db_path'));
            self::assertSame($dataDirectory . '/app.key', $config->get('app_key_path'));
            self::assertSame($dataDirectory . '/logs', $config->get('log_dir'));
            self::assertSame($dataDirectory . '/backups', $config->get('backup_dir'));
        } finally {
            foreach ($original as $variable => $value) {
                $value === false ? putenv($variable) : putenv($variable . '=' . $value);
            }
        }
    }
}
