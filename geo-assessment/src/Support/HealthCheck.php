<?php

declare(strict_types=1);

namespace GeoAssessment\Support;

use PDO;

final class HealthCheck
{
    /** @var Config */
    private $config;

    public function __construct(Config $config)
    {
        $this->config = $config;
    }

    /** @return array{ok: bool, checks: list<array{name: string, status: string, message: string}>} */
    public function run(): array
    {
        $checks = [];
        $add = static function (string $name, bool $ok, string $message) use (&$checks): void {
            $checks[] = ['name' => $name, 'status' => $ok ? 'ok' : 'fail', 'message' => $message];
        };

        $add('PHP', version_compare(PHP_VERSION, '7.3.5', '>='), PHP_VERSION);
        $required = ['json', 'mbstring', 'openssl', 'pdo', 'pdo_sqlite'];
        $missing = array_values(array_filter($required, static function (string $extension): bool {
            return !extension_loaded($extension);
        }));
        $add('扩展', $missing === [], $missing === [] ? '所有必需扩展已加载' : '缺少 ' . implode(', ', $missing));

        $root = (string) $this->config->get('root');
        $appKey = (string) ($this->config->get('app_key') ?: (is_file($root . '/storage/app.key') ? trim((string) file_get_contents($root . '/storage/app.key')) : ''));
        $add('应用密钥', strlen($appKey) >= 32, strlen($appKey) >= 32 ? '已配置' : '未配置');

        foreach (['log_dir' => '日志目录', 'backup_dir' => '备份目录'] as $key => $name) {
            $directory = (string) $this->config->get($key);
            $add($name, is_dir($directory) && is_writable($directory), $directory);
        }

        $chartPath = $root . '/public/assets/vendor/chart.umd.min.js';
        $chartVersion = is_file($chartPath) ? (string) file_get_contents($chartPath, false, null, 0, 80) : '';
        $hasChartVersion = strpos($chartVersion, 'Chart.js v4.5.1') !== false;
        $add('Chart.js', $hasChartVersion, $hasChartVersion ? '4.5.1 本地文件' : '本地文件缺失或版本不匹配');

        $analyticsId = trim((string) $this->config->get('baidu_analytics_id', ''));
        $analyticsValid = $analyticsId === '' || BaiduAnalytics::enabled($analyticsId);
        $analyticsMessage = $analyticsId === '' ? '未启用' : ($analyticsValid ? '已启用' : 'ID 应为 32 位十六进制字符串');
        $add('百度统计', $analyticsValid, $analyticsMessage);

        $databasePath = (string) $this->config->get('db_path');
        if (!is_file($databasePath)) {
            $add('数据库', false, '文件不存在');
            return ['ok' => false, 'checks' => $checks];
        }
        $worldBits = fileperms($databasePath) & 0007;
        $add('数据库权限', $worldBits === 0, $worldBits === 0 ? '未向其他用户开放' : '建议移除 other 权限位');

        try {
            $pdo = Database::connect($databasePath);
            $sqliteVersion = (string) $pdo->query('SELECT sqlite_version()')->fetchColumn();
            $add('SQLite', version_compare($sqliteVersion, '3.24.0', '>='), $sqliteVersion);
            $integrity = (string) $pdo->query('PRAGMA quick_check')->fetchColumn();
            $add('数据库完整性', $integrity === 'ok', $integrity);
            $migrations = $this->tableCount($pdo, 'schema_migrations');
            $add('数据库迁移', $migrations >= 1, $migrations . ' 个版本');
            $activeSets = (int) $pdo->query('SELECT COUNT(*) FROM question_sets WHERE active = 1')->fetchColumn();
            $questions = (int) $pdo->query('SELECT COUNT(*) FROM questions q JOIN question_sets qs ON qs.id = q.set_id WHERE qs.active = 1')->fetchColumn();
            $add('活动题集', $activeSets === 1 && $questions === 30, $activeSets . ' 个题集，' . $questions . ' 道题');
        } catch (\Throwable $error) {
            $add('数据库', false, $error->getMessage());
        }

        return ['ok' => !in_array('fail', array_column($checks, 'status'), true), 'checks' => $checks];
    }

    private function tableCount(PDO $pdo, string $table): int
    {
        $exists = $pdo->prepare("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = :name");
        $exists->execute(['name' => $table]);
        return (int) $exists->fetchColumn() === 1 ? (int) $pdo->query('SELECT COUNT(*) FROM ' . $table)->fetchColumn() : 0;
    }
}
