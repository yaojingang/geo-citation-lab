<?php

declare(strict_types=1);

namespace GeoAssessment\Console;

use GeoAssessment\Assessment\QuestionImporter;
use GeoAssessment\Assessment\QuestionSetValidator;
use GeoAssessment\Support\BackupService;
use GeoAssessment\Support\Config;
use GeoAssessment\Support\Database;
use GeoAssessment\Support\HealthCheck;
use GeoAssessment\Support\MigrationRunner;
use RuntimeException;

final class ConsoleApplication
{
    private const DEFAULT_QUESTION_SET = '/database/seeds/geo-30-v1.2.json';

    /** @var Config */
    private $config;

    public function __construct(Config $config)
    {
        $this->config = $config;
    }

    /** @param list<string> $arguments */
    public function run(array $arguments): int
    {
        $command = $arguments[1] ?? 'help';
        try {
            switch ($command) {
                case 'app:install':
                    return $this->install();
                case 'db:migrate':
                    return $this->migrate();
                case 'questions:import':
                    return $this->import($this->firstPositional($arguments), !in_array('--no-activate', $arguments, true));
                case 'questions:verify':
                    return $this->verifyQuestions($this->firstPositional($arguments));
                case 'backup:create':
                    return $this->backupCreate();
                case 'backup:verify':
                    return $this->backupVerify($this->firstPositional($arguments));
                case 'logs:prune':
                    return $this->pruneLogs($this->firstPositional($arguments) ?? '30');
                case 'app:health':
                    return $this->health();
                case 'version':
                case '--version':
                case '-V':
                    return $this->version();
                case 'help':
                case '--help':
                case '-h':
                    return $this->help();
                default:
                    throw new RuntimeException("未知命令：{$command}");
            }
        } catch (\Throwable $error) {
            fwrite(STDERR, "[FAIL] {$error->getMessage()}" . PHP_EOL);
            return 1;
        }
    }

    private function install(): int
    {
        $root = (string) $this->config->get('root');
        foreach ([(string) $this->config->get('log_dir'), (string) $this->config->get('backup_dir'), dirname((string) $this->config->get('db_path'))] as $directory) {
            if (!is_dir($directory) && !mkdir($directory, 0770, true) && !is_dir($directory)) {
                throw new RuntimeException("无法创建目录：{$directory}");
            }
        }
        $keyPath = $root . '/storage/app.key';
        if (!is_file($keyPath)) {
            $key = rtrim(strtr(base64_encode(random_bytes(48)), '+/', '-_'), '=');
            file_put_contents($keyPath, $key . PHP_EOL, LOCK_EX);
            chmod($keyPath, 0600);
            $this->line('[OK] 已生成应用密钥');
        } else {
            $this->line('[OK] 已保留现有应用密钥');
        }
        $this->migrate();
        chmod((string) $this->config->get('db_path'), 0660);
        $this->import(null, true);
        return $this->health();
    }

    private function migrate(): int
    {
        $pdo = Database::connect((string) $this->config->get('db_path'));
        if ($this->hasPendingMigrations($pdo)) {
            $hasQuestionSets = (int) $pdo->query("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'question_sets'")->fetchColumn() === 1;
            if ($hasQuestionSets) {
                $backup = $this->backups()->create($pdo);
                $this->line('[OK] 迁移前备份 ' . $backup);
            }
        }
        $count = (new MigrationRunner($pdo, (string) $this->config->get('root') . '/database/migrations'))->migrate();
        $this->line('[OK] 数据库迁移完成，新应用 ' . $count . ' 个版本');
        return 0;
    }

    private function import(?string $path, bool $activate): int
    {
        if ($path === null) {
            $path = (string) $this->config->get('root') . self::DEFAULT_QUESTION_SET;
        }
        $pdo = Database::connect((string) $this->config->get('db_path'));
        $result = (new QuestionImporter($pdo))->import($path, $activate);
        $this->line('[OK] 题库 ' . ($result === 'imported' ? '已导入' : '内容未变更'));
        return 0;
    }

    private function verifyQuestions(?string $path): int
    {
        if ($path === null) {
            $path = (string) $this->config->get('root') . self::DEFAULT_QUESTION_SET;
        }
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
        $summary = (new QuestionSetValidator())->validate($payload);
        $this->line(sprintf('[OK] %d 题 · %d 分 · %d 单选 · %d 多选 · %d 篇论文 · %d 秒', $summary['question_count'], $summary['total_points'], $summary['single_count'], $summary['multiple_count'], $summary['covered_papers'], $summary['expected_seconds']));
        $this->line('[OK] 内容指纹 ' . $summary['content_hash']);
        return 0;
    }

    private function backupCreate(): int
    {
        $path = $this->backups()->create();
        $this->line('[OK] 备份已创建 ' . $path);
        return 0;
    }

    private function backupVerify(?string $path): int
    {
        if ($path === null) {
            $files = glob(rtrim((string) $this->config->get('backup_dir'), '/') . '/*.sqlite') ?: [];
            rsort($files, SORT_STRING);
            $path = $files[0] ?? null;
        }
        if ($path === null) {
            throw new RuntimeException('没有可验证的备份。');
        }
        $result = $this->backups()->verify($path);
        $this->line('[OK] 备份完整，' . $result['tables'] . ' 张数据表，SHA-256 ' . $result['checksum']);
        return 0;
    }

    private function pruneLogs(string $daysText): int
    {
        $days = filter_var($daysText, FILTER_VALIDATE_INT, ['options' => ['min_range' => 1, 'max_range' => 3650]]);
        if (!is_int($days)) {
            throw new RuntimeException('保留天数必须位于 1 至 3650。');
        }
        $removed = 0;
        $threshold = time() - ($days * 86400);
        foreach (glob(rtrim((string) $this->config->get('log_dir'), '/') . '/*.jsonl') ?: [] as $file) {
            if (is_file($file) && filemtime($file) < $threshold && unlink($file)) {
                $removed++;
            }
        }
        $this->line('[OK] 已清理 ' . $removed . ' 个过期日志文件');
        return 0;
    }

    private function health(): int
    {
        $result = (new HealthCheck($this->config))->run();
        foreach ($result['checks'] as $check) {
            $this->line(sprintf('[%s] %-16s %s', strtoupper($check['status']), $check['name'], $check['message']));
        }
        return $result['ok'] ? 0 : 1;
    }

    private function help(): int
    {
        $this->line(<<<'TEXT'
GEO Assessment 命令

  app:install                   安装应用、迁移数据库并导入题库
  db:migrate                    应用尚未执行的 SQLite 迁移
  questions:import [file]       验证并导入不可变题库版本
  questions:verify [file]       验证 30 题蓝图、分值与证据覆盖
  backup:create                 创建 SQLite 一致性备份和校验和
  backup:verify [file]          验证备份校验和与数据库完整性
  logs:prune [days]             删除超过保留期的 JSONL 日志，默认 30 天
  app:health                    检查运行环境、密钥、SQLite 和活动题集
  version                       显示应用版本
TEXT);
        return 0;
    }

    private function backups(): BackupService
    {
        return new BackupService((string) $this->config->get('db_path'), (string) $this->config->get('backup_dir'));
    }

    private function version(): int
    {
        $versionFile = (string) $this->config->get('root') . '/VERSION';
        $version = is_file($versionFile) ? trim((string) file_get_contents($versionFile)) : 'development';
        $this->line('GEO Assessment ' . $version);
        return 0;
    }

    /** @param list<string> $arguments */
    private function firstPositional(array $arguments): ?string
    {
        foreach (array_slice($arguments, 2) as $argument) {
            if (strncmp($argument, '-', 1) !== 0) {
                return $argument;
            }
        }
        return null;
    }

    private function hasPendingMigrations(\PDO $pdo): bool
    {
        $tableExists = (int) $pdo->query("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'")->fetchColumn() === 1;
        $applied = $tableExists ? array_flip($pdo->query('SELECT version FROM schema_migrations')->fetchAll(\PDO::FETCH_COLUMN)) : [];
        $files = glob((string) $this->config->get('root') . '/database/migrations/*.sql') ?: [];
        foreach ($files as $file) {
            if (!isset($applied[pathinfo($file, PATHINFO_FILENAME)])) {
                return true;
            }
        }
        return false;
    }

    private function line(string $message): void
    {
        fwrite(STDOUT, $message . PHP_EOL);
    }
}
