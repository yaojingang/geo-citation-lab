<?php

declare(strict_types=1);

$root = dirname(__DIR__);
$resolve = static function (string $value) use ($root): string {
    if ($value === '' || $value[0] === '/') {
        return $value;
    }
    return $root . '/' . $value;
};

return [
    'root' => $root,
    'environment' => getenv('APP_ENV') ?: 'production',
    'debug' => filter_var(getenv('APP_DEBUG') ?: '0', FILTER_VALIDATE_BOOLEAN),
    'base_path' => rtrim(getenv('APP_BASE_PATH') ?: '', '/'),
    'timezone' => getenv('APP_TIMEZONE') ?: 'Asia/Shanghai',
    'db_path' => $resolve(getenv('GEO_DB_PATH') ?: 'storage/app.sqlite'),
    'log_dir' => $resolve(getenv('GEO_LOG_DIR') ?: 'storage/logs'),
    'backup_dir' => $resolve(getenv('GEO_BACKUP_DIR') ?: 'storage/backups'),
    'cookie_secure' => getenv('GEO_COOKIE_SECURE') ?: 'auto',
    'trust_proxy' => filter_var(getenv('GEO_TRUST_PROXY') ?: '0', FILTER_VALIDATE_BOOLEAN),
    'app_key' => getenv('GEO_APP_KEY') ?: null,
];
