<?php

declare(strict_types=1);

$root = dirname(__DIR__);
$resolve = static function (string $value) use ($root): string {
    if ($value === '' || $value[0] === '/') {
        return $value;
    }
    return $root . '/' . $value;
};
$dataDir = rtrim($resolve(getenv('GEO_DATA_DIR') ?: 'storage'), '/');
if ($dataDir === '') {
    throw new InvalidArgumentException('GEO_DATA_DIR 不能指向文件系统根目录。');
}
$configuredPath = static function (string $name, string $default) use ($resolve): string {
    $value = getenv($name);
    return $resolve(is_string($value) && $value !== '' ? $value : $default);
};

return [
    'root' => $root,
    'data_dir' => $dataDir,
    'environment' => getenv('APP_ENV') ?: 'production',
    'debug' => filter_var(getenv('APP_DEBUG') ?: '0', FILTER_VALIDATE_BOOLEAN),
    'base_path' => rtrim(getenv('APP_BASE_PATH') ?: '', '/'),
    'public_url' => rtrim(getenv('GEO_PUBLIC_URL') ?: '', '/'),
    'timezone' => getenv('APP_TIMEZONE') ?: 'Asia/Shanghai',
    'db_path' => $configuredPath('GEO_DB_PATH', $dataDir . '/app.sqlite'),
    'log_dir' => $configuredPath('GEO_LOG_DIR', $dataDir . '/logs'),
    'backup_dir' => $configuredPath('GEO_BACKUP_DIR', $dataDir . '/backups'),
    'app_key_path' => $dataDir . '/app.key',
    'cookie_secure' => getenv('GEO_COOKIE_SECURE') ?: 'auto',
    'trust_proxy' => filter_var(getenv('GEO_TRUST_PROXY') ?: '0', FILTER_VALIDATE_BOOLEAN),
    'app_key' => getenv('GEO_APP_KEY') ?: null,
    'baidu_analytics_id' => getenv('GEO_BAIDU_ANALYTICS_ID') ?: '',
];
