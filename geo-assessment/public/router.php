<?php

declare(strict_types=1);

$path = parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH);
$file = __DIR__ . ($path ?: '/');
$publicRoot = realpath(__DIR__);
$resolved = realpath($file);
if ($path !== '/' && is_string($publicRoot) && is_string($resolved) && strncmp($resolved, $publicRoot . DIRECTORY_SEPARATOR, strlen($publicRoot . DIRECTORY_SEPARATOR)) === 0 && is_file($resolved)) {
    return false;
}

require __DIR__ . '/index.php';
