<?php

declare(strict_types=1);

namespace GeoAssessment\Support;

final class JsonLogger
{
    /** @var string */
    private $directory;

    public function __construct(string $directory)
    {
        $this->directory = $directory;
    }

    public function error(\Throwable $error, array $context = []): void
    {
        if (!is_dir($this->directory) && !@mkdir($this->directory, 0770, true) && !is_dir($this->directory)) {
            return;
        }
        $payload = [
            'timestamp' => gmdate('Y-m-d\TH:i:s\Z'),
            'level' => 'error',
            'type' => get_class($error),
            'code' => (string) $error->getCode(),
            'context' => $context,
        ];
        $path = rtrim($this->directory, '/') . '/app-' . gmdate('Y-m-d') . '.jsonl';
        @file_put_contents($path, json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) . PHP_EOL, FILE_APPEND | LOCK_EX);
        @chmod($path, 0600);
    }
}
