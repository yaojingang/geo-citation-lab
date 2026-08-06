<?php

declare(strict_types=1);

use GeoAssessment\Assessment\AttemptService;
use GeoAssessment\Assessment\QuestionImporter;
use GeoAssessment\Identity\IdentityService;
use GeoAssessment\Support\Database;
use GeoAssessment\Support\MigrationRunner;

require dirname(__DIR__, 2) . '/vendor/autoload.php';

$options = getopt('', ['users::', 'write-rps::', 'duration::']);
$users = max(1, min(500, (int) ($options['users'] ?? 20)));
$writeRps = max(1, min(200, (int) ($options['write-rps'] ?? 10)));
$duration = max(1, min(600, (int) ($options['duration'] ?? 10)));
$root = dirname(__DIR__, 2);
$databasePath = sys_get_temp_dir() . '/geo-load-' . bin2hex(random_bytes(6)) . '.sqlite';
$exitCode = 1;

try {
    $pdo = Database::connect($databasePath);
    (new MigrationRunner($pdo, $root . '/database/migrations'))->migrate();
    (new QuestionImporter($pdo))->import($root . '/database/seeds/geo-30-v1.2.json');
    $identities = new IdentityService($pdo);
    $attempts = new AttemptService($pdo);
    $actors = [];
    for ($index = 0; $index < $users; $index++) {
        $identity = $identities->create('压测用户' . str_pad((string) $index, 3, '0', STR_PAD_LEFT));
        $attempt = $attempts->start((string) $identity['user']['id']);
        $items = $attempts->items((string) $attempt['id'], (string) $identity['user']['id']);
        $actors[] = ['user_id' => $identity['user']['id'], 'attempt_id' => $attempt['id'], 'items' => $items, 'sequences' => array_fill(0, 30, 0)];
    }

    $latencies = [];
    $errors = 0;
    $writes = 0;
    $started = microtime(true);
    $interval = 1 / $writeRps;
    while (microtime(true) - $started < $duration) {
        $target = $started + ($writes / $writeRps);
        if (($wait = $target - microtime(true)) > 0) {
            usleep((int) ($wait * 1_000_000));
        }
        $actorIndex = $writes % count($actors);
        $itemIndex = ($writes * 7) % 30;
        $actor = &$actors[$actorIndex];
        $item = $actor['items'][$itemIndex];
        $choice = (string) $item['snapshot']['choices'][0]['code'];
        $actor['sequences'][$itemIndex]++;
        $writeStarted = microtime(true);
        try {
            $attempts->saveAnswer((string) $actor['attempt_id'], (string) $actor['user_id'], (string) $item['snapshot']['code'], [$choice], $actor['sequences'][$itemIndex], 1);
        } catch (Throwable) {
            $errors++;
        }
        $latencies[] = (microtime(true) - $writeStarted) * 1000;
        $writes++;
        unset($actor);
    }
    sort($latencies, SORT_NUMERIC);
    $p95Index = max(0, (int) ceil(count($latencies) * .95) - 1);
    $p95 = $latencies[$p95Index] ?? 0;
    printf("[OK] users=%d writes=%d errors=%d p95=%.2fms target_rps=%d duration=%ds\n", $users, $writes, $errors, $p95, $writeRps, $duration);
    $exitCode = $errors === 0 ? 0 : 1;
} finally {
    foreach ([$databasePath, $databasePath . '-wal', $databasePath . '-shm'] as $file) {
        if (is_file($file)) {
            unlink($file);
        }
    }
}
exit($exitCode);
