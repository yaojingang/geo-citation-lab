<?php

declare(strict_types=1);

namespace GeoAssessment\Assessment;

use DomainException;
use GeoAssessment\Scoring\ScoringService;
use GeoAssessment\Support\Clock;
use GeoAssessment\Support\SystemClock;
use GeoAssessment\Support\Uuid;
use PDO;

final class AttemptService
{
    /** @var PDO */
    private $pdo;

    /** @var ScoringService */
    private $scoring;

    /** @var Clock */
    private $clock;

    /** @var ChoicePresenter */
    private $choicePresenter;

    public function __construct(
        PDO $pdo,
        ?ScoringService $scoring = null,
        ?Clock $clock = null,
        ?ChoicePresenter $choicePresenter = null
    ) {
        $this->pdo = $pdo;
        $this->scoring = $scoring ?? new ScoringService();
        $this->clock = $clock ?? new SystemClock();
        $this->choicePresenter = $choicePresenter ?? new ChoicePresenter();
    }

    /** @return array<string, mixed> */
    public function start(string $userId): array
    {
        $this->pdo->exec('BEGIN IMMEDIATE');
        $transactionStarted = true;
        try {
            $current = $this->pdo->prepare("SELECT * FROM attempts WHERE user_id = :user_id AND status = 'in_progress' LIMIT 1");
            $current->execute(['user_id' => $userId]);
            $attempt = $current->fetch();
            if (is_array($attempt)) {
                if ((string) $attempt['deadline_at'] <= $this->nowText()) {
                    $this->finalizeInTransaction($attempt, 'timed_out');
                } else {
                    $this->pdo->exec('COMMIT');
                    $transactionStarted = false;
                    return $attempt;
                }
            }

            $countStatement = $this->pdo->prepare('SELECT COUNT(*) FROM attempts WHERE user_id = :user_id');
            $countStatement->execute(['user_id' => $userId]);
            $count = (int) $countStatement->fetchColumn();
            if ($count >= 10) {
                throw new DomainException('每个浏览器身份最多可测试 10 次。');
            }

            $set = $this->pdo->query('SELECT * FROM question_sets WHERE active = 1 LIMIT 1')->fetch();
            if (!is_array($set)) {
                throw new DomainException('当前没有可用题集。');
            }
            $questions = $this->loadQuestions((string) $set['id']);
            $this->secureShuffle($questions);

            $attemptId = Uuid::v4();
            $now = $this->clock->now();
            $nowText = $this->format($now);
            $deadline = $this->format($now->modify('+' . (int) $set['time_limit_seconds'] . ' seconds'));
            $attemptNo = $count + 1;
            $insertAttempt = $this->pdo->prepare("INSERT INTO attempts (id, user_id, set_id, attempt_no, status, started_at, deadline_at, scoring_version, created_at, updated_at) VALUES (:id, :user_id, :set_id, :attempt_no, 'in_progress', :started_at, :deadline_at, :scoring_version, :created_at, :updated_at)");
            $insertAttempt->execute([
                'id' => $attemptId,
                'user_id' => $userId,
                'set_id' => $set['id'],
                'attempt_no' => $attemptNo,
                'started_at' => $nowText,
                'deadline_at' => $deadline,
                'scoring_version' => $set['scoring_version'],
                'created_at' => $nowText,
                'updated_at' => $nowText,
            ]);

            $insertItem = $this->pdo->prepare("INSERT INTO attempt_items (attempt_id, question_id, position, question_snapshot_json, choice_order_json, response_json) VALUES (:attempt_id, :question_id, :position, :snapshot, :choice_order, '[]')");
            foreach ($questions as $index => $question) {
                $choices = $question['choices'];
                $this->secureShuffle($choices);
                $choices = $this->choicePresenter->present($choices);
                $snapshot = [
                    'schema_version' => 'attempt-item-v1',
                    'code' => $question['code'],
                    'title' => $question['title'],
                    'type' => $question['type'],
                    'dimension' => $question['dimension'],
                    'dimension_label' => $question['dimension_label'],
                    'difficulty' => $question['difficulty'],
                    'difficulty_label' => $question['difficulty_label'],
                    'cognitive_level' => $question['cognitive_level'],
                    'prompt' => $question['prompt'],
                    'choices' => $choices,
                    'correct_codes' => array_values(array_map(static function (array $choice): string {
                        return $choice['code'];
                    }, array_filter($choices, static function (array $choice): bool {
                        return $choice['is_correct'];
                    }))),
                    'weight' => (int) $question['weight'],
                    'explanation' => $question['explanation'],
                    'principle' => $question['principle'],
                    'misconception_tag' => $question['misconception_tag'],
                    'source_refs' => $question['source_refs'],
                    'sources' => $question['sources'],
                    'expected_seconds' => (int) $question['expected_seconds'],
                ];
                $insertItem->execute([
                    'attempt_id' => $attemptId,
                    'question_id' => $question['id'],
                    'position' => $index + 1,
                    'snapshot' => $this->json($snapshot),
                    'choice_order' => $this->json(array_column($choices, 'source_code')),
                ]);
            }
            $this->pdo->exec('COMMIT');
            $transactionStarted = false;

            return [
                'id' => $attemptId,
                'user_id' => $userId,
                'set_id' => $set['id'],
                'attempt_no' => $attemptNo,
                'status' => 'in_progress',
                'started_at' => $nowText,
                'deadline_at' => $deadline,
                'scoring_version' => $set['scoring_version'],
            ];
        } catch (\Throwable $error) {
            if ($transactionStarted) {
                $this->pdo->exec('ROLLBACK');
            }
            throw $error;
        }
    }

    /** @return list<array<string, mixed>> */
    public function items(string $attemptId, string $userId): array
    {
        $this->assertOwnership($attemptId, $userId);
        $statement = $this->pdo->prepare('SELECT ai.* FROM attempt_items ai WHERE ai.attempt_id = :attempt_id ORDER BY ai.position');
        $statement->execute(['attempt_id' => $attemptId]);
        return array_map(function (array $row): array {
            return $this->decodeItem($row);
        }, $statement->fetchAll());
    }

    /** @return array<string, mixed> */
    public function itemAt(string $attemptId, string $userId, int $position): array
    {
        $attempt = $this->assertOwnership($attemptId, $userId);
        if ($attempt['status'] === 'in_progress' && (string) $attempt['deadline_at'] <= $this->nowText()) {
            $this->submit($attemptId, $userId);
            throw new DomainException('测试时间已到。');
        }
        $statement = $this->pdo->prepare('SELECT * FROM attempt_items WHERE attempt_id = :attempt_id AND position = :position');
        $statement->execute(['attempt_id' => $attemptId, 'position' => $position]);
        $row = $statement->fetch();
        if (!is_array($row)) {
            throw new DomainException('题目不存在。');
        }
        if ($row['first_viewed_at'] === null && $attempt['status'] === 'in_progress') {
            $now = $this->nowText();
            $update = $this->pdo->prepare('UPDATE attempt_items SET first_viewed_at = :now, last_viewed_at = :now WHERE attempt_id = :attempt_id AND position = :position');
            $update->execute(['now' => $now, 'attempt_id' => $attemptId, 'position' => $position]);
            $row['first_viewed_at'] = $now;
            $row['last_viewed_at'] = $now;
        }
        return $this->decodeItem($row);
    }

    /** @return array{stale: bool, selected_codes: list<string>, activity_seq: int, saved_at: string, attempt_status: string, deadline_at: string} */
    public function saveAnswer(string $attemptId, string $userId, string $questionCode, array $selectedCodes, int $activitySeq, int $activeSecondsDelta): array
    {
        $this->pdo->exec('BEGIN IMMEDIATE');
        $transactionStarted = true;
        try {
            $attempt = $this->assertOwnership($attemptId, $userId);
            if ($attempt['status'] !== 'in_progress') {
                throw new DomainException('该测试已结束。');
            }
            if ((string) $attempt['deadline_at'] <= $this->nowText()) {
                $this->finalizeInTransaction($attempt, 'timed_out');
                $this->pdo->exec('COMMIT');
                $transactionStarted = false;
                throw new DomainException('测试时间已到。');
            }

            $statement = $this->pdo->prepare('SELECT ai.* FROM attempt_items ai JOIN questions q ON q.id = ai.question_id WHERE ai.attempt_id = :attempt_id AND q.code = :code');
            $statement->execute(['attempt_id' => $attemptId, 'code' => $questionCode]);
            $row = $statement->fetch();
            if (!is_array($row)) {
                throw new DomainException('题目不存在。');
            }
            $item = $this->decodeItem($row);
            $selectedCodes = array_values(array_unique(array_map('strval', $selectedCodes)));
            sort($selectedCodes, SORT_STRING);
            $legalCodes = array_column($item['snapshot']['choices'], 'code');
            if (array_diff($selectedCodes, $legalCodes) !== []) {
                throw new DomainException('答案包含无效选项。');
            }
            if ($item['snapshot']['type'] === 'single' && count($selectedCodes) > 1) {
                throw new DomainException('单选题只能选择一项。');
            }
            if ($activitySeq <= (int) $row['activity_seq']) {
                $this->pdo->exec('COMMIT');
                $transactionStarted = false;
                return [
                    'stale' => true,
                    'selected_codes' => $item['selected_codes'],
                    'activity_seq' => (int) $row['activity_seq'],
                    'saved_at' => (string) ($row['last_viewed_at'] ?? $this->nowText()),
                    'attempt_status' => 'in_progress',
                    'deadline_at' => (string) $attempt['deadline_at'],
                ];
            }

            $now = $this->nowText();
            $delta = max(0, min(30, $activeSecondsDelta));
            $changed = $selectedCodes === $item['selected_codes'] ? 0 : 1;
            $update = $this->pdo->prepare("UPDATE attempt_items SET response_json = :response, activity_seq = :activity_seq, time_spent_seconds = MIN(1800, time_spent_seconds + :delta), change_count = change_count + :changed, last_viewed_at = :now WHERE attempt_id = :attempt_id AND question_id = :question_id AND activity_seq < :activity_seq AND EXISTS (SELECT 1 FROM attempts a WHERE a.id = attempt_items.attempt_id AND a.user_id = :user_id AND a.status = 'in_progress')");
            $update->execute([
                'response' => $this->json($selectedCodes),
                'activity_seq' => $activitySeq,
                'delta' => $delta,
                'changed' => $changed,
                'now' => $now,
                'attempt_id' => $attemptId,
                'question_id' => $row['question_id'],
                'user_id' => $userId,
            ]);
            if ($update->rowCount() !== 1) {
                $current = $this->assertOwnership($attemptId, $userId);
                if ($current['status'] !== 'in_progress') {
                    throw new DomainException('该测试已结束。');
                }
                throw new DomainException('答案未能保存，请重试。');
            }
            $touch = $this->pdo->prepare("UPDATE attempts SET updated_at = :now WHERE id = :id AND user_id = :user_id AND status = 'in_progress'");
            $touch->execute(['now' => $now, 'id' => $attemptId, 'user_id' => $userId]);
            if ($touch->rowCount() !== 1) {
                throw new DomainException('该测试已结束。');
            }
            $this->pdo->exec('COMMIT');
            $transactionStarted = false;

            return [
                'stale' => false,
                'selected_codes' => $selectedCodes,
                'activity_seq' => $activitySeq,
                'saved_at' => $now,
                'attempt_status' => 'in_progress',
                'deadline_at' => (string) $attempt['deadline_at'],
            ];
        } catch (\Throwable $error) {
            if ($transactionStarted) {
                $this->pdo->exec('ROLLBACK');
            }
            throw $error;
        }
    }

    /** @return array<string, mixed> */
    public function submit(string $attemptId, string $userId): array
    {
        $this->pdo->exec('BEGIN IMMEDIATE');
        $transactionStarted = true;
        try {
            $attempt = $this->assertOwnership($attemptId, $userId);
            if ($attempt['status'] !== 'in_progress') {
                $this->pdo->exec('COMMIT');
                $transactionStarted = false;
                return $this->resultShape($attempt);
            }
            $status = (string) $attempt['deadline_at'] <= $this->nowText() ? 'timed_out' : 'submitted';
            $result = $this->finalizeInTransaction($attempt, $status);
            $this->pdo->exec('COMMIT');
            $transactionStarted = false;
            return $result;
        } catch (\Throwable $error) {
            if ($transactionStarted) {
                $this->pdo->exec('ROLLBACK');
            }
            throw $error;
        }
    }

    /** @return array<string, mixed> */
    public function getAttempt(string $attemptId, string $userId): array
    {
        return $this->assertOwnership($attemptId, $userId);
    }

    /** @return list<array<string, mixed>> */
    public function history(string $userId): array
    {
        $statement = $this->pdo->prepare('SELECT a.*, qs.version AS set_version FROM attempts a JOIN question_sets qs ON qs.id = a.set_id WHERE a.user_id = :user_id ORDER BY a.attempt_no DESC LIMIT 10');
        $statement->execute(['user_id' => $userId]);
        return $statement->fetchAll();
    }

    public function expireOverdueForUser(string $userId): void
    {
        $statement = $this->pdo->prepare("SELECT id FROM attempts WHERE user_id = :user_id AND status = 'in_progress' AND deadline_at <= :now");
        $statement->execute(['user_id' => $userId, 'now' => $this->nowText()]);
        foreach ($statement->fetchAll() as $attempt) {
            $this->submit((string) $attempt['id'], $userId);
        }
    }

    /** @return array<string, mixed>|null */
    public function current(string $userId): ?array
    {
        $this->expireOverdueForUser($userId);
        $statement = $this->pdo->prepare("SELECT * FROM attempts WHERE user_id = :user_id AND status = 'in_progress' LIMIT 1");
        $statement->execute(['user_id' => $userId]);
        $attempt = $statement->fetch();
        return is_array($attempt) ? $attempt : null;
    }

    /** @return array<string, mixed> */
    private function finalizeInTransaction(array $attempt, string $status): array
    {
        $statement = $this->pdo->prepare('SELECT * FROM attempt_items WHERE attempt_id = :attempt_id ORDER BY position');
        $statement->execute(['attempt_id' => $attempt['id']]);
        $items = array_map(function (array $row): array {
            return $this->decodeItem($row);
        }, $statement->fetchAll());
        $scoreInput = array_map(static function (array $item): array {
            return ['snapshot' => $item['snapshot'], 'selected_codes' => $item['selected_codes']];
        }, $items);
        $scored = $this->scoring->score($scoreInput);
        $pointsByCode = array_column($scored['items'], 'points', 'code');
        $updateItem = $this->pdo->prepare('UPDATE attempt_items SET points = :points WHERE attempt_id = :attempt_id AND question_id = :question_id');
        foreach ($items as $item) {
            $updateItem->execute(['points' => $pointsByCode[$item['snapshot']['code']], 'attempt_id' => $attempt['id'], 'question_id' => $item['question_id']]);
        }
        $submittedAt = $this->nowText();
        $duration = min(1800, max(0, $this->clock->now()->getTimestamp() - (new \DateTimeImmutable((string) $attempt['started_at']))->getTimestamp()));
        $update = $this->pdo->prepare('UPDATE attempts SET status = :status, submitted_at = :submitted_at, duration_seconds = :duration, score = :score, correct_count = :correct_count, updated_at = :updated_at WHERE id = :id');
        $update->execute([
            'status' => $status,
            'submitted_at' => $submittedAt,
            'duration' => $duration,
            'score' => $scored['score'],
            'correct_count' => $scored['correct_count'],
            'updated_at' => $submittedAt,
            'id' => $attempt['id'],
        ]);
        return $this->resultShape(array_merge($attempt, [
            'status' => $status,
            'submitted_at' => $submittedAt,
            'duration_seconds' => $duration,
            'score' => $scored['score'],
            'correct_count' => $scored['correct_count'],
        ]));
    }

    /** @return array{id: string, status: string, score: int, correct_count: int, duration_seconds: int, submitted_at: string} */
    private function resultShape(array $attempt): array
    {
        return [
            'id' => (string) $attempt['id'],
            'status' => (string) $attempt['status'],
            'score' => (int) $attempt['score'],
            'correct_count' => (int) $attempt['correct_count'],
            'duration_seconds' => (int) $attempt['duration_seconds'],
            'submitted_at' => (string) $attempt['submitted_at'],
        ];
    }

    /** @return array<string, mixed> */
    private function assertOwnership(string $attemptId, string $userId): array
    {
        $statement = $this->pdo->prepare('SELECT * FROM attempts WHERE id = :id AND user_id = :user_id');
        $statement->execute(['id' => $attemptId, 'user_id' => $userId]);
        $attempt = $statement->fetch();
        if (!is_array($attempt)) {
            throw new DomainException('测试不存在。');
        }
        return $attempt;
    }

    /** @return list<array<string, mixed>> */
    private function loadQuestions(string $setId): array
    {
        $statement = $this->pdo->prepare('SELECT * FROM questions WHERE set_id = :set_id ORDER BY sort_order');
        $statement->execute(['set_id' => $setId]);
        $questions = [];
        foreach ($statement->fetchAll() as $row) {
            $choiceStatement = $this->pdo->prepare('SELECT code, text, is_correct, rationale FROM choices WHERE question_id = :question_id ORDER BY code');
            $choiceStatement->execute(['question_id' => $row['id']]);
            $choices = array_map(static function (array $choice): array {
                return array_merge($choice, ['is_correct' => (bool) $choice['is_correct']]);
            }, $choiceStatement->fetchAll());
            $questions[] = array_merge($row, [
                'choices' => $choices,
                'source_refs' => json_decode((string) $row['source_refs_json'], true, 512, JSON_THROW_ON_ERROR),
                'sources' => json_decode((string) $row['source_objects_json'], true, 512, JSON_THROW_ON_ERROR),
            ]);
        }
        return $questions;
    }

    /** @return array<string, mixed> */
    private function decodeItem(array $row): array
    {
        return array_merge($row, [
            'position' => (int) $row['position'],
            'activity_seq' => (int) $row['activity_seq'],
            'time_spent_seconds' => (int) $row['time_spent_seconds'],
            'change_count' => (int) $row['change_count'],
            'points' => $row['points'] === null ? null : (int) $row['points'],
            'snapshot' => json_decode((string) $row['question_snapshot_json'], true, 512, JSON_THROW_ON_ERROR),
            'selected_codes' => json_decode((string) $row['response_json'], true, 512, JSON_THROW_ON_ERROR),
        ]);
    }

    private function secureShuffle(array &$items): void
    {
        for ($index = count($items) - 1; $index > 0; $index--) {
            $swap = random_int(0, $index);
            [$items[$index], $items[$swap]] = [$items[$swap], $items[$index]];
        }
    }

    /** @param mixed $value */
    private function json($value): string
    {
        return json_encode($value, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR);
    }

    private function nowText(): string
    {
        return $this->format($this->clock->now());
    }

    private function format(\DateTimeImmutable $date): string
    {
        return $date->setTimezone(new \DateTimeZone('UTC'))->format('Y-m-d\TH:i:s\Z');
    }
}
