<?php

declare(strict_types=1);

namespace GeoAssessment\Assessment;

use PDO;
use RuntimeException;

final class QuestionImporter
{
    /** @var PDO */
    private $pdo;

    /** @var QuestionSetValidator */
    private $validator;

    public function __construct(PDO $pdo, ?QuestionSetValidator $validator = null)
    {
        $this->pdo = $pdo;
        $this->validator = $validator ?? new QuestionSetValidator();
    }

    public function import(string $seedPath, bool $activate = true): string
    {
        $json = file_get_contents($seedPath);
        if (!is_string($json)) {
            throw new RuntimeException("无法读取题库种子：{$seedPath}");
        }
        $payload = json_decode($json, true, 512, JSON_THROW_ON_ERROR);
        $summary = $this->validator->validate($payload);
        $set = $payload['set'];
        $version = (string) $set['version'];

        $existing = $this->pdo->prepare('SELECT id, content_hash FROM question_sets WHERE version = :version');
        $existing->execute(['version' => $version]);
        $row = $existing->fetch();
        if (is_array($row)) {
            if (!hash_equals((string) $row['content_hash'], $summary['content_hash'])) {
                throw new RuntimeException("题集 {$version} 已存在且内容指纹不同，请创建新版本。");
            }
            if ($activate) {
                $this->activate((string) $row['id']);
            }
            return 'unchanged';
        }

        $sourceMap = [];
        foreach ($payload['sources'] as $source) {
            $sourceMap[$source['id']] = $source;
        }
        $setId = substr(hash('sha256', $version), 0, 32);
        $now = gmdate('Y-m-d\TH:i:s\Z');

        $this->pdo->exec('BEGIN IMMEDIATE');
        $transactionStarted = true;
        try {
            if ($activate) {
                $this->pdo->exec('UPDATE question_sets SET active = 0 WHERE active = 1');
            }
            $insertSet = $this->pdo->prepare('INSERT INTO question_sets (id, version, title, total_points, time_limit_seconds, scoring_version, evidence_frozen_at, content_hash, sources_json, active, created_at) VALUES (:id, :version, :title, :total_points, :time_limit_seconds, :scoring_version, :evidence_frozen_at, :content_hash, :sources_json, :active, :created_at)');
            $insertSet->execute([
                'id' => $setId,
                'version' => $version,
                'title' => $set['title'],
                'total_points' => $set['total_points'],
                'time_limit_seconds' => $set['time_limit_seconds'],
                'scoring_version' => $set['scoring_version'],
                'evidence_frozen_at' => $set['evidence_frozen_at'],
                'content_hash' => $summary['content_hash'],
                'sources_json' => $this->json($payload['sources']),
                'active' => $activate ? 1 : 0,
                'created_at' => $now,
            ]);

            $insertQuestion = $this->pdo->prepare('INSERT INTO questions (id, set_id, code, title, type, dimension, dimension_label, difficulty, difficulty_label, cognitive_level, prompt, weight, sort_order, misconception_tag, explanation, principle, expected_seconds, source_refs_json, source_objects_json) VALUES (:id, :set_id, :code, :title, :type, :dimension, :dimension_label, :difficulty, :difficulty_label, :cognitive_level, :prompt, :weight, :sort_order, :misconception_tag, :explanation, :principle, :expected_seconds, :source_refs_json, :source_objects_json)');
            $insertChoice = $this->pdo->prepare('INSERT INTO choices (id, question_id, code, text, is_correct, rationale) VALUES (:id, :question_id, :code, :text, :is_correct, :rationale)');

            foreach ($payload['questions'] as $question) {
                $questionId = substr(hash('sha256', $version . ':' . $question['code']), 0, 32);
                $sourceObjects = array_values(array_map(static function (string $ref) use ($sourceMap): array {
                    return $sourceMap[$ref];
                }, $question['source_refs']));
                $insertQuestion->execute([
                    'id' => $questionId,
                    'set_id' => $setId,
                    'code' => $question['code'],
                    'title' => $question['title'],
                    'type' => $question['type'],
                    'dimension' => $question['dimension'],
                    'dimension_label' => $question['dimension_label'],
                    'difficulty' => $question['difficulty'],
                    'difficulty_label' => $question['difficulty_label'],
                    'cognitive_level' => $question['cognitive_level'],
                    'prompt' => $question['prompt'],
                    'weight' => $question['weight'],
                    'sort_order' => $question['sort_order'],
                    'misconception_tag' => $question['misconception_tag'],
                    'explanation' => $question['explanation'],
                    'principle' => $question['principle'],
                    'expected_seconds' => $question['expected_seconds'],
                    'source_refs_json' => $this->json($question['source_refs']),
                    'source_objects_json' => $this->json($sourceObjects),
                ]);
                foreach ($question['choices'] as $choice) {
                    $insertChoice->execute([
                        'id' => substr(hash('sha256', $version . ':' . $question['code'] . ':' . $choice['code']), 0, 32),
                        'question_id' => $questionId,
                        'code' => $choice['code'],
                        'text' => $choice['text'],
                        'is_correct' => $choice['is_correct'] ? 1 : 0,
                        'rationale' => $choice['rationale'],
                    ]);
                }
            }
            $this->pdo->exec('COMMIT');
            $transactionStarted = false;
        } catch (\Throwable $error) {
            if ($transactionStarted) {
                $this->pdo->exec('ROLLBACK');
            }
            throw $error;
        }

        return 'imported';
    }

    private function activate(string $setId): void
    {
        $this->pdo->beginTransaction();
        try {
            $this->pdo->exec('UPDATE question_sets SET active = 0 WHERE active = 1');
            $statement = $this->pdo->prepare('UPDATE question_sets SET active = 1 WHERE id = :id');
            $statement->execute(['id' => $setId]);
            $this->pdo->commit();
        } catch (\Throwable $error) {
            if ($this->pdo->inTransaction()) {
                $this->pdo->rollBack();
            }
            throw $error;
        }
    }

    /** @param mixed $value */
    private function json($value): string
    {
        return json_encode($value, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR);
    }
}
