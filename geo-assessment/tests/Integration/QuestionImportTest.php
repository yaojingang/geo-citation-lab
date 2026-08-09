<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Integration;

use GeoAssessment\Assessment\QuestionSetValidator;
use PHPUnit\Framework\TestCase;

final class QuestionImportTest extends TestCase
{
    public function test_the_published_question_set_satisfies_the_frozen_blueprint(): void
    {
        $path = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json';
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
        self::assertSame('beginner-domestic-v1', $payload['set']['validation_profile']);

        $summary = (new QuestionSetValidator())->validate($payload);

        self::assertSame(30, $summary['question_count']);
        self::assertSame(100, $summary['total_points']);
        self::assertSame(20, $summary['single_count']);
        self::assertSame(10, $summary['multiple_count']);
        self::assertSame(['basic' => 10, 'advanced' => 13, 'challenge' => 7], $summary['difficulties']);
        self::assertSame(['domestic' => 18, 'general' => 9, 'overseas' => 3], $summary['region_scopes']);
        self::assertSame(56, $summary['source_count']);
        self::assertSame(54, $summary['covered_papers']);
        self::assertSame(1520, $summary['expected_seconds']);
    }

    public function test_it_rejects_a_changed_answer_that_is_not_an_available_choice(): void
    {
        $path = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json';
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
        $payload['questions'][0]['correct_codes'] = ['Z'];

        $this->expectExceptionMessage('Q01');
        (new QuestionSetValidator())->validate($payload);
    }

    public function test_it_rejects_a_non_https_source_link(): void
    {
        $path = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json';
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
        $payload['sources'][0]['url'] = 'javascript:alert(1)';

        $this->expectExceptionMessage('来源 P01');
        (new QuestionSetValidator())->validate($payload);
    }

    public function test_it_rejects_choice_flags_that_disagree_with_correct_codes(): void
    {
        $path = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json';
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
        $payload['questions'][0]['choices'][0]['is_correct'] = !$payload['questions'][0]['choices'][0]['is_correct'];

        $this->expectExceptionMessage('Q01');
        (new QuestionSetValidator())->validate($payload);
    }

    public function test_it_rejects_research_jargon_in_a_beginner_prompt(): void
    {
        $path = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json';
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
        $payload['questions'][0]['prompt'] = '如何判断候选集里的 RAG 答案吸收？';

        $this->expectExceptionMessage('额外研究背景');
        (new QuestionSetValidator())->validate($payload);
    }

    public function test_it_rejects_terminal_periods_in_beginner_question_copy(): void
    {
        $path = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json';
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
        $payload['questions'][0]['explanation'] .= '。';

        $this->expectExceptionMessage('结尾句号');
        (new QuestionSetValidator())->validate($payload);
    }

    public function test_beginner_copy_rules_are_dispatched_by_an_explicit_profile(): void
    {
        $path = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json';
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
        $payload['set']['version'] = 'geo-30-v1.3';
        unset($payload['set']['validation_profile']);
        $payload['questions'][0]['prompt'] = '如何判断候选集里的 RAG 答案吸收？';

        $summary = (new QuestionSetValidator())->validate($payload);

        self::assertSame(30, $summary['question_count']);
    }

    public function test_it_rejects_inconsistent_labels_within_one_dimension(): void
    {
        $path = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.2.json';
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
        $payload['questions'][1]['dimension_label'] = '另一个维度名称';

        $this->expectExceptionMessage('维度名称不一致');
        (new QuestionSetValidator())->validate($payload);
    }
}
