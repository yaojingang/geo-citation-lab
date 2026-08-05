<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Integration;

use GeoAssessment\Assessment\QuestionSetValidator;
use PHPUnit\Framework\TestCase;

final class QuestionImportTest extends TestCase
{
    public function test_the_published_question_set_satisfies_the_frozen_blueprint(): void
    {
        $path = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.1.json';
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);

        $summary = (new QuestionSetValidator())->validate($payload);

        self::assertSame(30, $summary['question_count']);
        self::assertSame(100, $summary['total_points']);
        self::assertSame(20, $summary['single_count']);
        self::assertSame(10, $summary['multiple_count']);
        self::assertSame(['basic' => 10, 'advanced' => 13, 'challenge' => 7], $summary['difficulties']);
        self::assertSame(56, $summary['source_count']);
        self::assertSame(54, $summary['covered_papers']);
        self::assertSame(1520, $summary['expected_seconds']);
    }

    public function test_it_rejects_a_changed_answer_that_is_not_an_available_choice(): void
    {
        $path = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.1.json';
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
        $payload['questions'][0]['correct_codes'] = ['Z'];

        $this->expectExceptionMessage('Q01');
        (new QuestionSetValidator())->validate($payload);
    }

    public function test_it_rejects_a_non_https_source_link(): void
    {
        $path = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.1.json';
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
        $payload['sources'][0]['url'] = 'javascript:alert(1)';

        $this->expectExceptionMessage('来源 P01');
        (new QuestionSetValidator())->validate($payload);
    }

    public function test_it_rejects_choice_flags_that_disagree_with_correct_codes(): void
    {
        $path = dirname(__DIR__, 2) . '/database/seeds/geo-30-v1.1.json';
        $payload = json_decode((string) file_get_contents($path), true, 512, JSON_THROW_ON_ERROR);
        $payload['questions'][0]['choices'][0]['is_correct'] = !$payload['questions'][0]['choices'][0]['is_correct'];

        $this->expectExceptionMessage('Q01');
        (new QuestionSetValidator())->validate($payload);
    }
}
