<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Reporting\InsightService;
use PHPUnit\Framework\TestCase;

final class InsightServiceTest extends TestCase
{
    private const LABELS = [
        'mechanism' => 'GEO 基础理解',
        'content' => '内容与证据优化',
        'measurement' => '测量与实验判断',
        'overseas' => '平台与来源差异',
        'domestic' => '国内平台与内容生态',
        'governance' => '落地与风险治理',
    ];

    public function testAllUnansweredDoesNotInventAStrength(): void
    {
        $dimensions = $this->dimensions(0);
        $questions = $this->questions(false, 0);

        $insights = (new InsightService())->build($dimensions, $questions);

        self::assertSame('尚无可判定强项', $insights['strength']['title']);
        self::assertStringContainsString('未作答', $insights['strength']['text']);
        self::assertSame('当前证据不足，完成作答后再判断能力结构', $insights['headline']);
        self::assertArrayHasKey('time_strategy', $insights);
    }

    public function testPerfectScoreReturnsConsolidationGuidance(): void
    {
        $insights = (new InsightService())->build($this->dimensions(100), $this->questions(true, 25));

        self::assertSame('六维均已系统掌握', $insights['strength']['title']);
        self::assertSame('六维表现均衡，下一步关注迁移与复验', $insights['headline']);
        self::assertStringContainsString('巩固', $insights['recommendations'][0]['title']);
    }

    public function testTieBreakingUsesFrozenDimensionOrder(): void
    {
        $dimensions = array_reverse($this->dimensions(50), true);

        $insights = (new InsightService())->build($dimensions, $this->questions(false, 20));

        self::assertSame('mechanism', $insights['recommendations'][0]['dimension']);
        self::assertSame('content', $insights['recommendations'][1]['dimension']);
    }

    public function test_zero_time_answers_are_treated_as_insufficient_telemetry(): void
    {
        $questions = $this->questions(false, 0);
        foreach ($questions as &$question) {
            $question['is_answered'] = true;
        }
        unset($question);

        $insights = (new InsightService())->build($this->dimensions(30), $questions);

        self::assertSame('用时样本不足', $insights['time_strategy']['title']);
    }

    public function test_sparse_answers_do_not_claim_a_stable_strength(): void
    {
        $questions = $this->questions(false, 0);
        $questions[0]['is_answered'] = true;
        $questions[0]['is_correct'] = true;
        $questions[0]['time_spent_seconds'] = 12;

        $insights = (new InsightService())->build($this->dimensions(25), $questions);

        self::assertSame('作答样本较少，先补齐证据再判断能力结构', $insights['headline']);
        self::assertSame('当前证据有限', $insights['strength']['title']);
        self::assertStringContainsString('仅完成 1 道题', $insights['strength']['text']);
        self::assertStringStartsWith('优先补齐', $insights['recommendations'][0]['title']);
    }

    private function dimensions(float $percentage): array
    {
        $result = [];
        foreach (self::LABELS as $key => $label) {
            $result[$key] = ['key' => $key, 'label' => $label, 'earned' => (int) round(17 * $percentage / 100), 'possible' => 17, 'percentage' => $percentage];
        }
        return $result;
    }

    private function questions(bool $correct, int $seconds): array
    {
        $questions = [];
        $number = 1;
        foreach (self::LABELS as $key => $label) {
            $questions[] = ['code' => sprintf('Q%02d', $number++), 'dimension' => $key, 'is_correct' => $correct, 'is_answered' => $seconds > 0, 'weight' => 3, 'time_spent_seconds' => $seconds];
        }
        return $questions;
    }
}
