<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Assessment\ChoicePresenter;
use PHPUnit\Framework\TestCase;

final class ChoicePresenterTest extends TestCase
{
    public function test_choices_receive_neutral_display_codes_and_specific_distractor_rationales(): void
    {
        $choices = [
            ['code' => 'D', 'text' => '选项 D', 'is_correct' => false, 'rationale' => 'A、B 忽略了检索过程；D 把相关性当成因果性。'],
            ['code' => 'B', 'text' => '选项 B', 'is_correct' => false, 'rationale' => 'A、B 忽略了检索过程；D 把相关性当成因果性。'],
            ['code' => 'C', 'text' => '选项 C', 'is_correct' => true, 'rationale' => '这是正确判断的完整理由。'],
            ['code' => 'A', 'text' => '选项 A', 'is_correct' => false, 'rationale' => 'A、B 忽略了检索过程；D 把相关性当成因果性。'],
        ];

        $presented = (new ChoicePresenter())->present($choices);

        self::assertSame(['A', 'B', 'C', 'D'], array_column($presented, 'code'));
        self::assertSame(['D', 'B', 'C', 'A'], array_column($presented, 'source_code'));
        self::assertSame('把相关性当成因果性。', $presented[0]['rationale']);
        self::assertSame('忽略了检索过程', $presented[1]['rationale']);
        self::assertSame('这是正确判断的完整理由。', $presented[2]['rationale']);
        self::assertSame('忽略了检索过程', $presented[3]['rationale']);
    }
}
