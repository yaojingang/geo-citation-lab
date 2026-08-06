<?php

declare(strict_types=1);

namespace GeoAssessment\Reporting;

final class InsightService
{
    private const ORDER = ['mechanism', 'content', 'measurement', 'overseas', 'domestic', 'governance'];
    private const GUIDANCE = [
        'mechanism' => '按网页能否被找到、能否成为来源、内容能否进入答案三个步骤复盘',
        'content' => '把用户问题、事实依据、适用条件和页面结构放在一起检查',
        'measurement' => '用重复测试、对照页面和完整记录区分稳定表现与偶然变化',
        'overseas' => '分别记录平台、端别、时间、来源数量和内容采用程度',
        'domestic' => '重点检查国内平台差异、第三方信息、内容时效和数据缺失',
        'governance' => '保留来源记录、失败案例、回归测试和可恢复的旧版本',
    ];

    /** @param array<string, array<string, mixed>> $dimensions */
    public function build(array $dimensions, array $questions): array
    {
        $ranked = [];
        foreach (self::ORDER as $order => $key) {
            if (!isset($dimensions[$key])) {
                continue;
            }
            $dimensionQuestions = array_values(array_filter($questions, static function (array $question) use ($key): bool {
                return $question['dimension'] === $key;
            }));
            $ranked[] = array_merge($dimensions[$key], [
                'order' => $order,
                'correct_count' => count(array_filter($dimensionQuestions, static function (array $question): bool {
                    return (bool) $question['is_correct'];
                })),
                'high_weight_correct' => count(array_filter($dimensionQuestions, static function (array $question): bool {
                    return (bool) $question['is_correct'] && (int) $question['weight'] === 4;
                })),
            ]);
        }
        usort($ranked, static function (array $left, array $right): int {
            return [
                (float) $left['percentage'],
                (int) $left['high_weight_correct'],
                (int) $left['correct_count'],
                (int) $left['order'],
            ] <=> [
                (float) $right['percentage'],
                (int) $right['high_weight_correct'],
                (int) $right['correct_count'],
                (int) $right['order'],
            ];
        });
        $weak = array_slice($ranked, 0, 2);
        $strongRanked = $ranked;
        usort($strongRanked, static function (array $left, array $right): int {
            return [
                (float) $right['percentage'],
                (int) $right['high_weight_correct'],
                (int) $right['correct_count'],
                -(int) $right['order'],
            ] <=> [
                (float) $left['percentage'],
                (int) $left['high_weight_correct'],
                (int) $left['correct_count'],
                -(int) $left['order'],
            ];
        });
        $strong = $strongRanked[0] ?? ['key' => 'mechanism', 'label' => 'GEO 基础理解', 'percentage' => 0];
        $answeredQuestions = array_filter($questions, static function (array $question): bool {
            return (bool) ($question['is_answered'] ?? false);
        });
        $answeredCount = count($answeredQuestions);
        $answeredDimensions = count(array_unique(array_column($answeredQuestions, 'dimension')));
        $perfect = $ranked !== [] && count(array_filter($ranked, static function (array $dimension): bool {
            return (float) $dimension['percentage'] >= 100;
        })) === count($ranked);
        $limitedEvidence = $answeredCount > 0 && ($answeredCount < 12 || $answeredDimensions < 4);
        if ($answeredCount === 0) {
            $headline = '当前证据不足，完成作答后再判断能力结构';
        } elseif ($perfect) {
            $headline = '六维表现均衡，下一步关注迁移与复验';
        } elseif ($limitedEvidence) {
            $headline = '作答样本较少，先补齐证据再判断能力结构';
        } elseif ($ranked !== [] && max(array_column($ranked, 'percentage')) <= 0) {
            $headline = '先重建基本判断框架，再集中补强两个维度';
        } else {
            $headline = '先巩固强项，再集中补强两个维度';
        }

        $recommendations = [];
        foreach ($weak as $dimension) {
            $questionCodes = array_values(array_map(
                static function (array $question): string {
                    return $question['code'];
                },
                array_slice(array_filter($questions, static function (array $question) use ($dimension): bool {
                    return $question['dimension'] === $dimension['key'] && !$question['is_correct'];
                }), 0, 3)
            ));
            if ($perfect) {
                $title = '巩固「' . $dimension['label'] . '」的迁移能力';
                $text = '把当前判断方法用于新平台、新模型和新内容，并记录测试条件';
            } elseif ($limitedEvidence) {
                $title = '优先补齐「' . $dimension['label'] . '」的作答证据';
                $text = '先完成该维度的题目，再结合解析区分知识缺口和未作答情况';
            } else {
                $title = '优先补强「' . $dimension['label'] . '」';
                $text = self::GUIDANCE[$dimension['key']];
            }
            $recommendations[] = [
                'dimension' => $dimension['key'],
                'title' => $title,
                'text' => $text,
                'question_codes' => $questionCodes,
            ];
        }

        if ($answeredCount === 0) {
            $strength = ['dimension' => null, 'title' => '尚无可判定强项', 'text' => '本次未作答，六维得分只表示当前缺少可用信息。下次可从基础题开始完整作答'];
        } elseif ($perfect) {
            $strength = ['dimension' => null, 'title' => '六维均已系统掌握', 'text' => '本次六个维度均为 100.0%，后续可以用新的平台和内容场景复验'];
        } elseif ($limitedEvidence) {
            $strength = ['dimension' => null, 'title' => '当前证据有限', 'text' => '本次仅完成 ' . $answeredCount . ' 道题，覆盖 ' . $answeredDimensions . ' 个维度。当前得分可用于核对已答内容，完整作答后再判断能力结构'];
        } elseif ((float) $strong['percentage'] === 0.0) {
            $strength = ['dimension' => $strong['key'], 'title' => '当前未形成稳定强项', 'text' => '已作答题目尚未得分，可以结合题目解析重新理解基本判断方法'];
        } else {
            $strength = ['dimension' => $strong['key'], 'title' => '当前强项：' . $strong['label'], 'text' => '本维度得分率为 ' . number_format((float) $strong['percentage'], 1) . '%，可以继续用实际页面和平台测试巩固'];
        }

        return [
            'headline' => $headline,
            'strength' => $strength,
            'recommendations' => $recommendations,
            'time_strategy' => $this->timeStrategy($questions),
            'learning_path' => [
                ['step' => 1, 'title' => '回看错题', 'text' => '阅读错题的解析、原理和每个选项的说明'],
                ['step' => 2, 'title' => '联系实际页面', 'text' => '从两个最低维度各选一个真实页面或平台案例重新判断'],
                ['step' => 3, 'title' => '间隔后再测', 'text' => '完成复盘后再开始下一次测试，对比同版本维度变化'],
            ],
        ];
    }

    /** @return array{title: string, text: string, question_codes: list<string>} */
    private function timeStrategy(array $questions): array
    {
        $answered = array_values(array_filter($questions, static function (array $question): bool {
            return (bool) ($question['is_answered'] ?? false);
        }));
        if ($answered === []) {
            return ['title' => '暂无用时诊断', 'text' => '完成作答后，报告会区分知识缺口和时间分配问题', 'question_codes' => []];
        }
        $timed = array_values(array_filter($answered, static function (array $question): bool {
            return (int) $question['time_spent_seconds'] > 0;
        }));
        if (count($timed) < 3) {
            return ['title' => '用时样本不足', 'text' => '至少记录 3 道题的活跃用时后，报告才会判断时间分配', 'question_codes' => []];
        }
        $times = array_map(static function (array $question): int {
            return (int) $question['time_spent_seconds'];
        }, $timed);
        sort($times, SORT_NUMERIC);
        $middle = intdiv(count($times), 2);
        $median = count($times) % 2 === 0 ? ($times[$middle - 1] + $times[$middle]) / 2 : $times[$middle];
        $threshold = max(60, $median * 2.5);
        $slowIncorrect = array_values(array_filter($timed, static function (array $question) use ($threshold): bool {
            return !(bool) $question['is_correct'] && (int) $question['time_spent_seconds'] >= $threshold;
        }));
        usort($slowIncorrect, static function (array $left, array $right): int {
            return (int) $right['time_spent_seconds'] <=> (int) $left['time_spent_seconds'];
        });
        $codes = array_column(array_slice($slowIncorrect, 0, 3), 'code');
        if ($codes !== []) {
            return ['title' => '优先复盘高耗时错题', 'text' => '已答题活跃用时中位数约为 ' . number_format($median, 0) . ' 秒。可以先回看高耗时且失分的题目', 'question_codes' => $codes];
        }
        return ['title' => '当前时间分配较稳定', 'text' => '已答题活跃用时中位数约为 ' . number_format($median, 0) . ' 秒，未出现明显的高耗时错题集中', 'question_codes' => []];
    }
}
