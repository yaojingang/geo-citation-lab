<?php

declare(strict_types=1);

namespace GeoAssessment\Scoring;

use InvalidArgumentException;

final class ScoreBands
{
    /** @return array{label: string, range: string, summary: string} */
    public static function total(int $score): array
    {
        if ($score < 0 || $score > 100) {
            throw new InvalidArgumentException('总分必须位于 0 至 100。');
        }

        if ($score <= 39) {
            return ['label' => '探索期', 'range' => '0 至 39', 'summary' => '正在建立 GEO 的基本概念与分析语言。'];
        }
        if ($score <= 59) {
            return ['label' => '基础期', 'range' => '40 至 59', 'summary' => '已理解核心概念，需要强化测量与情境判断。'];
        }
        if ($score <= 74) {
            return ['label' => '实践期', 'range' => '60 至 74', 'summary' => '可以将主要原理应用到内容与评估任务。'];
        }
        if ($score <= 89) {
            return ['label' => '系统期', 'range' => '75 至 89', 'summary' => '已建立跨机制、平台与治理的系统认知。'];
        }
        return ['label' => '研究策略期', 'range' => '90 至 100', 'summary' => '具备建立 GEO 研究与实验策略的综合判断能力。'];
    }

    public static function dimension(float $percentage): string
    {
        if ($percentage < 40) {
            return '需补课';
        }
        if ($percentage < 60) {
            return '基础理解';
        }
        if ($percentage < 80) {
            return '可实践';
        }
        return '系统掌握';
    }
}
