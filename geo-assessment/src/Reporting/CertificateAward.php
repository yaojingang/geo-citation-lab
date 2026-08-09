<?php

declare(strict_types=1);

namespace GeoAssessment\Reporting;

use InvalidArgumentException;

final class CertificateAward
{
    /** @return array{title: string, highlight: string, encouragement: string} */
    public static function forScore(int $score): array
    {
        if ($score < 0 || $score > 100) {
            throw new InvalidArgumentException('证书分数必须位于 0 至 100');
        }

        if ($score >= 90) {
            return [
                'title' => 'GEO 大师',
                'highlight' => '系统、可复用的 GEO 专业判断力',
                'encouragement' => '你已形成系统、可复用的 GEO 专业判断力，能够独立完成策略设计、内容证据组织与效果验证',
            ];
        }
        if ($score >= 80) {
            return [
                'title' => 'GEO 专家',
                'highlight' => '扎实、清晰的 GEO 专业框架',
                'encouragement' => '你已建立扎实、清晰的 GEO 专业框架，能够把策略判断转化为稳定的内容与验证行动',
            ];
        }
        if ($score >= 70) {
            return [
                'title' => 'GEO 精英',
                'highlight' => '关键方法与实践能力',
                'encouragement' => '你已掌握 GEO 的关键方法，继续强化证据组织与测量实验，将更快形成完整的实战能力',
            ];
        }

        return [
            'title' => '',
            'highlight' => '清晰、可持续的 GEO 成长路径',
            'encouragement' => '你已经建立 GEO 学习基础，聚焦核心原理、内容证据与效果验证，将逐步形成清晰、可持续的 GEO 成长路径',
        ];
    }
}
