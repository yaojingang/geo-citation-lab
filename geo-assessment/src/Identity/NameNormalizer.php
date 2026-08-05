<?php

declare(strict_types=1);

namespace GeoAssessment\Identity;

use InvalidArgumentException;

final class NameNormalizer
{
    /** @return array{display_name: string, normalized_name: string} */
    public function normalize(string $input): array
    {
        if (preg_match('/[\p{Cc}\p{Cf}]/u', $input) === 1) {
            throw new InvalidArgumentException('姓名不能包含控制字符。');
        }

        $displayName = preg_replace('/[\p{Z}\s]+/u', ' ', trim($input));
        if (!is_string($displayName)) {
            throw new InvalidArgumentException('姓名格式无法识别。');
        }

        $length = mb_strlen($displayName, 'UTF-8');
        if ($length < 2 || $length > 40) {
            throw new InvalidArgumentException('请输入 2 至 40 个字符的姓名。');
        }

        return [
            'display_name' => $displayName,
            'normalized_name' => mb_strtolower($displayName, 'UTF-8'),
        ];
    }
}
