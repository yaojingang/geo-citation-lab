<?php

declare(strict_types=1);

namespace GeoAssessment\Scoring;

final class SelectionDiagnostics
{
    /** @return array{missing_codes: list<string>, extra_codes: list<string>, label: string} */
    public function diagnose(array $correct, array $selected): array
    {
        $correct = $this->canonical($correct);
        $selected = $this->canonical($selected);
        $missing = array_values(array_diff($correct, $selected));
        $extra = array_values(array_diff($selected, $correct));

        if ($missing !== [] && $extra !== []) {
            $label = '同时存在漏选与误选';
        } elseif ($missing !== []) {
            $label = '漏选正确项';
        } elseif ($extra !== []) {
            $label = '选入错误项';
        } else {
            $label = '选择集合完全匹配';
        }

        return ['missing_codes' => $missing, 'extra_codes' => $extra, 'label' => $label];
    }

    /** @return list<string> */
    public function canonical(array $codes): array
    {
        $codes = array_values(array_unique(array_map('strval', $codes)));
        sort($codes, SORT_STRING);
        return $codes;
    }
}
