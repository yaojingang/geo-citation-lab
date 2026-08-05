<?php

declare(strict_types=1);

namespace GeoAssessment\Assessment;

final class ChoicePresenter
{
    /** @param list<array<string, mixed>> $choices @return list<array<string, mixed>> */
    public function present(array $choices): array
    {
        $presented = [];
        foreach (array_values($choices) as $index => $choice) {
            $sourceCode = (string) $choice['code'];
            $presented[] = array_merge($choice, [
                'code' => chr(65 + $index),
                'source_code' => $sourceCode,
                'rationale' => (bool) $choice['is_correct']
                    ? (string) $choice['rationale']
                    : $this->specificRationale((string) $choice['rationale'], $sourceCode),
            ]);
        }
        return $presented;
    }

    private function specificRationale(string $rationale, string $sourceCode): string
    {
        foreach (preg_split('/[;；]/u', $rationale) ?: [] as $segment) {
            if (preg_match('/^\s*([A-E](?:、[A-E])*)\s+(.+?)\s*$/u', $segment, $matches) !== 1) {
                continue;
            }
            $codes = explode('、', $matches[1]);
            if (in_array($sourceCode, $codes, true)) {
                return trim($matches[2]);
            }
        }
        return $rationale;
    }
}
