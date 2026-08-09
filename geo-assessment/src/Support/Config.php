<?php

declare(strict_types=1);

namespace GeoAssessment\Support;

final class Config
{
    /** @var array<string, mixed> */
    private $values;

    public function __construct(array $values)
    {
        $this->values = $values;
    }

    public static function load(string $path): self
    {
        $values = require $path;
        return new self($values);
    }

    /** @return mixed */
    public function get(string $key, $default = null)
    {
        return $this->values[$key] ?? $default;
    }

    public function all(): array
    {
        return $this->values;
    }
}
