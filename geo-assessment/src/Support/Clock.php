<?php

declare(strict_types=1);

namespace GeoAssessment\Support;

interface Clock
{
    public function now(): \DateTimeImmutable;
}
