<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Http\Response;
use PHPUnit\Framework\TestCase;

final class ResponseTest extends TestCase
{
    public function test_response_transport_values_are_readable_and_immutable(): void
    {
        $response = Response::html('ok', 201);

        self::assertSame('ok', $response->body);
        self::assertSame(201, $response->status);
        self::assertSame('text/html; charset=UTF-8', $response->headers['Content-Type']);

        $this->expectException(\LogicException::class);
        $response->status = 500;
    }
}
