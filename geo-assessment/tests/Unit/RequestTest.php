<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Http\Request;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

final class RequestTest extends TestCase
{
    #[DataProvider('pathProvider')]
    public function test_base_path_is_removed_only_at_a_path_boundary(string $requestUri, string $expected): void
    {
        $server = $_SERVER;
        $get = $_GET;
        $post = $_POST;
        $cookie = $_COOKIE;
        try {
            $_SERVER = ['REQUEST_METHOD' => 'GET', 'REQUEST_URI' => $requestUri, 'REMOTE_ADDR' => '127.0.0.1'];
            $_GET = [];
            $_POST = [];
            $_COOKIE = [];

            self::assertSame($expected, Request::fromGlobals('/geo')->path);
        } finally {
            $_SERVER = $server;
            $_GET = $get;
            $_POST = $post;
            $_COOKIE = $cookie;
        }
    }

    public static function pathProvider(): array
    {
        return [
            'prefixed route' => ['/geo/reports/1', '/reports/1'],
            'prefix root' => ['/geo', '/'],
            'similar route' => ['/geology', '/geology'],
        ];
    }

    public function test_route_identity_cannot_be_mutated_after_construction(): void
    {
        $request = new Request('GET', '/reports/example');

        $this->expectException(\LogicException::class);
        $request->method = 'POST';
    }
}
