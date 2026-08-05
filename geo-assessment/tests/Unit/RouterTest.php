<?php

declare(strict_types=1);

namespace GeoAssessment\Tests\Unit;

use GeoAssessment\Http\Request;
use GeoAssessment\Http\Response;
use GeoAssessment\Http\Router;
use PHPUnit\Framework\TestCase;

final class RouterTest extends TestCase
{
    public function testDispatchesDynamicRouteParameters(): void
    {
        $router = new Router();
        $router->add('GET', '/attempts/{id}/questions/{position}', static fn (Request $request, array $params): Response => Response::json($params));

        $response = $router->dispatch(new Request('GET', '/attempts/abc-123/questions/9'));

        self::assertNotNull($response);
        self::assertSame(200, $response->status);
        self::assertSame(['id' => 'abc-123', 'position' => '9'], json_decode($response->body, true, 512, JSON_THROW_ON_ERROR));
    }

    public function testMethodAndFullPathMustMatch(): void
    {
        $router = new Router();
        $router->add('POST', '/identity', static fn (): Response => Response::html('ok'));

        self::assertNull($router->dispatch(new Request('GET', '/identity')));
        self::assertNull($router->dispatch(new Request('POST', '/identity/extra')));
    }
}
