<?php

declare(strict_types=1);

namespace GeoAssessment\Http;

final class Router
{
    /** @var list<array{method: string, pattern: string, handler: callable}> */
    private $routes = [];

    public function add(string $method, string $pattern, callable $handler): void
    {
        $this->routes[] = ['method' => strtoupper($method), 'pattern' => $pattern, 'handler' => $handler];
    }

    public function dispatch(Request $request): ?Response
    {
        foreach ($this->routes as $route) {
            if ($route['method'] !== $request->method) {
                continue;
            }
            $regex = preg_quote($route['pattern'], '#');
            $regex = preg_replace('/\\\\\{([A-Za-z_][A-Za-z0-9_]*)\\\\\}/', '(?P<$1>[^/]+)', $regex);
            if (!is_string($regex) || preg_match('#^' . $regex . '$#', $request->path, $matches) !== 1) {
                continue;
            }
            $params = array_filter($matches, 'is_string', ARRAY_FILTER_USE_KEY);
            return ($route['handler'])($request, $params);
        }
        return null;
    }
}
