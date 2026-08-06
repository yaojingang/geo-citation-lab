<?php

declare(strict_types=1);

namespace GeoAssessment\Http;

final class Request
{
    /** @var string */
    private $method;

    /** @var string */
    private $path;

    /** @var array<string, mixed> */
    private $input;

    /** @var array<string, string> */
    private $cookies;

    /** @var array<string, string> */
    private $headers;

    /** @var array<string, mixed> */
    private $query;

    /** @var string */
    private $remoteAddress;

    /** @var bool */
    private $secure;

    /** @param array<string, mixed> $input @param array<string, string> $cookies @param array<string, string> $headers */
    public function __construct(
        string $method,
        string $path,
        array $input = [],
        array $cookies = [],
        array $headers = [],
        array $query = [],
        string $remoteAddress = '127.0.0.1',
        bool $secure = false
    ) {
        $this->method = $method;
        $this->path = $path;
        $this->input = $input;
        $this->cookies = $cookies;
        $this->headers = $headers;
        $this->query = $query;
        $this->remoteAddress = $remoteAddress;
        $this->secure = $secure;
    }

    /** @return mixed */
    public function __get(string $name)
    {
        if ($name === 'method') {
            return $this->method;
        }
        if ($name === 'path') {
            return $this->path;
        }

        throw new \OutOfBoundsException("未知请求属性：{$name}");
    }

    /** @param mixed $value */
    public function __set(string $name, $value): void
    {
        throw new \LogicException("请求属性不可修改：{$name}");
    }

    public function __isset(string $name): bool
    {
        return $name === 'method' || $name === 'path';
    }

    public static function fromGlobals(string $basePath = ''): self
    {
        $method = strtoupper($_SERVER['REQUEST_METHOD'] ?? 'GET');
        $uri = parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH) ?: '/';
        if ($basePath !== '' && ($uri === $basePath || self::startsWith($uri, rtrim($basePath, '/') . '/'))) {
            $uri = substr($uri, strlen($basePath)) ?: '/';
        }
        if (isset($_GET['r']) && is_string($_GET['r'])) {
            $uri = '/' . ltrim($_GET['r'], '/');
        }
        $headers = [];
        foreach ($_SERVER as $key => $value) {
            if (self::startsWith($key, 'HTTP_') && is_string($value)) {
                $headers[strtolower(str_replace('_', '-', substr($key, 5)))] = $value;
            }
        }
        if (isset($_SERVER['CONTENT_TYPE'])) {
            $headers['content-type'] = (string) $_SERVER['CONTENT_TYPE'];
        }
        $input = $_POST;
        if (strpos(strtolower($headers['content-type'] ?? ''), 'application/json') !== false) {
            $decoded = json_decode((string) file_get_contents('php://input'), true);
            $input = is_array($decoded) ? $decoded : [];
        }

        $secure = !empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off';

        return new self($method, rawurldecode($uri), $input, array_map('strval', $_COOKIE), $headers, $_GET, (string) ($_SERVER['REMOTE_ADDR'] ?? '127.0.0.1'), $secure);
    }

    /** @return mixed */
    public function input(string $key, $default = null)
    {
        return $this->input[$key] ?? $default;
    }

    public function cookie(string $key): ?string
    {
        return $this->cookies[$key] ?? null;
    }

    public function header(string $key): ?string
    {
        return $this->headers[strtolower($key)] ?? null;
    }

    /** @return mixed */
    public function query(string $key, $default = null)
    {
        return $this->query[$key] ?? $default;
    }

    public function expectsJson(): bool
    {
        return strpos(strtolower($this->header('accept') ?? ''), 'application/json') !== false
            || strpos(strtolower($this->header('content-type') ?? ''), 'application/json') !== false;
    }

    public function remoteAddress(bool $trustProxy = false): string
    {
        if ($trustProxy && $this->header('x-forwarded-for')) {
            return trim(explode(',', (string) $this->header('x-forwarded-for'))[0]);
        }
        return $this->remoteAddress;
    }

    public function absoluteUrl(string $path, bool $trustProxy = false): string
    {
        $forwardedProto = strtolower(trim(explode(',', (string) ($this->header('x-forwarded-proto') ?? ''))[0]));
        $scheme = $this->secure || ($trustProxy && $forwardedProto === 'https') ? 'https' : 'http';
        $host = trim((string) ($this->header('host') ?? 'localhost'));
        if (preg_match('/\A(?:[A-Za-z0-9.-]+|\[[A-Fa-f0-9:.]+\])(?::[0-9]{1,5})?\z/D', $host) !== 1) {
            $host = 'localhost';
        }

        return $scheme . '://' . $host . '/' . ltrim($path, '/');
    }

    private static function startsWith(string $value, string $prefix): bool
    {
        return strncmp($value, $prefix, strlen($prefix)) === 0;
    }
}
