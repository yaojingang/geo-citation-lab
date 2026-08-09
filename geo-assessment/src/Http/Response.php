<?php

declare(strict_types=1);

namespace GeoAssessment\Http;

final class Response
{
    /** @var string */
    private $body;

    /** @var int */
    private $status;

    /** @var array<string, string|list<string>> */
    private $headers;

    /** @var bool */
    private $analyticsAllowed;

    /** @param array<string, string|list<string>> $headers */
    public function __construct(string $body = '', int $status = 200, array $headers = [], bool $analyticsAllowed = false)
    {
        $this->body = $body;
        $this->status = $status;
        $this->headers = $headers;
        $this->analyticsAllowed = $analyticsAllowed;
    }

    /** @return mixed */
    public function __get(string $name)
    {
        if ($name === 'body') {
            return $this->body;
        }
        if ($name === 'status') {
            return $this->status;
        }
        if ($name === 'headers') {
            return $this->headers;
        }
        if ($name === 'analyticsAllowed') {
            return $this->analyticsAllowed;
        }

        throw new \OutOfBoundsException("未知响应属性：{$name}");
    }

    /** @param mixed $value */
    public function __set(string $name, $value): void
    {
        throw new \LogicException("响应属性不可修改：{$name}");
    }

    public function __isset(string $name): bool
    {
        return $name === 'body' || $name === 'status' || $name === 'headers' || $name === 'analyticsAllowed';
    }

    public static function html(string $body, int $status = 200): self
    {
        return new self($body, $status, ['Content-Type' => 'text/html; charset=UTF-8']);
    }

    public static function json(array $data, int $status = 200): self
    {
        return new self(json_encode($data, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_THROW_ON_ERROR), $status, ['Content-Type' => 'application/json; charset=UTF-8']);
    }

    public static function redirect(string $url, int $status = 303): self
    {
        return new self('', $status, ['Location' => $url]);
    }

    public function withHeader(string $name, string $value, bool $append = false): self
    {
        $headers = $this->headers;
        if ($append && isset($headers[$name])) {
            $current = is_array($headers[$name]) ? $headers[$name] : [$headers[$name]];
            $headers[$name] = array_merge($current, [$value]);
        } else {
            $headers[$name] = $value;
        }
        return new self($this->body, $this->status, $headers, $this->analyticsAllowed);
    }

    /** @param array<string, string> $headers */
    public function withHeaders(array $headers): self
    {
        return new self($this->body, $this->status, array_merge($this->headers, $headers), $this->analyticsAllowed);
    }

    public function withAnalyticsAllowed(bool $allowed): self
    {
        return new self($this->body, $this->status, $this->headers, $allowed);
    }

    public function send(): void
    {
        http_response_code($this->status);
        foreach ($this->headers as $name => $values) {
            foreach ((array) $values as $index => $value) {
                header($name . ': ' . $value, $index === 0);
            }
        }
        echo $this->body;
    }
}
