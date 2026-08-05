<?php

declare(strict_types=1);

namespace GeoAssessment;

use DomainException;
use GeoAssessment\Assessment\AttemptService;
use GeoAssessment\Http\Controller\AttemptController;
use GeoAssessment\Http\Controller\HomeController;
use GeoAssessment\Http\Controller\IdentityController;
use GeoAssessment\Http\Controller\ReportController;
use GeoAssessment\Http\Csrf;
use GeoAssessment\Http\RateLimiter;
use GeoAssessment\Http\Request;
use GeoAssessment\Http\Response;
use GeoAssessment\Http\Router;
use GeoAssessment\Http\SecurityHeaders;
use GeoAssessment\Identity\IdentityService;
use GeoAssessment\Reporting\ReportViewModelFactory;
use GeoAssessment\Support\Config;
use GeoAssessment\Support\BaiduAnalytics;
use GeoAssessment\Support\Database;
use GeoAssessment\Support\JsonLogger;
use GeoAssessment\Support\View;

final class Bootstrap
{
    /** @var Config */
    private $config;

    public function __construct(Config $config)
    {
        $this->config = $config;
    }

    public function handle(Request $request): Response
    {
        $view = new View(
            $this->config->get('root') . '/templates',
            (string) $this->config->get('base_path', ''),
            $this->baiduAnalyticsId()
        );
        try {
            $appKey = $this->appKey();
            $pdo = Database::connect((string) $this->config->get('db_path'));
            $csrf = new Csrf($appKey);
            $identities = new IdentityService($pdo);
            $attempts = new AttemptService($pdo);
            $limiter = new RateLimiter($pdo, $appKey);
            $home = new HomeController($view, $identities, $attempts, $csrf);
            $secureCookie = $this->secureCookie($request);
            $identity = new IdentityController($view, $home, $identities, $attempts, $limiter, (bool) $this->config->get('trust_proxy', false), $secureCookie);
            $attempt = new AttemptController($view, $identities, $attempts, $csrf, $limiter);
            $report = new ReportController($view, $identities, new ReportViewModelFactory($pdo));

            if ($request->method === 'POST' && !$csrf->validate($request)) {
                $response = $request->expectsJson()
                    ? Response::json(['error' => '页面验证已失效，请刷新后重试。'], 403)
                    : $view->render('error', ['view' => $view, 'statusCode' => 403, 'heading' => '页面验证已失效', 'message' => '刷新页面后可继续当前操作。'], '页面验证已失效', 'page-error', 403);
                return $this->finish($response, $csrf, $secureCookie);
            }

            $router = new Router();
            $router->add('GET', '/', static function (Request $req) use ($home): Response {
                return $home->index($req);
            });
            $router->add('POST', '/identity', static function (Request $req) use ($identity): Response {
                return $identity->create($req);
            });
            $router->add('POST', '/switch-user', static function (Request $req) use ($identity): Response {
                return $identity->switch($req);
            });
            $router->add('POST', '/me/delete', static function (Request $req) use ($identity): Response {
                return $identity->delete($req);
            });
            $router->add('POST', '/attempts', static function (Request $req) use ($attempt): Response {
                return $attempt->start($req);
            });
            $router->add('GET', '/attempts/{id}/questions/{position}', static function (Request $req, array $params) use ($attempt): Response {
                return $attempt->question($req, $params);
            });
            $router->add('POST', '/attempts/{id}/answers', static function (Request $req, array $params) use ($attempt): Response {
                return $attempt->save($req, $params);
            });
            $router->add('POST', '/attempts/{id}/submit', static function (Request $req, array $params) use ($attempt): Response {
                return $attempt->submit($req, $params);
            });
            $router->add('GET', '/reports/{id}', static function (Request $req, array $params) use ($report): Response {
                return $report->show($req, $params);
            });

            $response = $router->dispatch($request);
            if ($response === null) {
                $response = $view->render('error', ['view' => $view, 'statusCode' => 404, 'heading' => '页面未找到', 'message' => '检查地址，或返回首页继续测试。'], '页面未找到', 'page-error', 404);
            }
            return $this->finish($response, $csrf, $secureCookie);
        } catch (DomainException $error) {
            $response = $request->expectsJson()
                ? Response::json(['error' => $error->getMessage()], 401)
                : Response::redirect($view->url('/'));
            return $response->withHeaders(SecurityHeaders::all($this->baiduAnalyticsId()));
        } catch (\Throwable $error) {
            (new JsonLogger((string) $this->config->get('log_dir')))->error($error, [
                'method' => $request->method,
                'path' => $request->path,
            ]);
            $debug = (bool) $this->config->get('debug', false);
            $message = $debug ? $error->getMessage() : '应用正在完成初始化，或当前服务暂不可用。';
            $response = $view->render('maintenance', ['view' => $view, 'message' => $message], '服务暂不可用', 'page-maintenance', 503);
            return $response->withHeaders(SecurityHeaders::all($this->baiduAnalyticsId()))->withHeader('Retry-After', '60');
        }
    }

    private function appKey(): string
    {
        $configured = $this->config->get('app_key');
        if (is_string($configured) && strlen($configured) >= 32) {
            return $configured;
        }
        $path = $this->config->get('root') . '/storage/app.key';
        $key = is_file($path) ? trim((string) file_get_contents($path)) : '';
        if (strlen($key) < 32) {
            throw new \RuntimeException('应用尚未安装，请运行 php bin/console app:install。');
        }
        return $key;
    }

    private function secureCookie(Request $request): bool
    {
        $setting = (string) $this->config->get('cookie_secure', 'auto');
        if ($setting === '1') {
            return true;
        }
        if ($setting === '0') {
            return false;
        }
        $directHttps = !empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off';
        $trustedProxyHttps = (bool) $this->config->get('trust_proxy', false)
            && strtolower($request->header('x-forwarded-proto') ?? '') === 'https';
        return $directHttps || $trustedProxyHttps;
    }

    private function finish(Response $response, Csrf $csrf, bool $secureCookie): Response
    {
        $response = $response->withHeaders(SecurityHeaders::all($this->baiduAnalyticsId()))->withHeader('X-Request-Id', bin2hex(random_bytes(8)));
        if (($secret = $csrf->issuedSecret()) !== null) {
            $parts = [
                Csrf::COOKIE . '=' . rawurlencode($secret),
                'Max-Age=315360000',
                'Path=' . ((string) $this->config->get('base_path', '') ?: '/'),
                'HttpOnly',
                'SameSite=Lax',
            ];
            if ($secureCookie) {
                $parts[] = 'Secure';
            }
            $response = $response->withHeader('Set-Cookie', implode('; ', $parts), true);
        }
        return $response;
    }

    private function baiduAnalyticsId(): string
    {
        return BaiduAnalytics::normalize($this->config->get('baidu_analytics_id', ''));
    }
}
