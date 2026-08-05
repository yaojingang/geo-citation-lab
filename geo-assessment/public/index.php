<?php

declare(strict_types=1);

use GeoAssessment\Bootstrap;
use GeoAssessment\Http\Request;
use GeoAssessment\Support\Config;

require dirname(__DIR__) . '/vendor/autoload.php';

$config = Config::load(dirname(__DIR__) . '/config/app.php');
date_default_timezone_set((string) $config->get('timezone', 'Asia/Shanghai'));
$request = Request::fromGlobals((string) $config->get('base_path', ''));
(new Bootstrap($config))->handle($request)->send();
