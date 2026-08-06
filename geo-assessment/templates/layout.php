<?php

declare(strict_types=1);

use GeoAssessment\Support\View;
use GeoAssessment\Support\BaiduAnalytics;

$assetUrl = static function (string $path) use ($basePath): string {
    $absolutePath = dirname(__DIR__) . '/public' . $path;
    $version = is_file($absolutePath) ? (string) filemtime($absolutePath) : '1';

    return $basePath . $path . '?v=' . rawurlencode($version);
};
$footerQuestionSetVersion = isset($attempt) && is_array($attempt) && trim((string) ($attempt['set_version'] ?? '')) !== ''
    ? (string) $attempt['set_version']
    : 'geo-30-v1.2';
?>
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="color-scheme" content="light">
  <meta name="theme-color" content="#FFFFFF">
  <meta name="description" content="30 道题、30 分钟的 GEO 在线能力测试，面向初学者并以国内应用场景为主">
  <title><?= View::e($title) ?></title>
  <link rel="stylesheet" href="<?= View::e($assetUrl('/assets/app.css')) ?>">
</head>
<body class="<?= View::e($pageClass) ?>">
  <a class="skip-link" href="#main-content">跳到主要内容</a>
  <header class="site-header" aria-label="产品导航">
    <a class="brand" href="<?= View::e($basePath . '/') ?>" aria-label="GEO 能力测试首页">
      <span class="brand-mark" aria-hidden="true">G</span>
      <span>GEO Assessment</span>
    </a>
    <a class="quiet-link header-home" href="<?= View::e($basePath . '/') ?>">首页</a>
  </header>
  <?= $content ?>
  <footer class="site-footer no-print">
    <p>GEO Citation Lab · 题集 <?= View::e($footerQuestionSetVersion) ?></p>
    <nav aria-label="隐私与规则"><a href="<?= View::e($basePath . '/#privacy') ?>">隐私边界</a><span aria-hidden="true">·</span><a href="<?= View::e($basePath . '/#terms') ?>">测试规则</a></nav>
  </footer>
  <?php if (BaiduAnalytics::enabled($baiduAnalyticsId)): ?>
    <script><?= BaiduAnalytics::inlineLoader($baiduAnalyticsId) ?></script>
  <?php endif; ?>
  <script src="<?= View::e($assetUrl('/assets/app.js')) ?>" defer></script>
  <?php if ($pageClass === 'page-question'): ?>
    <script src="<?= View::e($assetUrl('/assets/quiz-state.js')) ?>" defer></script>
    <script src="<?= View::e($assetUrl('/assets/quiz.js')) ?>" defer></script>
  <?php endif; ?>
  <?php if ($pageClass === 'page-report'): ?>
    <script src="<?= View::e($assetUrl('/assets/vendor/chart.umd.min.js')) ?>" defer></script>
    <script src="<?= View::e($assetUrl('/assets/report.js')) ?>" defer></script>
  <?php endif; ?>
</body>
</html>
