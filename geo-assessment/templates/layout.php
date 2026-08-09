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
$metaDescription = $pageClass === 'page-certificate'
    ? 'GEO专业能力测试评估证书，展示综合得分、专业称号与六维能力画像'
    : '30 道题、30 分钟的 GEO 在线能力测试，面向初学者并以国内应用场景为主';
?>
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="color-scheme" content="light">
  <meta name="theme-color" content="#FFFFFF">
  <meta name="description" content="<?= View::e($metaDescription) ?>">
  <title><?= View::e($title) ?></title>
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%231b365d'/%3E%3Ctext x='32' y='43' text-anchor='middle' font-family='Georgia,serif' font-size='38' font-weight='700' fill='white'%3EG%3C/text%3E%3C/svg%3E">
  <?php if (BaiduAnalytics::enabled($baiduAnalyticsId)): ?>
    <script>var _hmt = _hmt || [];
(function() {
  var hm = document.createElement("script");
  hm.src = "https://hm.baidu.com/hm.js?<?= View::e($baiduAnalyticsId) ?>";
  var s = document.getElementsByTagName("script")[0];
  s.parentNode.insertBefore(hm, s);
})();</script>
  <?php endif; ?>
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
  <script src="<?= View::e($assetUrl('/assets/app.js')) ?>" defer></script>
  <?php if ($pageClass === 'page-question'): ?>
    <script src="<?= View::e($assetUrl('/assets/quiz-state.js')) ?>" defer></script>
    <script src="<?= View::e($assetUrl('/assets/quiz.js')) ?>" defer></script>
  <?php endif; ?>
  <?php if ($pageClass === 'page-report'): ?>
    <script src="<?= View::e($assetUrl('/assets/vendor/chart.umd.min.js')) ?>" defer></script>
    <script src="<?= View::e($assetUrl('/assets/report.js')) ?>" defer></script>
  <?php endif; ?>
  <?php if ($pageClass === 'page-certificate'): ?>
    <script src="<?= View::e($assetUrl('/assets/certificate.js')) ?>" defer></script>
  <?php endif; ?>
</body>
</html>
