<?php

declare(strict_types=1);

use GeoAssessment\Support\View;

$dimensionSummary = implode('，', array_map(static function (array $dimension): string {
    return $dimension['short_label'] . ' ' . round((float) $dimension['score']);
}, $certificate['dimensions']));
$encouragement = (string) $certificate['award']['encouragement'];
$highlight = (string) $certificate['award']['highlight'];
$highlightPosition = mb_strpos($encouragement, $highlight, 0, 'UTF-8');
$encouragementBefore = $highlightPosition === false ? $encouragement : mb_substr($encouragement, 0, $highlightPosition, 'UTF-8');
$encouragementAfter = $highlightPosition === false ? '' : mb_substr(
    $encouragement,
    $highlightPosition + mb_strlen($highlight, 'UTF-8'),
    mb_strlen($encouragement, 'UTF-8'),
    'UTF-8'
);
$certificateData = [
    'name' => $certificate['recipient_name'],
    'score' => $certificate['score'],
    'tier' => $certificate['award']['title'],
    'id' => $certificate['number'],
    'date' => $certificate['issued_on'],
    'issuer' => $certificate['issuer'],
    'labels' => array_column($certificate['dimensions'], 'short_label'),
    'values' => array_column($certificate['dimensions'], 'score'),
    'encouragement' => $encouragement,
];
?>
<main id="main-content" class="certificate-page" data-certificate>
  <div id="certificate-data" hidden data-certificate-json="<?= View::jsonAttribute($certificateData) ?>"></div>

  <header class="certificate-toolbar no-print">
    <a class="certificate-page-brand" href="<?= View::e($view->url('/reports/' . $certificate['attempt_id'])) ?>" aria-label="返回 GEO 能力报告">
      <span class="certificate-brand-symbol" aria-hidden="true">G</span>
      <span><strong>GEO Assessment</strong><small>PROFESSIONAL CAPABILITY</small></span>
    </a>
    <nav class="certificate-actions" aria-label="证书操作">
      <a class="button button-secondary" href="<?= View::e($view->url('/reports/' . $certificate['attempt_id'])) ?>">返回报告</a>
      <button class="button button-primary" type="button" data-download-certificate>
        <svg class="certificate-download-icon" viewBox="0 0 20 20" aria-hidden="true"><path d="M10 2v10m0 0 4-4m-4 4L6 8M3 14v3h14v-3" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <span class="certificate-button-full">下载证书图片</span><span class="certificate-button-short">下载图片</span>
      </button>
    </nav>
  </header>

  <section class="certificate-stage">
    <header class="certificate-page-intro no-print">
      <div><p>VERIFIABLE GEO CERTIFICATE</p><h1><?= View::e($certificate['title']) ?></h1></div>
      <p>证书二维码关联本次评估记录，可用于查询与结果归因</p>
    </header>

    <div class="certificate-frame">
      <article id="certificate" class="certificate-paper" aria-label="<?= View::e($certificate['recipient_name']) ?> 的 <?= View::e($certificate['title']) ?>，总分 <?= View::e($certificate['score']) ?> 分<?= $certificate['award']['title'] !== '' ? '，称号 ' . View::e($certificate['award']['title']) : '' ?>">
        <header class="certificate-head">
          <div class="certificate-brand">
            <span class="certificate-brand-symbol" aria-hidden="true">G</span>
            <span><strong>GEO Assessment</strong><small>GEO CITATION LAB</small></span>
          </div>
          <div class="certificate-index">PROFESSIONAL CAPABILITY CERTIFICATE<strong>NO. <?= View::e($certificate['number']) ?></strong></div>
        </header>

        <div class="certificate-content">
          <div class="certificate-copy">
            <h2 class="certificate-type"><?= View::e($certificate['title']) ?></h2>
            <p class="certificate-recipient-label">证书获得者</p>
            <p class="certificate-recipient-name"><?= View::e($certificate['recipient_name']) ?></p>

            <div class="certificate-achievement-row">
              <div class="certificate-score-block"><span>综合得分</span><strong><?= View::e($certificate['score']) ?> <small>/ 100</small></strong></div>
              <?php if ($certificate['award']['title'] !== ''): ?><div class="certificate-tier-block"><span>专业称号</span><strong><?= View::e($certificate['award']['title']) ?></strong></div><?php endif; ?>
            </div>

            <p class="certificate-encouragement"><?= View::e($encouragementBefore) ?><?php if ($highlightPosition !== false): ?><strong><?= View::e($highlight) ?></strong><?= View::e($encouragementAfter) ?><?php endif; ?></p>
          </div>

          <figure class="certificate-radar-figure">
            <div class="certificate-radar-wrap">
              <canvas data-certificate-radar width="880" height="880" role="img" aria-label="六维能力雷达图，<?= View::e($dimensionSummary) ?>"></canvas>
              <div class="certificate-radar-score"><strong><?= View::e($certificate['score']) ?></strong><span>综合得分</span></div>
            </div>
            <figcaption>六维专业能力画像</figcaption>
          </figure>
        </div>

        <footer class="certificate-footer">
          <div class="certificate-issue-grid">
            <div><span>签发日期</span><strong><?= View::e($certificate['issued_on']) ?></strong></div>
            <div><span>证书编号</span><strong><?= View::e($certificate['number']) ?></strong></div>
            <div><span>签发方</span><strong><?= View::e($certificate['issuer']) ?></strong></div>
          </div>
          <div class="certificate-verification">
            <div><span>扫描二维码查询证书</span><strong>VERIFY CERTIFICATE</strong></div>
            <img src="<?= View::e($certificate['qr_data_uri']) ?>" width="350" height="350" alt="证书查询二维码">
          </div>
        </footer>
      </article>
    </div>
  </section>

  <canvas class="certificate-export-canvas" data-certificate-export width="1680" height="1188" aria-hidden="true"></canvas>
  <div class="certificate-toast" role="status" aria-live="polite" data-certificate-toast>证书图片已生成</div>
</main>
