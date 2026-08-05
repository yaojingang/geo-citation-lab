<?php

declare(strict_types=1);

use GeoAssessment\Support\View;

$attemptStatusLabels = [
    'in_progress' => '进行中',
    'timed_out' => '已超时',
];
?>
<main id="main-content" class="home-main">
  <?php if ($user === null): ?>
    <section class="home-intro" aria-labelledby="home-title">
      <p class="eyebrow">GEO RESEARCH ASSESSMENT · 2026</p>
      <h1 id="home-title">理解 AI 如何选择来源、<br>吸收内容与呈现实体。</h1>
      <p class="home-lede">这套测试依据 54 篇 GEO、AI 搜索与 RAG 论文，结合海外平台实验和 214,119 条国内引用记录设计。</p>

      <form class="identity-form" action="<?= View::e($view->url('/identity')) ?>" method="post" novalidate data-validate-name>
        <input type="hidden" name="_csrf" value="<?= View::e($csrfToken) ?>">
        <label for="name">输入姓名即可开始</label>
        <div class="identity-row">
          <input id="name" name="name" type="text" minlength="2" maxlength="40" autocomplete="name" required placeholder="例如：陈星河" aria-describedby="name-help<?= $error ? ' name-error' : '' ?>">
          <button class="button button-primary" type="submit">开始测试</button>
        </div>
        <p id="name-help" class="field-help">姓名只用于当前系统中的记录与报告显示。</p>
        <?php if ($error): ?><p id="name-error" class="field-error" role="alert"><?= View::e($error) ?></p><?php endif; ?>
      </form>

      <div class="test-facts" aria-label="测试信息">
        <div><strong>30</strong><span>道题</span></div>
        <div><strong>30:00</strong><span>时间上限</span></div>
        <div><strong>100</strong><span>总分</span></div>
        <div><strong>10</strong><span>次机会</span></div>
      </div>
    </section>
  <?php else: ?>
    <section class="dashboard-head" aria-labelledby="welcome-title">
      <div>
        <p class="eyebrow">测试者</p>
        <h1 id="welcome-title"><?= View::e($user['display_name']) ?>，欢迎回来</h1>
        <p class="dashboard-lede">你还有 <strong><?= View::e($remaining) ?></strong> 次测试机会。所有成绩和报告保存在本站 SQLite 数据库中。</p>
      </div>
      <div class="dashboard-action">
        <?php if ($current): ?>
          <a class="button button-primary" href="<?= View::e($view->url('/attempts/' . $current['id'] . '/questions/' . $currentPosition)) ?>">继续第 <?= View::e($current['attempt_no']) ?> 次测试</a>
        <?php elseif ($remaining > 0): ?>
          <form action="<?= View::e($view->url('/attempts')) ?>" method="post">
            <input type="hidden" name="_csrf" value="<?= View::e($csrfToken) ?>">
            <button class="button button-primary" type="submit">开始第 <?= View::e(count($history) + 1) ?> 次测试</button>
          </form>
        <?php else: ?>
          <p class="limit-note">已完成 10 次测试，历史报告仍可随时查看。</p>
        <?php endif; ?>
      </div>
    </section>

    <?php if ($error): ?><div class="notice notice-error" role="alert"><?= View::e($error) ?></div><?php endif; ?>

    <section class="history-section" aria-labelledby="history-title">
      <div class="section-heading-row"><h2 id="history-title">测试记录</h2><span><?= View::e(count($history)) ?> / 10</span></div>
      <?php if ($history === []): ?>
        <p class="empty-state">完成第一次交卷后，这里会显示分数、用时和详细报告。</p>
      <?php else: ?>
        <div class="history-table-wrap">
          <table class="history-table">
            <thead><tr><th>次数</th><th>日期</th><th>状态</th><th class="number">分数</th><th class="number">正确</th><th class="number">用时</th><th>题集</th><th><span class="sr-only">操作</span></th></tr></thead>
            <tbody>
            <?php foreach ($history as $entry): ?>
              <tr>
                <td data-label="次数">第 <?= View::e($entry['attempt_no']) ?> 次</td>
                <td data-label="日期"><?= View::e($entry['submitted_at'] ? date('Y.m.d', strtotime($entry['submitted_at'])) : date('Y.m.d', strtotime($entry['started_at']))) ?></td>
                <td data-label="状态"><span class="status status-<?= View::e($entry['status']) ?>"><?= View::e($attemptStatusLabels[$entry['status']] ?? '已交卷') ?></span></td>
                <td data-label="分数" class="number score-cell"><?= $entry['score'] === null ? '待生成' : View::e($entry['score']) ?></td>
                <td data-label="正确" class="number"><?= $entry['correct_count'] === null ? '待生成' : View::e($entry['correct_count'] . '/30') ?></td>
                <td data-label="用时" class="number"><?= $entry['duration_seconds'] === null ? '进行中' : View::duration((int) $entry['duration_seconds']) ?></td>
                <td data-label="题集"><?= View::e($entry['set_version']) ?></td>
                <td class="row-action">
                  <?php if ($entry['status'] === 'in_progress'): ?><a href="<?= View::e($view->url('/attempts/' . $entry['id'] . '/questions/' . $currentPosition)) ?>">继续</a><?php else: ?><a href="<?= View::e($view->url('/reports/' . $entry['id'])) ?>">报告</a><?php endif; ?>
                </td>
              </tr>
            <?php endforeach; ?>
            </tbody>
          </table>
        </div>
      <?php endif; ?>
    </section>

    <section class="account-section no-print" aria-labelledby="account-title">
      <h2 id="account-title">当前测试者</h2>
      <div class="account-actions">
        <details class="switch-disclosure">
          <summary>切换测试者</summary>
          <form action="<?= View::e($view->url('/switch-user')) ?>" method="post">
            <input type="hidden" name="_csrf" value="<?= View::e($csrfToken) ?>">
            <label class="confirmation-check"><input type="checkbox" name="confirm_switch" value="1" required><span>我已了解：切换后，当前浏览器将无法再打开「<?= View::e($user['display_name']) ?>」的历史报告。</span></label>
            <button class="button button-secondary" type="submit">确认切换</button>
          </form>
        </details>
        <details class="delete-disclosure">
          <summary>删除本人记录</summary>
          <form action="<?= View::e($view->url('/me/delete')) ?>" method="post">
            <input type="hidden" name="_csrf" value="<?= View::e($csrfToken) ?>">
            <label for="confirmation-name">输入「<?= View::e($user['display_name']) ?>」确认删除</label>
            <div class="delete-row"><input id="confirmation-name" name="confirmation_name" required autocomplete="off"><button class="button button-danger" type="submit">删除记录</button></div>
          </form>
        </details>
      </div>
    </section>
  <?php endif; ?>

  <section id="privacy" class="boundary-section">
    <div><p class="eyebrow">PRIVACY</p><h2>测试记录保存在本站 SQLite 数据库</h2></div>
    <p>应用保存姓名、作答、用时与报告。<?php if ($view->analyticsEnabled()): ?>当前部署已启用百度统计，用于汇总访问情况，百度可能接收浏览器、设备与访问页面等信息。<?php endif; ?>浏览器随机令牌承担登录凭证，姓名无法用于跨设备找回。删除本人记录会级联删除作答与报告。</p>
  </section>
  <section id="terms" class="boundary-section">
    <div><p class="eyebrow">RULES</p><h2>题集与评分规则</h2></div>
    <p>测试限时 30 分钟，单选与多选都采用正确集合完整匹配。每个浏览器身份终身最多 10 次，清除站点数据或更换设备会建立新身份。</p>
  </section>
</main>
