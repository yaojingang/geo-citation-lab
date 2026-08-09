<?php

declare(strict_types=1);

use GeoAssessment\Support\View;

$snapshot = $item['snapshot'];
$selectedCodes = $item['selected_codes'];
$position = (int) $item['position'];
?>
<main id="main-content" class="quiz-main" data-quiz data-deadline="<?= View::e($attempt['deadline_at']) ?>" data-remaining-seconds="<?= View::e(max(0, strtotime((string) $attempt['deadline_at']) - time())) ?>" data-attempt-id="<?= View::e($attempt['id']) ?>" data-position="<?= View::e($position) ?>" data-answered="<?= View::e($answered) ?>" data-current-answered="<?= $selectedCodes === [] ? '0' : '1' ?>" data-csrf="<?= View::e($csrfToken) ?>">
  <div class="quiz-topline" aria-label="测试进度与时间">
    <div class="progress-copy"><strong><?= View::e($position) ?></strong><span>/ 30</span><span class="progress-answered"><?= View::e($answered) ?> 已答</span></div>
    <progress class="progress-track" value="<?= View::e($position) ?>" max="30" aria-label="当前位于第 <?= View::e($position) ?> 题，共 30 题"><?= View::e($position) ?> / 30</progress>
    <div class="quiz-runtime"><span class="save-state" data-save-state aria-live="polite">已保存</span><time class="timer" data-timer datetime="PT30M">30:00</time></div>
  </div>

  <div class="quiz-layout">
    <section class="question-panel" aria-labelledby="question-title">
      <div class="question-meta">
        <span><?= View::e($snapshot['code']) ?></span><span><?= View::e($snapshot['dimension_label']) ?></span><span><?= View::e($snapshot['type'] === 'multiple' ? '多选' : '单选') ?></span><span><?= View::e($snapshot['weight']) ?> 分</span>
      </div>
      <h1 id="question-title"><?= View::e($snapshot['prompt']) ?></h1>
      <?php if ($snapshot['type'] === 'multiple'): ?><p class="question-instruction">可多选，完整选择所有正确项才得分</p><?php else: ?><p class="question-instruction">请选择一个最准确的答案</p><?php endif; ?>

      <form id="question-form" class="question-form" action="<?= View::e($view->url('/attempts/' . $attempt['id'] . '/answers')) ?>" method="post" data-question-form>
        <input type="hidden" name="_csrf" value="<?= View::e($csrfToken) ?>">
        <input type="hidden" name="question_code" value="<?= View::e($snapshot['code']) ?>">
        <input type="hidden" name="activity_seq" value="<?= View::e($item['activity_seq'] + 1) ?>" data-activity-seq>
        <input type="hidden" name="active_seconds_delta" value="0" data-active-seconds>
        <fieldset class="choice-list">
          <legend class="sr-only"><?= View::e($snapshot['prompt']) ?></legend>
          <?php foreach ($snapshot['choices'] as $index => $choice): ?>
            <label class="choice-option">
              <input type="<?= $snapshot['type'] === 'multiple' ? 'checkbox' : 'radio' ?>" name="selected_codes[]" value="<?= View::e($choice['code']) ?>" <?= in_array($choice['code'], $selectedCodes, true) ? 'checked' : '' ?>>
              <span class="choice-key" aria-hidden="true"><?= View::e($index + 1) ?></span>
              <span class="choice-code"><?= View::e($choice['code']) ?></span>
              <span class="choice-text"><?= View::e($choice['text']) ?></span>
              <span class="choice-check" aria-hidden="true"></span>
            </label>
          <?php endforeach; ?>
        </fieldset>

        <div class="quiz-actions">
          <?php if ($position > 1): ?><button class="button button-secondary" type="submit" name="navigate_to" value="<?= View::e($position - 1) ?>">上一题</button><?php else: ?><span></span><?php endif; ?>
          <button class="button button-quiet quiz-grid-trigger" type="button" data-nav-toggle aria-expanded="false" aria-controls="mobile-question-nav">题号</button>
          <?php if ($position < 30): ?>
            <button class="button button-primary" type="submit" name="navigate_to" value="<?= View::e($position + 1) ?>">下一题</button>
          <?php else: ?>
            <button class="button button-primary" type="submit" formaction="<?= View::e($view->url('/attempts/' . $attempt['id'] . '/submit')) ?>" data-submit-test aria-haspopup="dialog" aria-controls="submit-confirmation-dialog" aria-expanded="false">交卷并查看报告</button>
          <?php endif; ?>
        </div>
      </form>
    </section>

    <aside class="question-nav" aria-label="30 道题导航">
      <div class="question-nav-head"><strong>答题进度</strong><span><?= View::e($answered) ?> / 30</span></div>
      <nav class="question-grid">
        <?php foreach ($items as $navItem): ?><button type="submit" form="question-form" name="navigate_to" value="<?= View::e($navItem['position']) ?>" class="question-number <?= $navItem['position'] === $position ? 'is-current' : ($navItem['selected_codes'] !== [] ? 'is-answered' : '') ?>" <?= $navItem['position'] === $position ? 'aria-current="step"' : '' ?>><?= View::e($navItem['position']) ?><span class="sr-only"><?= $navItem['selected_codes'] !== [] ? '已答' : '未答' ?></span></button><?php endforeach; ?>
      </nav>
      <div class="nav-legend"><span><i class="legend-current"></i>当前</span><span><i class="legend-answered"></i>已答</span><span><i></i>未答</span></div>
    </aside>
  </div>

  <div class="mobile-nav-drawer" id="mobile-question-nav" data-mobile-nav hidden>
    <div class="mobile-nav-head"><strong>选择题号</strong><button type="button" class="icon-button" data-nav-close aria-label="关闭题号导航">×</button></div>
    <nav class="question-grid" aria-label="移动端题号导航">
      <?php foreach ($items as $navItem): ?><button type="submit" form="question-form" name="navigate_to" value="<?= View::e($navItem['position']) ?>" class="question-number <?= $navItem['position'] === $position ? 'is-current' : ($navItem['selected_codes'] !== [] ? 'is-answered' : '') ?>"><?= View::e($navItem['position']) ?></button><?php endforeach; ?>
    </nav>
  </div>

  <div class="submit-dialog" id="submit-confirmation-dialog" data-submit-dialog hidden>
    <button class="submit-dialog-backdrop" type="button" data-submit-dialog-cancel tabindex="-1" aria-label="关闭交卷确认"></button>
    <section class="submit-dialog-panel" role="dialog" aria-modal="true" aria-labelledby="submit-dialog-title" aria-describedby="submit-dialog-description">
      <div class="submit-dialog-mark" aria-hidden="true">✓</div>
      <div class="submit-dialog-copy">
        <p class="submit-dialog-eyebrow">完成测试</p>
        <h2 id="submit-dialog-title">确认交卷</h2>
        <p id="submit-dialog-description">本次已完成 <strong data-submit-answered>0</strong> 题，仍有 <strong data-submit-unanswered>30</strong> 题未答</p>
        <p class="submit-dialog-note">交卷后将结束本次测试，并生成完整的 GEO 能力报告</p>
      </div>
      <div class="submit-dialog-actions">
        <button class="button button-secondary" type="button" data-submit-dialog-cancel>继续答题</button>
        <button class="button button-primary" type="button" data-submit-dialog-confirm>确认交卷</button>
      </div>
    </section>
  </div>
</main>
