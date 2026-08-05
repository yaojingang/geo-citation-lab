<?php

declare(strict_types=1);

use GeoAssessment\Support\View;

$summary = $report['summary'];
$attempt = $report['attempt'];
$learningSectionNumber = $report['cohort']['visible'] ? '07' : '06';
$reviewSectionNumber = $report['cohort']['visible'] ? '08' : '07';
$reportCopy = static function ($value): string {
    return View::e(View::trimTerminalPeriod($value));
};
$reportStatusLabel = static function (string $status): string {
    if ($status === 'correct') {
        return '正确';
    }
    if ($status === 'incorrect') {
        return '错误';
    }
    return '未答';
};
?>
<main id="main-content" class="report-main" data-report>
  <div id="report-data" hidden data-report-json="<?= View::jsonAttribute($report['charts']) ?>"></div>

  <header class="report-header">
    <div>
      <p class="eyebrow">GEO CAPABILITY REPORT · <?= View::e($attempt['set_version']) ?></p>
      <h1><span class="personal-report-title">GEO 能力报告</span><span class="anonymous-report-title">GEO 测试报告</span></h1>
      <p class="report-meta"><span class="print-name"><?= View::e($report['user']['display_name']) ?> · </span>第 <?= View::e($summary['attempt_no']) ?> 次测试 · <?= View::e(date('Y.m.d H:i', strtotime($attempt['submitted_at']))) ?></p>
    </div>
    <div class="report-actions no-print">
      <a class="button button-secondary" href="<?= View::e($view->url('/')) ?>">回到首页</a>
      <label class="print-name-toggle"><input type="checkbox" checked data-print-name>打印姓名</label>
      <button class="button button-primary" type="button" data-print-report>打印报告</button>
    </div>
  </header>

  <section class="report-overview" aria-labelledby="overview-title">
    <div class="score-hero">
      <div class="score-ring-wrap">
        <canvas data-chart="score-ring" width="260" height="260" role="img" aria-label="总分 <?= View::e($summary['score']) ?> 分"></canvas>
        <div class="score-ring-number"><strong><?= View::e($summary['score']) ?></strong><span>/ 100</span></div>
      </div>
      <div class="score-interpretation">
        <p class="stage-label"><?= View::e($summary['stage']['label']) ?></p>
        <h2 id="overview-title"><?= $reportCopy($summary['stage']['summary']) ?></h2>
        <p>本次答对 <?= View::e($summary['correct_count']) ?> 道，完成 <?= View::e($summary['answered_count']) ?> 道，总用时 <?= View::duration($summary['duration_seconds']) ?></p>
      </div>
    </div>
    <dl class="overview-metrics">
      <div><dt>总分</dt><dd><?= View::e($summary['score']) ?></dd></div>
      <div><dt>正确题</dt><dd><?= View::e($summary['correct_count']) ?><small>/30</small></dd></div>
      <div><dt>已作答</dt><dd><?= View::e($summary['answered_count']) ?><small>/30</small></dd></div>
      <div><dt>总用时</dt><dd><?= View::duration($summary['duration_seconds']) ?></dd></div>
    </dl>
  </section>

  <section class="insight-section" aria-labelledby="insight-title">
    <div class="section-kicker"><span>01</span><p>诊断结论</p></div>
    <h2 id="insight-title"><?= $reportCopy($report['insights']['headline']) ?></h2>
    <div class="insight-grid">
      <article class="insight-strength">
        <p class="insight-label">强项</p>
        <h3><?= $reportCopy($report['insights']['strength']['title']) ?></h3>
        <p><?= $reportCopy($report['insights']['strength']['text']) ?></p>
      </article>
      <?php foreach ($report['insights']['recommendations'] as $index => $recommendation): ?>
        <article>
          <p class="insight-label">优先项 <?= View::e($index + 1) ?></p>
          <h3><?= $reportCopy($recommendation['title']) ?></h3>
          <p><?= $reportCopy($recommendation['text']) ?></p>
          <?php if ($recommendation['question_codes'] !== []): ?><p class="linked-questions">关联题目：<?php foreach ($recommendation['question_codes'] as $code): ?><a href="#q<?= View::e(substr($code, 1)) ?>"><?= View::e($code) ?></a><?php endforeach; ?></p><?php endif; ?>
        </article>
      <?php endforeach; ?>
    </div>
  </section>

  <section class="dimension-section" aria-labelledby="dimension-title">
    <div class="section-kicker"><span>02</span><p>六维能力</p></div>
    <h2 id="dimension-title">六个维度展示你的知识结构</h2>
    <div class="chart-grid chart-grid-two">
      <figure class="chart-panel">
        <figcaption><strong>六维轮廓</strong><span>得分率越接近外圈，当前掌握越系统</span></figcaption>
        <div class="chart-canvas chart-canvas-square"><canvas data-chart="dimension-radar" width="520" height="420" role="img" aria-label="六维得分率雷达图"></canvas></div>
      </figure>
      <div class="dimension-bars" data-chart="dimension-bars-html">
        <?php foreach ($report['dimensions'] as $dimension): ?>
          <a class="dimension-row" href="#q<?= View::e(substr($dimension['question_codes'][0], 1)) ?>">
            <span class="dimension-row-head"><strong><?= View::e($dimension['label']) ?></strong><em><?= View::e($dimension['earned']) ?>/<?= View::e($dimension['possible']) ?> · <?= View::e($dimension['label'] ? $dimension['percentage'] : 0) ?>%</em></span>
            <progress class="dimension-progress" value="<?= View::e($dimension['percentage']) ?>" max="100" aria-label="<?= View::e($dimension['label']) ?>得分率 <?= View::e($dimension['percentage']) ?>%"><?= View::e($dimension['percentage']) ?>%</progress>
            <span class="dimension-level"><?= View::e($dimension['mastery_label']) ?></span>
          </a>
        <?php endforeach; ?>
      </div>
    </div>
    <details class="data-fallback"><summary>查看六维数据表</summary>
      <table><thead><tr><th>维度</th><th class="number">得分</th><th class="number">得分率</th><th>掌握标签</th></tr></thead><tbody><?php foreach ($report['dimensions'] as $dimension): ?><tr><td><?= View::e($dimension['label']) ?></td><td class="number"><?= View::e($dimension['earned']) ?>/<?= View::e($dimension['possible']) ?></td><td class="number"><?= View::e($dimension['percentage']) ?>%</td><td><?= View::e($dimension['mastery_label']) ?></td></tr><?php endforeach; ?></tbody></table>
    </details>
  </section>

  <section class="performance-section" aria-labelledby="performance-title">
    <div class="section-kicker"><span>03</span><p>题目表现</p></div>
    <h2 id="performance-title">难度与用时帮助区分知识缺口和时间策略</h2>
    <div class="diagnostic-strip">
      <?php foreach ($report['types'] as $type): ?>
        <?php $typePercentage = $type['total'] > 0 ? round(($type['correct'] / $type['total']) * 100, 1) : 0; ?>
        <article><p><?= View::e($type['label']) ?>题</p><strong><?= View::e($type['correct']) ?><small>/<?= View::e($type['total']) ?></small></strong><progress value="<?= View::e($typePercentage) ?>" max="100" aria-label="<?= View::e($type['label']) ?>题正确率 <?= View::e($typePercentage) ?>%"><?= View::e($typePercentage) ?>%</progress></article>
      <?php endforeach; ?>
      <article class="time-strategy"><p>时间策略</p><strong><?= $reportCopy($report['insights']['time_strategy']['title']) ?></strong><span><?= $reportCopy($report['insights']['time_strategy']['text']) ?></span><?php if ($report['insights']['time_strategy']['question_codes'] !== []): ?><span class="linked-questions"><?php foreach ($report['insights']['time_strategy']['question_codes'] as $code): ?><a href="#q<?= View::e(substr($code, 1)) ?>"><?= View::e($code) ?></a><?php endforeach; ?></span><?php endif; ?></article>
    </div>
    <div class="chart-grid chart-grid-two">
      <figure class="chart-panel">
        <figcaption><strong>难度表现</strong><span>正确、错误与未答按基础、进阶和挑战分层</span></figcaption>
        <div class="chart-canvas"><canvas data-chart="difficulty-stack" width="640" height="360" role="img" aria-label="三个难度层级的正确、错误与未答数量"></canvas></div>
      </figure>
      <figure class="chart-panel">
        <figcaption><strong>逐题用时</strong><span>用时只用于诊断，不参与得分</span></figcaption>
        <div class="chart-canvas"><canvas data-chart="question-time" width="640" height="360" role="img" aria-label="30 道题的活跃作答时间"></canvas></div>
      </figure>
    </div>
    <details class="data-fallback"><summary>查看难度与用时表</summary>
      <div class="fallback-two"><table><thead><tr><th>难度</th><th>正确</th><th>错误</th><th>未答</th></tr></thead><tbody><?php foreach ($report['difficulties'] as $difficulty): ?><tr><td><?= View::e($difficulty['label']) ?></td><td><?= View::e($difficulty['correct']) ?></td><td><?= View::e($difficulty['incorrect']) ?></td><td><?= View::e($difficulty['unanswered']) ?></td></tr><?php endforeach; ?></tbody></table>
      <table><thead><tr><th>题号</th><th>用时</th></tr></thead><tbody><?php foreach ($report['questions'] as $question): ?><tr><td><a href="#q<?= View::e(substr($question['code'], 1)) ?>"><?= View::e($question['code']) ?></a></td><td><?= View::e($question['time_spent_seconds']) ?> 秒</td></tr><?php endforeach; ?></tbody></table></div>
    </details>
  </section>

  <section class="matrix-section" aria-labelledby="matrix-title">
    <div class="section-kicker"><span>04</span><p>30 题状态</p></div>
    <div class="section-heading-inline"><h2 id="matrix-title">点击任意题号查看完整证据链</h2><div class="matrix-tools"><div class="matrix-legend"><span><i class="is-correct"></i>正确</span><span><i class="is-incorrect"></i>错误</span><span><i class="is-unanswered"></i>未答</span></div><div class="matrix-filters no-print" aria-label="筛选题目状态"><button type="button" class="is-active" data-matrix-filter="all" aria-pressed="true">全部</button><button type="button" data-matrix-filter="incorrect" aria-pressed="false">错误</button><button type="button" data-matrix-filter="correct" aria-pressed="false">正确</button><button type="button" data-matrix-filter="unanswered" aria-pressed="false">未答</button></div></div></div>
    <nav class="report-matrix" aria-label="30 道题结果矩阵">
      <?php foreach ($report['matrix'] as $cell): ?><a class="matrix-cell is-<?= View::e($cell['status']) ?>" data-status="<?= View::e($cell['status']) ?>" href="#q<?= View::e(substr($cell['code'], 1)) ?>" aria-label="<?= View::e($cell['code'] . ' ' . View::trimTerminalPeriod($cell['title']) . ' ' . $reportStatusLabel($cell['status'])) ?>"><strong><?= View::e(substr($cell['code'], 1)) ?></strong><span><?= View::e($cell['points']) ?>/<?= View::e($cell['possible_points']) ?></span></a><?php endforeach; ?>
    </nav>
    <details class="data-fallback"><summary>查看 30 题状态表</summary>
      <table><thead><tr><th>题号</th><th>题目</th><th>状态</th><th class="number">得分</th></tr></thead><tbody><?php foreach ($report['matrix'] as $cell): ?><?php $cellStatusLabel = $reportStatusLabel($cell['status']); ?><tr><td><a href="#q<?= View::e(substr($cell['code'], 1)) ?>"><?= View::e($cell['code']) ?></a></td><td><a href="#q<?= View::e(substr($cell['code'], 1)) ?>"><?= $reportCopy($cell['title']) ?></a></td><td><?= View::e($cellStatusLabel) ?></td><td class="number"><?= View::e($cell['points']) ?>/<?= View::e($cell['possible_points']) ?></td></tr><?php endforeach; ?></tbody></table>
    </details>
  </section>

  <section class="trend-section" aria-labelledby="trend-title">
    <div class="section-kicker"><span>05</span><p>历次趋势</p></div>
    <h2 id="trend-title">同版本趋势可直接对比，版本变化会保留独立语境</h2>
    <?php if ($report['charts']['trend_ready']): ?>
      <div class="chart-grid chart-grid-two">
        <figure class="chart-panel"><figcaption><strong>总分趋势</strong><span>显示当前题集最近 5 次记录</span></figcaption><div class="chart-canvas"><canvas data-chart="score-trend" width="640" height="360" role="img" aria-label="历次总分趋势"></canvas></div></figure>
        <figure class="chart-panel"><figcaption><strong>六维变化</strong><span>每条折线对应一个知识维度</span></figcaption><div class="chart-canvas"><canvas data-chart="dimension-trend" width="640" height="360" role="img" aria-label="历次六维得分率变化"></canvas></div></figure>
      </div>
    <?php else: ?>
      <p class="trend-empty">完成第 2 次同版本测试后，这里会显示总分和六维变化。当前已有 <?= View::e($report['charts']['trend_count']) ?> 条可比记录</p>
    <?php endif; ?>
    <details class="data-fallback"><summary>查看历次数据表</summary><table><thead><tr><th>次数</th><th>题集</th><th>总分</th><th>正确</th><th>用时</th><th>报告</th></tr></thead><tbody><?php foreach ($report['history'] as $entry): ?><tr><td>第 <?= View::e($entry['attempt_no']) ?> 次</td><td><?= View::e($entry['set_version']) ?></td><td><?= View::e($entry['score']) ?></td><td><?= View::e($entry['correct_count']) ?>/30</td><td><?= View::duration($entry['duration_seconds']) ?></td><td><a href="<?= View::e($view->url('/reports/' . $entry['id'])) ?>">打开</a></td></tr><?php endforeach; ?></tbody></table></details>
  </section>

  <?php if ($report['cohort']['visible']): ?>
  <section class="cohort-section" aria-labelledby="cohort-title"><div class="section-kicker"><span>06</span><p>同版本群体</p></div><h2 id="cohort-title"><?= $reportCopy($report['cohort']['label']) ?></h2><p>当前得分位于约 <?= View::e($report['cohort']['percentile']) ?> 百分位，样本量 <?= View::e($report['cohort']['sample_size']) ?>。每位测试者只取最近一次同版本成绩</p></section>
  <?php endif; ?>

  <section class="learning-section" aria-labelledby="learning-title">
    <div class="section-kicker"><span><?= View::e($learningSectionNumber) ?></span><p>学习路径</p></div>
    <h2 id="learning-title">三步完成从错题到下一次测试的转化</h2>
    <ol class="learning-path"><?php foreach ($report['insights']['learning_path'] as $step): ?><li><span><?= View::e($step['step']) ?></span><div><h3><?= $reportCopy($step['title']) ?></h3><p><?= $reportCopy($step['text']) ?></p></div></li><?php endforeach; ?></ol>
  </section>

  <section class="question-review" aria-labelledby="review-title">
    <div class="section-kicker"><span><?= View::e($reviewSectionNumber) ?></span><p>逐题证据</p></div>
    <h2 id="review-title">30 道题的选择、答案、解析与来源</h2>
    <p class="review-lede">屏幕上可按需展开，打印版会自动展示全部详情</p>

    <div class="question-detail-list">
      <?php foreach ($report['questions'] as $question): ?>
        <details id="q<?= View::e(substr($question['code'], 1)) ?>" class="question-detail is-<?= View::e($question['status']) ?>" open>
          <summary>
            <span class="detail-number"><?= View::e(substr($question['code'], 1)) ?></span>
            <span class="detail-title"><small><?= View::e($question['dimension_label']) ?> · <?= View::e($question['difficulty_label']) ?> · <?= View::e($question['type'] === 'multiple' ? '多选' : '单选') ?></small><strong><?= $reportCopy($question['prompt']) ?></strong></span>
            <span class="detail-score"><?= View::e($question['points']) ?>/<?= View::e($question['possible_points']) ?></span>
          </summary>
          <div class="question-detail-body">
            <div class="detail-outcome">
              <p><span>本人选择</span><strong><?= $question['selected_codes'] === [] ? '未作答' : View::e(implode('、', $question['selected_codes'])) ?></strong></p>
              <p><span>正确答案</span><strong><?= View::e(implode('、', $question['correct_codes'])) ?></strong></p>
              <p><span>得分</span><strong><?= View::e($question['points']) ?>/<?= View::e($question['possible_points']) ?></strong></p>
              <p><span>作答过程</span><strong><?= View::e($question['time_spent_seconds']) ?> 秒 · 改选 <?= View::e($question['change_count']) ?> 次</strong></p>
            </div>
            <div class="review-choice-list">
              <?php foreach ($question['choices'] as $choice): ?>
                <?php $isSelected = in_array($choice['code'], $question['selected_codes'], true); $isCorrect = in_array($choice['code'], $question['correct_codes'], true); ?>
                <div class="review-choice <?= $isSelected ? 'is-selected ' : '' ?><?= $isCorrect ? 'is-correct-choice' : 'is-wrong-choice' ?>">
                  <div class="review-choice-main"><strong><?= View::e($choice['code']) ?></strong><span><?= $reportCopy($choice['text']) ?></span><em><?= $isSelected ? '已选' : '' ?><?= $isSelected && $isCorrect ? ' · ' : '' ?><?= $isCorrect ? '正确项' : '' ?></em></div>
                  <p><?= $reportCopy($choice['rationale']) ?></p>
                </div>
              <?php endforeach; ?>
            </div>
            <div class="explanation-grid">
              <article><p class="detail-label">答案解析</p><p><?= $reportCopy($question['explanation']) ?></p></article>
              <article><p class="detail-label">底层原理</p><p><?= $reportCopy($question['principle']) ?></p></article>
            </div>
            <?php if (!$question['is_correct']): ?><p class="misconception"><span>主要误区</span><?= $reportCopy($question['misconception_tag']) ?><?php if ($question['label'] !== '选择集合完全匹配'): ?> · <?= $reportCopy($question['label']) ?><?php endif; ?></p><?php endif; ?>
            <div class="sources"><p class="detail-label">证据来源</p><ol><?php foreach ($question['sources'] as $source): ?><li><span><?= View::e($source['id']) ?></span><?php if ($source['url']): ?><a href="<?= View::e($source['url']) ?>" target="_blank" rel="noopener nofollow"><?= $reportCopy($source['title']) ?></a><?php else: ?><strong><?= $reportCopy($source['title']) ?></strong><?php endif; ?><small><?= View::e($source['citation']) ?></small></li><?php endforeach; ?></ol></div>
          </div>
        </details>
      <?php endforeach; ?>
    </div>
  </section>

  <footer class="report-colophon">
    <p>GEO Assessment · <?= View::e($attempt['set_version']) ?> · 评分 <?= View::e($attempt['scoring_version']) ?></p>
    <p>证据冻结日 <?= View::e($attempt['evidence_frozen_at']) ?> · 用时为近似活跃时间</p>
  </footer>
</main>
