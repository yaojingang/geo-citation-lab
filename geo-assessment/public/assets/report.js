(() => {
  'use strict';

  const report = document.querySelector('[data-report]');
  const dataNode = document.querySelector('#report-data');
  if (!report || !dataNode) return;

  const charts = JSON.parse(dataNode.dataset.reportJson || '{}');
  let canvases = [];
  let render = () => {};
  const details = [...report.querySelectorAll('.question-detail')];
  const fallbackDetails = [...report.querySelectorAll('.data-fallback')];
  const matrixCells = [...report.querySelectorAll('.matrix-cell')];
  const matrixFilters = [...report.querySelectorAll('[data-matrix-filter]')];
  let activeMatrixFilter = 'all';
  let printMatrixFilter = 'all';
  const applyMatrixFilter = (status, shouldScroll = true) => {
    activeMatrixFilter = status;
    matrixCells.forEach((cell) => { cell.hidden = status !== 'all' && cell.dataset.status !== status; });
    matrixFilters.forEach((button) => {
      const active = button.dataset.matrixFilter === status;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    if (shouldScroll) document.querySelector('.matrix-section')?.scrollIntoView({ block: 'start' });
  };
  matrixFilters.forEach((button) => button.addEventListener('click', () => applyMatrixFilter(button.dataset.matrixFilter || 'all')));
  const hashTarget = () => {
    if (!/^#q\d{2}$/.test(location.hash)) return null;
    return document.getElementById(location.hash.slice(1));
  };
  const target = hashTarget();
  details.forEach((detail) => { detail.open = detail === target; });
  const revealTarget = (detail) => {
    detail.open = true;
    window.requestAnimationFrame(() => {
      detail.scrollIntoView({ block: 'start' });
      detail.querySelector('summary')?.focus({ preventScroll: true });
    });
  };
  if (target) revealTarget(target);
  window.addEventListener('hashchange', () => {
    const nextTarget = hashTarget();
    if (nextTarget) revealTarget(nextTarget);
  });

  document.body.classList.add('show-print-name');
  const printName = report.querySelector('[data-print-name]');
  printName?.addEventListener('change', () => {
    document.body.classList.toggle('hide-print-name', !printName.checked);
  });

  let printState = [];
  let fallbackPrintState = [];
  window.addEventListener('beforeprint', () => {
    canvases.forEach((canvas) => render(canvas));
    printState = details.map((detail) => detail.open);
    fallbackPrintState = fallbackDetails.map((detail) => detail.open);
    details.forEach((detail) => { detail.open = true; });
    fallbackDetails.forEach((detail) => { detail.open = true; });
    printMatrixFilter = activeMatrixFilter;
    applyMatrixFilter('all', false);
  });
  window.addEventListener('afterprint', () => {
    details.forEach((detail, index) => { detail.open = printState[index] ?? false; });
    fallbackDetails.forEach((detail, index) => { detail.open = fallbackPrintState[index] ?? false; });
    applyMatrixFilter(printMatrixFilter, false);
  });
  report.querySelector('[data-print-report]')?.addEventListener('click', () => {
    canvases.forEach((canvas) => render(canvas));
    window.requestAnimationFrame(() => window.print());
  });

  if (typeof window.Chart !== 'function') {
    report.querySelectorAll('.data-fallback').forEach((fallback) => { fallback.open = true; });
    return;
  }

  const Chart = window.Chart;
  Chart.defaults.color = '#6e6e73';
  Chart.defaults.borderColor = '#e8e8ed';
  Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "PingFang SC", "Noto Sans SC", sans-serif';
  Chart.defaults.font.size = 11;
  Chart.defaults.animation.duration = matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 240;

  const palette = ['#0066cc', '#16794b', '#946200', '#7c3aed', '#c2410c', '#1b365d'];

  const positiveTimes = (charts.question_time.values || []).filter((value) => Number(value) > 0).map(Number).sort((left, right) => left - right);
  const timeMiddle = Math.floor(positiveTimes.length / 2);
  const timeMedian = positiveTimes.length === 0 ? 0 : (positiveTimes.length % 2 ? positiveTimes[timeMiddle] : (positiveTimes[timeMiddle - 1] + positiveTimes[timeMiddle]) / 2);
  const common = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { intersect: false, mode: 'nearest' },
    plugins: {
      legend: { labels: { usePointStyle: true, boxWidth: 8, boxHeight: 8, padding: 16 } },
      tooltip: { backgroundColor: '#1d1d1f', padding: 10, titleFont: { weight: '600' }, displayColors: true },
    },
  };

  const builders = {
    'score-ring': (canvas) => new Chart(canvas, {
      type: 'doughnut',
      data: { datasets: [{ data: [charts.score_ring.score, charts.score_ring.remaining], backgroundColor: ['#1b365d', '#e8e8ed'], borderWidth: 0, hoverOffset: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, cutout: '82%', rotation: -90, circumference: 360, events: [], plugins: { legend: { display: false }, tooltip: { enabled: false } }, animation: { animateRotate: true, duration: Chart.defaults.animation.duration } },
    }),
    'dimension-radar': (canvas) => new Chart(canvas, {
      type: 'radar',
      data: { labels: charts.dimension_radar.labels, datasets: [{ label: '得分率', data: charts.dimension_radar.values, borderColor: '#0066cc', backgroundColor: 'rgba(0,102,204,.10)', pointBackgroundColor: '#0066cc', pointRadius: 3, borderWidth: 2 }] },
      options: { ...common, scales: { r: { min: 0, max: 100, beginAtZero: true, ticks: { stepSize: 20, showLabelBackdrop: false }, pointLabels: { color: '#1d1d1f', font: { size: 11 } }, grid: { color: '#e2e2e7' }, angleLines: { color: '#e2e2e7' } } }, plugins: { ...common.plugins, legend: { display: false } } },
    }),
    'difficulty-stack': (canvas) => new Chart(canvas, {
      type: 'bar',
      data: { labels: charts.difficulty_stack.labels, datasets: [
        { label: '正确', data: charts.difficulty_stack.correct, backgroundColor: '#16794b', borderRadius: 3 },
        { label: '错误', data: charts.difficulty_stack.incorrect, backgroundColor: '#b42318', borderRadius: 3 },
        { label: '未答', data: charts.difficulty_stack.unanswered, backgroundColor: '#c9a640', borderRadius: 3 },
      ] },
      options: { ...common, scales: { x: { stacked: true, grid: { display: false } }, y: { stacked: true, beginAtZero: true, ticks: { precision: 0 } } }, onClick: (_, elements) => { if (elements[0]) applyMatrixFilter(['correct', 'incorrect', 'unanswered'][elements[0].datasetIndex] || 'all'); } },
    }),
    'question-time': (canvas) => new Chart(canvas, {
      type: 'bar',
      data: { labels: charts.question_time.labels, datasets: [
        { type: 'bar', label: '活跃用时（秒）', data: charts.question_time.values, backgroundColor: '#1b365d', borderRadius: 2, order: 2 },
        { type: 'line', label: '个人中位线', data: charts.question_time.labels.map(() => timeMedian), borderColor: '#946200', borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0, tension: 0, order: 1 },
      ] },
      options: { ...common, scales: { x: { grid: { display: false }, ticks: { autoSkip: false, maxRotation: 0, callback: (_, index) => (index % 5 === 0 ? charts.question_time.labels[index] : '') } }, y: { beginAtZero: true } }, onClick: (_, elements) => { if (elements[0]) location.hash = `q${String(elements[0].index + 1).padStart(2, '0')}`; } },
    }),
    'score-trend': (canvas) => new Chart(canvas, {
      type: 'line',
      data: { labels: charts.score_trend.labels, datasets: [{ label: '总分', data: charts.score_trend.scores, borderColor: '#0066cc', backgroundColor: '#0066cc', pointRadius: 4, pointHoverRadius: 6, tension: .25 }] },
      options: { ...common, scales: { x: { grid: { display: false } }, y: { min: 0, max: 100, ticks: { stepSize: 20 } } }, plugins: { ...common.plugins, legend: { display: false } }, onClick: (_, elements) => { const id = elements[0] ? charts.score_trend.attempt_ids[elements[0].index] : null; if (id) location.assign(`${location.pathname.replace(/\/reports\/[^/]+$/, '')}/reports/${id}`); } },
    }),
    'dimension-trend': (canvas) => new Chart(canvas, {
      type: 'line',
      data: { labels: charts.dimension_trend.labels, datasets: charts.dimension_trend.series.map((series, index) => ({ label: series.name, data: series.values, borderColor: palette[index], backgroundColor: palette[index], pointRadius: 3, tension: .25, borderWidth: 1.8 })) },
      options: { ...common, scales: { x: { grid: { display: false } }, y: { min: 0, max: 100, ticks: { stepSize: 20 } } } },
    }),
  };

  render = (canvas) => {
    const type = canvas.dataset.chart;
    if (!canvas.dataset.rendered && builders[type]) {
      builders[type](canvas);
      canvas.dataset.rendered = 'true';
    }
  };

  canvases = [...report.querySelectorAll('canvas[data-chart]')];
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          render(entry.target);
          observer.unobserve(entry.target);
        }
      });
    }, { rootMargin: '180px' });
    canvases.forEach((canvas) => observer.observe(canvas));
  } else {
    canvases.forEach(render);
  }
})();
