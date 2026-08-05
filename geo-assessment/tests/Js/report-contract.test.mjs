import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const reportSource = await readFile(new URL('../../public/assets/report.js', import.meta.url), 'utf8');
const stylesheet = await readFile(new URL('../../public/assets/app.css', import.meta.url), 'utf8');

test('filters and print preparation remain available when chart rendering is unavailable', () => {
  const chartGuard = reportSource.indexOf("if (typeof window.Chart !== 'function')");

  assert.ok(chartGuard > 0);
  assert.ok(reportSource.indexOf('const applyMatrixFilter') < chartGuard);
  assert.ok(reportSource.indexOf('let render = () => {}') < chartGuard);
  assert.ok(reportSource.indexOf("window.addEventListener('beforeprint'") < chartGuard);
});

test('printing expands chart fallback tables and question details', () => {
  const beforePrint = reportSource.slice(
    reportSource.indexOf("window.addEventListener('beforeprint'"),
    reportSource.indexOf("window.addEventListener('afterprint'"),
  );

  assert.match(beforePrint, /details\.forEach\(\(detail\) => \{ detail\.open = true; \}\)/);
  assert.match(beforePrint, /fallbackDetails\.forEach\(\(detail\) => \{ detail\.open = true; \}\)/);
  assert.doesNotMatch(stylesheet, /\.data-fallback\[open\]\s*\{\s*display:\s*none/);
});
