import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const certificateSource = await readFile(new URL('../../public/assets/certificate.js', import.meta.url), 'utf8');
const certificateTemplate = await readFile(new URL('../../templates/certificate.php', import.meta.url), 'utf8');
const reportTemplate = await readFile(new URL('../../templates/report.php', import.meta.url), 'utf8');
const layoutTemplate = await readFile(new URL('../../templates/layout.php', import.meta.url), 'utf8');
const stylesheet = await readFile(new URL('../../public/assets/app.css', import.meta.url), 'utf8');

test('certificate uses dynamic server data and a local canvas export', () => {
  assert.match(certificateSource, /certificateJson/);
  assert.match(certificateSource, /data-certificate-export/);
  assert.match(certificateSource, /toDataURL\('image\/png'\)/);
  assert.doesNotMatch(certificateSource, /html2canvas|https?:\/\//);
  assert.match(certificateTemplate, /data-certificate-json/);
  assert.match(certificateTemplate, /qr_data_uri/);
});

test('certificate keeps the approved title and report entry', () => {
  assert.match(certificateSource, /GEO专业能力测试评估证书/);
  assert.match(certificateTemplate, /certificate\['title'\]/);
  assert.doesNotMatch(certificateTemplate, /生成式引擎优化专业能力评估/);
  assert.match(reportTemplate, /report-certificate-entry/);
  assert.match(reportTemplate, /查看我的证书/);
});

test('certificate metadata and print layout stay isolated from the report', () => {
  assert.match(layoutTemplate, /\$pageClass === 'page-certificate'/);
  assert.match(layoutTemplate, /GEO专业能力测试评估证书，展示综合得分、专业称号与六维能力画像/);
  assert.match(stylesheet, /@page certificate-page\s*{[^}]*size:\s*A4 landscape/s);
  assert.match(stylesheet, /\.page-certificate \.certificate-paper\s*{[^}]*page:\s*certificate-page/s);
  assert.doesNotMatch(stylesheet, /@media print\s*{\s*@page\s*{\s*size:\s*A4 landscape/s);
});
