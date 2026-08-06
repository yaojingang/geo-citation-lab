(() => {
  'use strict';

  const certificate = document.querySelector('[data-certificate]');
  const dataNode = document.querySelector('#certificate-data');
  if (!certificate || !dataNode) return;

  const certificateData = JSON.parse(dataNode.dataset.certificateJson || '{}');
  const radarCanvas = certificate.querySelector('[data-certificate-radar]');
  const exportCanvas = certificate.querySelector('[data-certificate-export]');
  const qrImage = certificate.querySelector('.certificate-verification img');
  const downloadButton = certificate.querySelector('[data-download-certificate]');
  const toast = certificate.querySelector('[data-certificate-toast]');

  const polygonPoint = (cx, cy, radius, index, count) => {
    const angle = -Math.PI / 2 + (Math.PI * 2 * index / count);
    return [cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius];
  };

  const drawRadar = (canvas, labels, values) => {
    if (!canvas) return;
    const context = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * .31;
    const count = values.length;

    context.clearRect(0, 0, width, height);
    context.lineJoin = 'round';
    context.lineCap = 'round';

    for (let level = 1; level <= 5; level += 1) {
      context.beginPath();
      for (let index = 0; index < count; index += 1) {
        const point = polygonPoint(cx, cy, radius * level / 5, index, count);
        if (index === 0) context.moveTo(point[0], point[1]);
        else context.lineTo(point[0], point[1]);
      }
      context.closePath();
      context.strokeStyle = level === 5 ? '#b7c2cf' : '#e1e5ea';
      context.lineWidth = level === 5 ? 2.2 : 1.4;
      context.stroke();
    }

    for (let index = 0; index < count; index += 1) {
      const point = polygonPoint(cx, cy, radius, index, count);
      context.beginPath();
      context.moveTo(cx, cy);
      context.lineTo(point[0], point[1]);
      context.strokeStyle = '#e1e5ea';
      context.lineWidth = 1.2;
      context.stroke();
    }

    context.beginPath();
    values.forEach((value, index) => {
      const point = polygonPoint(cx, cy, radius * Number(value) / 100, index, count);
      if (index === 0) context.moveTo(point[0], point[1]);
      else context.lineTo(point[0], point[1]);
    });
    context.closePath();
    context.fillStyle = 'rgba(27, 54, 93, .13)';
    context.fill();
    context.strokeStyle = '#1b365d';
    context.lineWidth = 4;
    context.stroke();

    values.forEach((value, index) => {
      const point = polygonPoint(cx, cy, radius * Number(value) / 100, index, count);
      context.beginPath();
      context.arc(point[0], point[1], 7, 0, Math.PI * 2);
      context.fillStyle = '#fff';
      context.fill();
      context.strokeStyle = '#1b365d';
      context.lineWidth = 4;
      context.stroke();
    });

    context.fillStyle = '#4f5054';
    context.font = '500 30px -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    labels.forEach((label, index) => {
      const point = polygonPoint(cx, cy, radius + 80, index, count);
      context.fillText(label, point[0], point[1]);
    });
  };

  const fitFont = (context, text, maxWidth, maximum, minimum, weight, family) => {
    let size = maximum;
    while (size > minimum) {
      context.font = `${weight} ${size}px ${family}`;
      if (context.measureText(text).width <= maxWidth) break;
      size -= 2;
    }
    return size;
  };

  const wrapCanvasText = (context, text, x, y, maxWidth, lineHeight, maxLines) => {
    const characters = Array.from(text);
    let line = '';
    let lineIndex = 0;
    characters.forEach((character, index) => {
      const candidate = line + character;
      if (context.measureText(candidate).width > maxWidth && line !== '' && lineIndex < maxLines - 1) {
        context.fillText(line, x, y + lineIndex * lineHeight);
        line = character;
        lineIndex += 1;
      } else {
        line = candidate;
      }
      if (index === characters.length - 1) context.fillText(line, x, y + lineIndex * lineHeight);
    });
  };

  const drawExportRadar = (context, cx, cy, radius, labels, values) => {
    const count = values.length;
    for (let level = 1; level <= 5; level += 1) {
      context.beginPath();
      for (let index = 0; index < count; index += 1) {
        const point = polygonPoint(cx, cy, radius * level / 5, index, count);
        if (index === 0) context.moveTo(point[0], point[1]);
        else context.lineTo(point[0], point[1]);
      }
      context.closePath();
      context.strokeStyle = level === 5 ? '#b7c2cf' : '#e1e5ea';
      context.lineWidth = level === 5 ? 2 : 1;
      context.stroke();
    }
    for (let index = 0; index < count; index += 1) {
      const point = polygonPoint(cx, cy, radius, index, count);
      context.beginPath();
      context.moveTo(cx, cy);
      context.lineTo(point[0], point[1]);
      context.strokeStyle = '#e1e5ea';
      context.stroke();
    }
    context.beginPath();
    values.forEach((value, index) => {
      const point = polygonPoint(cx, cy, radius * Number(value) / 100, index, count);
      if (index === 0) context.moveTo(point[0], point[1]);
      else context.lineTo(point[0], point[1]);
    });
    context.closePath();
    context.fillStyle = 'rgba(27, 54, 93, .14)';
    context.fill();
    context.strokeStyle = '#1b365d';
    context.lineWidth = 5;
    context.stroke();
    values.forEach((value, index) => {
      const point = polygonPoint(cx, cy, radius * Number(value) / 100, index, count);
      context.beginPath();
      context.arc(point[0], point[1], 7, 0, Math.PI * 2);
      context.fillStyle = '#fff';
      context.fill();
      context.strokeStyle = '#1b365d';
      context.lineWidth = 4;
      context.stroke();
    });
    context.fillStyle = '#4f5054';
    context.font = '500 25px -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    labels.forEach((label, index) => {
      const point = polygonPoint(cx, cy, radius + 65, index, count);
      context.fillText(label, point[0], point[1]);
    });
    context.fillStyle = '#1b365d';
    context.font = '700 46px -apple-system, BlinkMacSystemFont, sans-serif';
    context.fillText(String(certificateData.score), cx, cy - 5);
    context.fillStyle = '#6e6e73';
    context.font = '500 14px -apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif';
    context.fillText('综合得分', cx, cy + 36);
  };

  const drawExportCertificate = (image) => {
    if (!exportCanvas) return null;
    const context = exportCanvas.getContext('2d');
    const width = exportCanvas.width;
    const height = exportCanvas.height;
    const blue = '#1b365d';
    const serif = '"Songti SC", "STSong", serif';
    const sans = '-apple-system, BlinkMacSystemFont, "PingFang SC", sans-serif';

    context.clearRect(0, 0, width, height);
    context.fillStyle = '#fff';
    context.fillRect(0, 0, width, height);
    context.strokeStyle = '#c9c8c2';
    context.lineWidth = 2;
    context.strokeRect(28, 28, width - 56, height - 56);
    context.strokeStyle = '#eeede8';
    context.strokeRect(42, 42, width - 84, height - 84);

    context.strokeStyle = blue;
    context.lineWidth = 3;
    context.beginPath();
    context.arc(116, 112, 33, 0, Math.PI * 2);
    context.stroke();
    context.lineWidth = 1;
    context.globalAlpha = .34;
    context.beginPath();
    context.arc(116, 112, 27, 0, Math.PI * 2);
    context.stroke();
    context.globalAlpha = 1;
    context.fillStyle = blue;
    context.font = '700 34px Georgia, serif';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText('G', 116, 114);
    context.beginPath();
    context.arc(138, 89, 7, 0, Math.PI * 2);
    context.fillStyle = '#0066cc';
    context.fill();

    context.textAlign = 'left';
    context.fillStyle = blue;
    context.font = `650 26px ${sans}`;
    context.fillText('GEO Assessment', 168, 105);
    context.fillStyle = '#6e6e73';
    context.font = `500 14px ${sans}`;
    context.fillText('GEO CITATION LAB', 168, 132);

    context.textAlign = 'right';
    context.fillStyle = '#6e6e73';
    context.font = `500 13px ${sans}`;
    context.fillText('PROFESSIONAL CAPABILITY CERTIFICATE', width - 102, 101);
    context.fillStyle = blue;
    context.font = `650 16px ${sans}`;
    context.fillText(`NO. ${certificateData.id}`, width - 102, 130);

    context.textAlign = 'left';
    context.fillStyle = blue;
    fitFont(context, 'GEO专业能力测试评估证书', 760, 54, 42, 500, serif);
    context.fillText('GEO专业能力测试评估证书', 102, 330);
    context.fillStyle = '#6e6e73';
    context.font = `500 16px ${sans}`;
    context.fillText('证书获得者', 102, 428);
    context.fillStyle = '#1d1d1f';
    fitFont(context, String(certificateData.name), 670, 96, 28, 500, serif);
    context.fillText(String(certificateData.name), 102, 530);

    context.fillStyle = '#6e6e73';
    context.font = `500 15px ${sans}`;
    context.fillText('综合得分', 102, 628);
    context.fillStyle = blue;
    context.font = `500 64px ${serif}`;
    context.fillText(String(certificateData.score), 102, 696);

    if (certificateData.tier) {
      context.fillStyle = '#6e6e73';
      context.font = `500 15px ${sans}`;
      context.fillText('专业称号', 292, 628);
      context.fillStyle = blue;
      context.fillRect(292, 667, 188, 58);
      context.fillStyle = '#fff';
      context.font = `500 31px ${serif}`;
      context.textAlign = 'center';
      context.fillText(String(certificateData.tier), 386, 696);
      context.textAlign = 'left';
    }

    context.fillStyle = '#4d4d51';
    context.font = `500 25px ${serif}`;
    wrapCanvasText(context, String(certificateData.encouragement), 102, 777, 660, 43, 3);

    drawExportRadar(context, 1244, 574, 210, certificateData.labels, certificateData.values);

    context.strokeStyle = '#e8e8ed';
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(102, 954);
    context.lineTo(width - 102, 954);
    context.stroke();

    context.fillStyle = '#6e6e73';
    context.font = `500 13px ${sans}`;
    context.fillText('签发日期', 120, 1002);
    context.fillText('证书编号', 304, 1002);
    context.fillText('签发方', 560, 1002);
    context.fillStyle = '#1d1d1f';
    context.font = `650 16px ${sans}`;
    context.fillText(String(certificateData.date), 120, 1033);
    context.fillText(String(certificateData.id), 304, 1033);
    context.fillText(String(certificateData.issuer), 560, 1033);

    if (image && image.complete && image.naturalWidth > 0) context.drawImage(image, width - 232, 973, 124, 124);
    context.textAlign = 'right';
    context.fillStyle = blue;
    context.font = `700 13px ${sans}`;
    context.fillText('VERIFY CERTIFICATE', width - 252, 1036);
    document.documentElement.dataset.certificateExportReady = 'true';
    document.documentElement.dataset.certificateExportLength = String(exportCanvas.toDataURL('image/png').length);
    return exportCanvas;
  };

  const prepareExport = async () => {
    if (qrImage && typeof qrImage.decode === 'function') {
      try { await qrImage.decode(); } catch (error) { /* the load event remains available */ }
    }
    const canvas = drawExportCertificate(qrImage);
    if (canvas && new URLSearchParams(window.location.search).has('export-preview')) {
      const preview = document.createElement('img');
      preview.className = 'certificate-export-preview';
      preview.alt = '下载证书图片预览';
      preview.src = canvas.toDataURL('image/png');
      document.body.appendChild(preview);
      document.documentElement.classList.add('is-certificate-export-preview');
    }
  };

  drawRadar(radarCanvas, certificateData.labels || [], certificateData.values || []);
  document.documentElement.dataset.certificateReady = 'true';
  if (qrImage && !qrImage.complete) qrImage.addEventListener('load', prepareExport, { once: true });
  else prepareExport();

  if (downloadButton) {
    downloadButton.addEventListener('click', async () => {
      await prepareExport();
      const canvas = drawExportCertificate(qrImage);
      if (!canvas) return;
      const link = document.createElement('a');
      const safeName = String(certificateData.name || '测试者').replace(/[\\/:*?"<>|]/g, '-');
      link.download = `GEO专业能力测试评估证书-${safeName}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
      if (toast) {
        toast.classList.add('is-visible');
        window.setTimeout(() => toast.classList.remove('is-visible'), 1800);
      }
    });
  }
})();
