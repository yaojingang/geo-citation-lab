<?php

declare(strict_types=1);

namespace GeoAssessment\Reporting;

use BaconQrCode\Common\ErrorCorrectionLevel;
use BaconQrCode\Encoder\Encoder;
use InvalidArgumentException;

final class QrCodeSvg
{
    public function render(string $content): string
    {
        if ($content === '') {
            throw new InvalidArgumentException('二维码内容不能为空');
        }

        set_error_handler(static function (int $severity): bool {
            return $severity === E_DEPRECATED;
        }, E_DEPRECATED);
        try {
            $matrix = Encoder::encode($content, ErrorCorrectionLevel::M())->getMatrix();
        } finally {
            restore_error_handler();
        }
        $margin = 4;
        $size = $matrix->getWidth() + ($margin * 2);
        $path = '';
        for ($y = 0; $y < $matrix->getHeight(); $y++) {
            $x = 0;
            while ($x < $matrix->getWidth()) {
                if ($matrix->get($x, $y) !== 1) {
                    $x++;
                    continue;
                }
                $start = $x;
                while ($x < $matrix->getWidth() && $matrix->get($x, $y) === 1) {
                    $x++;
                }
                $length = $x - $start;
                $path .= 'M' . ($start + $margin) . ' ' . ($y + $margin) . 'h' . $length . 'v1h-' . $length . 'z';
            }
        }

        return '<svg xmlns="http://www.w3.org/2000/svg" width="350" height="350" viewBox="0 0 '
            . $size . ' ' . $size . '" shape-rendering="crispEdges" role="img" aria-label="证书查询二维码">'
            . '<rect width="100%" height="100%" fill="#fff"/>'
            . '<path d="' . $path . '" fill="#1b365d"/>'
            . '</svg>';
    }

    public function dataUri(string $content): string
    {
        return 'data:image/svg+xml;base64,' . base64_encode($this->render($content));
    }
}
