# Third-party notices

## Chart.js 4.5.1

- Project: <https://www.chartjs.org/>
- Source: <https://github.com/chartjs/Chart.js/tree/v4.5.1>
- Local artifact: `public/assets/vendor/chart.umd.min.js`
- License: MIT, copied to `public/assets/vendor/LICENSE.chartjs`
- SHA-256: `48444a82d4edcb5bec0f1965faacdde18d9c17db3063d042abada2f705c9f54a`

## BaconQrCode 2.x

- Project: <https://github.com/Bacon/BaconQrCode>
- Purpose: generate the certificate verification QR code as local SVG markup
- License: BSD 2-Clause, included in the installed Composer package
- Runtime dependencies and exact versions are recorded in `composer.lock`

## PHPUnit and Composer development dependencies

Development packages and exact versions are recorded in `composer.lock`. Their package metadata and licenses are available through:

```bash
composer licenses
```

Production installation uses `composer install --no-dev --classmap-authoritative` and does not include PHPUnit.

## Cited research and datasets

The assessment Release stores citation metadata, stable links and original assessment content. It does not redistribute the cited paper PDFs or the external datasets. The current canonical source list is in `database/seeds/geo-30-v1.2.json`, and the writing boundary is documented in `docs/question-provenance.md`.

## Optional Baidu Analytics

The public source and local development environment leave analytics disabled. A deployer may set `GEO_BAIDU_ANALYTICS_ID` to load Baidu Analytics from `https://hm.baidu.com`. This optional integration follows Baidu's terms and privacy policy. Deployers are responsible for user notice, consent requirements and regional compliance.
