<?php

declare(strict_types=1);

use GeoAssessment\Support\View;
?>
<main id="main-content" class="state-page">
  <p class="state-code"><?= View::e($statusCode) ?></p>
  <h1><?= View::e($heading) ?></h1>
  <p><?= View::e($message) ?></p>
  <a class="button button-primary" href="<?= View::e($view->url('/')) ?>">返回首页</a>
</main>
