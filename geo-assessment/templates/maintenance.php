<?php

declare(strict_types=1);

use GeoAssessment\Support\View;
?>
<main id="main-content" class="state-page">
  <p class="state-code">503</p>
  <h1>服务暂不可用</h1>
  <p><?= View::e($message) ?></p>
  <a class="button button-primary" href="<?= View::e($view->url('/')) ?>">重新检查</a>
</main>
