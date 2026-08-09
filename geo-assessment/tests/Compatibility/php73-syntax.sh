#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PHP73_IMAGE="${PHP73_IMAGE:-php:7.3-cli}"

docker run --rm \
    --volume "${PROJECT_ROOT}:/app:ro" \
    --workdir /app \
    "${PHP73_IMAGE}" \
    sh -eu -c "find bin config public src templates -type f \( -name '*.php' -o -path 'bin/console' \) -print0 | xargs -0 -n1 php -l"
