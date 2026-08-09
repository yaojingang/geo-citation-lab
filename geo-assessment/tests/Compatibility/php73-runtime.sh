#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PHP73_IMAGE="${PHP73_IMAGE:-php:7.3-cli}"
RUNTIME_DIR="$(mktemp -d /tmp/geo-php73-runtime.XXXXXX)"

cleanup() {
    rm -rf "${RUNTIME_DIR}"
}
trap cleanup EXIT

mkdir -p "${RUNTIME_DIR}/app/storage"
rsync -a \
    "${PROJECT_ROOT}/bin" \
    "${PROJECT_ROOT}/config" \
    "${PROJECT_ROOT}/database" \
    "${PROJECT_ROOT}/public" \
    "${PROJECT_ROOT}/src" \
    "${PROJECT_ROOT}/templates" \
    "${RUNTIME_DIR}/app/"
cp "${PROJECT_ROOT}/composer.json" "${PROJECT_ROOT}/composer.lock" "${RUNTIME_DIR}/app/"
cp "${PROJECT_ROOT}/storage/.gitignore" "${RUNTIME_DIR}/app/storage/.gitignore"

composer install \
    --working-dir="${RUNTIME_DIR}/app" \
    --no-dev \
    --classmap-authoritative \
    --no-interaction \
    --no-progress

docker run --rm \
    --volume "${RUNTIME_DIR}/app:/app" \
    --workdir /app \
    "${PHP73_IMAGE}" \
    sh -eu -c '
        php bin/console app:install
        php bin/console questions:verify
        php bin/console app:health
    '
