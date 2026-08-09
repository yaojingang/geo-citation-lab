#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
EXPECTED_ROOT="/www/wwwroot/ai.laoyao.cn/geo"

if [[ "${PROJECT_ROOT}" != "${EXPECTED_ROOT}" ]]; then
    printf '错误：项目应位于 %s，当前路径是 %s\n' "${EXPECTED_ROOT}" "${PROJECT_ROOT}" >&2
    exit 1
fi

GEO_DATA_DIR=/www/wwwroot/ai.laoyao.cn/geo-data GEO_PUBLIC_URL=https://ai.laoyao.cn/geo WEB_USER=www PHP_BIN=/www/server/php/73/bin/php "${PROJECT_ROOT}/deploy/install.sh"
