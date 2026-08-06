#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(tr -d '[:space:]' < "${PROJECT_ROOT}/VERSION")"
ARCHIVE="${1:-${PROJECT_ROOT}/dist/geo-assessment-${VERSION}.zip}"
VERIFY_DIR="$(mktemp -d "${TMPDIR:-/tmp}/geo-assessment-verify.XXXXXX")"

cleanup() {
    rm -rf "${VERIFY_DIR}"
}
trap cleanup EXIT

fail() {
    printf '[FAIL] %s\n' "$1" >&2
    exit 1
}

[[ -f "${ARCHIVE}" ]] || fail "找不到 Release：${ARCHIVE}"
command -v unzip >/dev/null 2>&1 || fail '缺少 unzip'
command -v php >/dev/null 2>&1 || fail '缺少 PHP'

if [[ -f "${ARCHIVE}.sha256" ]]; then
    expected="$(awk 'NR == 1 {print $1}' "${ARCHIVE}.sha256")"
    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
    else
        actual="$(shasum -a 256 "${ARCHIVE}" | awk '{print $1}')"
    fi
    [[ "${actual}" == "${expected}" ]] || fail 'SHA-256 校验失败'
fi

entries="$(unzip -Z1 "${ARCHIVE}")"
[[ -n "${entries}" ]] || fail 'Release 是空文件'
if printf '%s\n' "${entries}" | grep -Ev '^geo-assessment/[^/].*|^geo-assessment/.+/.+' >/dev/null; then
    fail 'Release 必须只有 geo-assessment/ 根目录'
fi
if printf '%s\n' "${entries}" | grep -E '(^|/)\.\.?(/|$)' >/dev/null; then
    fail 'Release 包含不安全路径'
fi

unzip -q "${ARCHIVE}" -d "${VERIFY_DIR}"
PACKAGE_DIR="${VERIFY_DIR}/geo-assessment"

for required in \
    VERSION README.md LICENSE LICENSE-CODE LICENSE-CONTENT THIRD_PARTY_NOTICES.md \
    vendor/autoload.php bin/console public/index.php database/seeds/geo-30-v1.2.json \
    deploy/install.sh deploy/nginx-subdirectory.conf.example storage/.gitignore; do
    [[ -f "${PACKAGE_DIR}/${required}" ]] || fail "Release 缺少 ${required}"
done

[[ "$(tr -d '[:space:]' < "${PACKAGE_DIR}/VERSION")" == "${VERSION}" ]] || fail 'Release 版本不一致'
[[ -x "${PACKAGE_DIR}/bin/console" ]] || fail 'bin/console 缺少可执行权限'
[[ -x "${PACKAGE_DIR}/deploy/install.sh" ]] || fail 'deploy/install.sh 缺少可执行权限'
[[ ! -d "${PACKAGE_DIR}/tests" ]] || fail 'Release 不应包含开发测试目录'
[[ ! -d "${PACKAGE_DIR}/vendor/phpunit" ]] || fail 'Release 不应包含 PHPUnit'

CHART_EXPECTED='48444a82d4edcb5bec0f1965faacdde18d9c17db3063d042abada2f705c9f54a'
if command -v sha256sum >/dev/null 2>&1; then
    chart_actual="$(sha256sum "${PACKAGE_DIR}/public/assets/vendor/chart.umd.min.js" | awk '{print $1}')"
else
    chart_actual="$(shasum -a 256 "${PACKAGE_DIR}/public/assets/vendor/chart.umd.min.js" | awk '{print $1}')"
fi
[[ "${chart_actual}" == "${CHART_EXPECTED}" ]] || fail 'Chart.js SHA-256 与许可清单不一致'

for forbidden in \
    '*.sqlite' '*.sqlite-shm' '*.sqlite-wal' 'app.key' '*.log' '*.jsonl' \
    '.env' '*.pem' '*.key' '*.pdf' '*.parquet' '*.duckdb' '*.jsonl' '.DS_Store'; do
    if find "${PACKAGE_DIR}" -type f -name "${forbidden}" -print -quit | grep -q .; then
        fail "Release 包含禁止文件：${forbidden}"
    fi
done

(
    cd "${PACKAGE_DIR}"
    php bin/console app:install
    php bin/console questions:verify
    php bin/console app:health
    [[ "$(php bin/console --version)" == "GEO Assessment ${VERSION}" ]]
)

printf '[OK] Release 内容、安全边界和隔离安装验证通过：%s\n' "${ARCHIVE}"
