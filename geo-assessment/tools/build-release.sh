#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(tr -d '[:space:]' < "${PROJECT_ROOT}/VERSION")"
TAG="${1:-assessment-v${VERSION}}"
EXPECTED_TAG="assessment-v${VERSION}"
DIST_DIR="${PROJECT_ROOT}/dist"
ARCHIVE="${DIST_DIR}/geo-assessment-${VERSION}.zip"
SIDECAR="${ARCHIVE}.sha256"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/geo-assessment-release.XXXXXX")"
PACKAGE_DIR="${BUILD_DIR}/geo-assessment"

cleanup() {
    rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

fail() {
    printf '[FAIL] %s\n' "$1" >&2
    exit 1
}

[[ "${TAG}" == "${EXPECTED_TAG}" ]] || fail "标签应为 ${EXPECTED_TAG}，当前值是 ${TAG}"
[[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "VERSION 格式无效：${VERSION}"
command -v composer >/dev/null 2>&1 || fail '缺少 Composer'
command -v zip >/dev/null 2>&1 || fail '缺少 zip'

mkdir -p "${PACKAGE_DIR}" "${DIST_DIR}"
while IFS= read -r entry; do
    [[ -n "${entry}" ]] || continue
    source_path="${PROJECT_ROOT}/${entry}"
    [[ -e "${source_path}" ]] || fail "发布清单文件不存在：${entry}"
    mkdir -p "$(dirname -- "${PACKAGE_DIR}/${entry}")"
    cp -R "${source_path}" "${PACKAGE_DIR}/${entry}"
done < "${PROJECT_ROOT}/tools/release-files.txt"

composer install \
    --working-dir="${PACKAGE_DIR}" \
    --no-dev \
    --classmap-authoritative \
    --no-interaction \
    --no-progress

find "${PACKAGE_DIR}" -name '.DS_Store' -delete
chmod 0755 \
    "${PACKAGE_DIR}/bin/console" \
    "${PACKAGE_DIR}/deploy/install.sh" \
    "${PACKAGE_DIR}/deploy/examples/ai.laoyao.cn/install.sh"

case "${ARCHIVE}" in
    "${DIST_DIR}"/geo-assessment-*.zip) rm -f "${ARCHIVE}" "${SIDECAR}" ;;
    *) fail "拒绝覆盖未验证路径：${ARCHIVE}" ;;
esac

(
    cd "${BUILD_DIR}"
    find geo-assessment -type f -print | LC_ALL=C sort | zip -q -X "${ARCHIVE}" -@
)

if command -v sha256sum >/dev/null 2>&1; then
    (cd "${DIST_DIR}" && sha256sum "$(basename -- "${ARCHIVE}")" > "$(basename -- "${SIDECAR}")")
else
    checksum="$(shasum -a 256 "${ARCHIVE}" | awk '{print $1}')"
    printf '%s  %s\n' "${checksum}" "$(basename -- "${ARCHIVE}")" > "${SIDECAR}"
fi

printf '[OK] Release 已生成 %s\n' "${ARCHIVE}"
printf '[OK] SHA-256 %s\n' "$(awk '{print $1}' "${SIDECAR}")"
