#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WEB_USER="${WEB_USER:-www-data}"
PHP_CANDIDATE="${PHP_BIN:-php}"
DATA_DIR="${GEO_DATA_DIR:-}"
PUBLIC_URL="${GEO_PUBLIC_URL:-}"
LEGACY_DATA_DIR="${GEO_LEGACY_DATA_DIR:-${PROJECT_ROOT}/storage}"

fail() {
    printf '错误：%s\n' "$1" >&2
    exit 1
}

[[ "$(id -u)" -eq 0 ]] || fail '请使用 sudo 或 root 用户执行该脚本'
command -v runuser >/dev/null 2>&1 || fail '系统缺少 runuser 命令'
id "${WEB_USER}" >/dev/null 2>&1 || fail "找不到 PHP-FPM 用户：${WEB_USER}"

PHP_BIN="$(command -v "${PHP_CANDIDATE}" 2>/dev/null || true)"
[[ -n "${PHP_BIN}" ]] || fail "找不到 PHP 命令：${PHP_CANDIDATE}"
[[ -n "${DATA_DIR}" ]] || fail '生产部署必须设置 GEO_DATA_DIR，并指向版本目录外的持久化目录'
[[ -n "${PUBLIC_URL}" ]] || fail '生产部署必须设置 GEO_PUBLIC_URL'
if ! "${PHP_BIN}" -r '$parts = parse_url($argv[1]); exit(is_array($parts) && isset($parts["scheme"], $parts["host"]) && strtolower($parts["scheme"]) === "https" && !isset($parts["user"]) && !isset($parts["pass"]) && !isset($parts["query"]) && !isset($parts["fragment"]) ? 0 : 1);' "${PUBLIC_URL}"; then
    fail 'GEO_PUBLIC_URL 必须是无用户信息、查询参数和片段的完整 HTTPS 地址'
fi
if [[ "${DATA_DIR}" != /* ]]; then
    DATA_DIR="${PROJECT_ROOT}/${DATA_DIR}"
fi
if [[ "${LEGACY_DATA_DIR}" != /* ]]; then
    LEGACY_DATA_DIR="${PROJECT_ROOT}/${LEGACY_DATA_DIR}"
fi
[[ -n "${DATA_DIR}" && "${DATA_DIR}" != "/" ]] || fail 'GEO_DATA_DIR 不能指向文件系统根目录'
[[ -n "${LEGACY_DATA_DIR}" && "${LEGACY_DATA_DIR}" != "/" ]] || fail 'GEO_LEGACY_DATA_DIR 不能指向文件系统根目录'
install -d -m 0770 "${DATA_DIR}"
DATA_DIR="$(cd -- "${DATA_DIR}" && pwd -P)"
if [[ "${DATA_DIR}" == "${PROJECT_ROOT}" || "${DATA_DIR}" == "${PROJECT_ROOT}/"* ]]; then
    fail 'GEO_DATA_DIR 必须位于版本目录外，避免升级时切换到新数据库'
fi
if [[ -d "${LEGACY_DATA_DIR}" ]]; then
    LEGACY_DATA_DIR="$(cd -- "${LEGACY_DATA_DIR}" && pwd -P)"
fi
if [[ "${LEGACY_DATA_DIR}" != "${DATA_DIR}" && ! -e "${DATA_DIR}/app.sqlite" && ! -e "${DATA_DIR}/app.key" && ( -e "${LEGACY_DATA_DIR}/app.sqlite" || -e "${LEGACY_DATA_DIR}/app.key" ) ]]; then
    fail "检测到旧数据目录 ${LEGACY_DATA_DIR}；请暂停写入并将其内容完整复制到 ${DATA_DIR}，再重新运行安装脚本"
fi

if ! "${PHP_BIN}" -r 'exit(PHP_VERSION_ID >= 70305 ? 0 : 1);'; then
    fail "需要 PHP 7.3.5 或更高版本，当前版本是 $("${PHP_BIN}" -r 'echo PHP_VERSION;')"
fi

for extension in json mbstring openssl PDO pdo_sqlite; do
    "${PHP_BIN}" -r "exit(extension_loaded('${extension}') ? 0 : 1);" \
        || fail "缺少 PHP 扩展：${extension}"
done

SQLITE_VERSION="$("${PHP_BIN}" -r '$pdo = new PDO("sqlite::memory:"); echo $pdo->query("SELECT sqlite_version()")->fetchColumn();')" \
    || fail '无法通过 PDO SQLite 读取运行库版本'
"${PHP_BIN}" -r 'exit(version_compare($argv[1], "3.24.0", ">=") ? 0 : 1);' "${SQLITE_VERSION}" \
    || fail "需要 SQLite 3.24.0 或更高版本，当前版本是 ${SQLITE_VERSION}"

WEB_GROUP="$(id -gn "${WEB_USER}")"
install -d -o "${WEB_USER}" -g "${WEB_GROUP}" -m 0770 \
    "${DATA_DIR}" \
    "${DATA_DIR}/logs" \
    "${DATA_DIR}/backups"
chown -R "${WEB_USER}:${WEB_GROUP}" "${DATA_DIR}"
find "${DATA_DIR}" -type d -exec chmod 0770 {} +
find "${DATA_DIR}" -type f -exec chmod 0660 {} +
if [[ -f "${DATA_DIR}/app.key" ]]; then
    chmod 0600 "${DATA_DIR}/app.key"
fi

run_as_web_user() {
    runuser -u "${WEB_USER}" -- env GEO_DATA_DIR="${DATA_DIR}" GEO_PUBLIC_URL="${PUBLIC_URL}" "${PHP_BIN}" "${PROJECT_ROOT}/bin/console" "$@"
}

run_as_web_user app:install
run_as_web_user questions:verify
run_as_web_user app:health

printf '\n部署初始化完成\n'
printf '项目目录：%s\n' "${PROJECT_ROOT}"
printf '持久化目录：%s\n' "${DATA_DIR}"
printf '公开地址：%s\n' "${PUBLIC_URL}"
printf '运行用户：%s:%s\n' "${WEB_USER}" "${WEB_GROUP}"
printf 'PHP 命令：%s\n' "${PHP_BIN}"
printf '运行环境：PHP %s · SQLite %s\n' "$("${PHP_BIN}" -r 'echo PHP_VERSION;')" "${SQLITE_VERSION}"
printf '下一步：配置 Web 文档根或参考 deploy/nginx-subdirectory.conf.example\n'
