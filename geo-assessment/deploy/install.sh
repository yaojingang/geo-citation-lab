#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WEB_USER="${WEB_USER:-www-data}"
PHP_CANDIDATE="${PHP_BIN:-php}"

fail() {
    printf '错误：%s\n' "$1" >&2
    exit 1
}

[[ "$(id -u)" -eq 0 ]] || fail '请使用 sudo 或 root 用户执行该脚本'
command -v runuser >/dev/null 2>&1 || fail '系统缺少 runuser 命令'
id "${WEB_USER}" >/dev/null 2>&1 || fail "找不到 PHP-FPM 用户：${WEB_USER}"

PHP_BIN="$(command -v "${PHP_CANDIDATE}" 2>/dev/null || true)"
[[ -n "${PHP_BIN}" ]] || fail "找不到 PHP 命令：${PHP_CANDIDATE}"

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
    "${PROJECT_ROOT}/storage" \
    "${PROJECT_ROOT}/storage/logs" \
    "${PROJECT_ROOT}/storage/backups"
chown -R "${WEB_USER}:${WEB_GROUP}" "${PROJECT_ROOT}/storage"
find "${PROJECT_ROOT}/storage" -type d -exec chmod 0770 {} +
find "${PROJECT_ROOT}/storage" -type f -exec chmod 0660 {} +
if [[ -f "${PROJECT_ROOT}/storage/app.key" ]]; then
    chmod 0600 "${PROJECT_ROOT}/storage/app.key"
fi

run_as_web_user() {
    runuser -u "${WEB_USER}" -- "${PHP_BIN}" "${PROJECT_ROOT}/bin/console" "$@"
}

run_as_web_user app:install
run_as_web_user questions:verify
run_as_web_user app:health

printf '\n部署初始化完成\n'
printf '项目目录：%s\n' "${PROJECT_ROOT}"
printf '运行用户：%s:%s\n' "${WEB_USER}" "${WEB_GROUP}"
printf 'PHP 命令：%s\n' "${PHP_BIN}"
printf '运行环境：PHP %s · SQLite %s\n' "$("${PHP_BIN}" -r 'echo PHP_VERSION;')" "${SQLITE_VERSION}"
printf '下一步：配置 Web 文档根或参考 deploy/nginx-subdirectory.conf.example\n'
