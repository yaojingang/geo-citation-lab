#!/usr/bin/env bash
set -euo pipefail

SMOKE_BASE="${1:-http://127.0.0.1:8080}"
SMOKE_BASE="${SMOKE_BASE%/}"
SMOKE_ORIGIN="$(printf '%s' "$SMOKE_BASE" | sed -E 's#^(https?://[^/]+).*$#\1#')"
SMOKE_PREFIX="${SMOKE_BASE#"$SMOKE_ORIGIN"}"
SMOKE_ASSET_BASE="${SMOKE_ASSET_BASE:-$SMOKE_BASE}"
SMOKE_ASSET_BASE="${SMOKE_ASSET_BASE%/}"
SMOKE_TMP="$(mktemp -d)"
trap 'rm -rf "$SMOKE_TMP"' EXIT
SMOKE_JAR="$SMOKE_TMP/cookies.txt"
SMOKE_NAME="冒烟测试-$(date +%s)"

smoke_url() {
  local location="$1"
  if [[ "$location" =~ ^https?:// ]]; then
    printf '%s' "$location"
  elif [[ -n "$SMOKE_PREFIX" && ("$location" == "$SMOKE_PREFIX" || "$location" == "$SMOKE_PREFIX/"*) ]]; then
    printf '%s%s' "$SMOKE_ORIGIN" "$location"
  else
    printf '%s%s' "$SMOKE_BASE" "$location"
  fi
}

curl -fsS -D "$SMOKE_TMP/home.headers" -c "$SMOKE_JAR" "$SMOKE_BASE/" -o "$SMOKE_TMP/home.html"
grep -q 'GEO 在线能力测试' "$SMOKE_TMP/home.html"
grep -qi '^Content-Security-Policy:' "$SMOKE_TMP/home.headers"
grep -qi '^X-Content-Type-Options: nosniff' "$SMOKE_TMP/home.headers"
grep -qi '^Cache-Control: no-store, private' "$SMOKE_TMP/home.headers"

for asset in assets/app.css assets/quiz.js assets/report.js assets/vendor/chart.umd.min.js; do
  curl -fsS "$SMOKE_ASSET_BASE/$asset" -o "$SMOKE_TMP/$(basename "$asset")"
  test -s "$SMOKE_TMP/$(basename "$asset")"
done
if [[ -n "${SMOKE_EXPECT_ANALYTICS_ID:-}" ]]; then
  sed -n '/<head>/,/<\/head>/p' "$SMOKE_TMP/home.html" | grep -q 'var _hmt = _hmt || \[\];'
  sed -n '/<head>/,/<\/head>/p' "$SMOKE_TMP/home.html" | grep -q "https://hm.baidu.com/hm.js?${SMOKE_EXPECT_ANALYTICS_ID}"
  grep -qi '^Content-Security-Policy:.*https://hm.baidu.com' "$SMOKE_TMP/home.headers"
else
  if grep -q 'var _hmt = _hmt || \[\];' "$SMOKE_TMP/home.html"; then
    printf '[FAIL] 未配置统计 ID 时不应加载百度统计\n' >&2
    exit 1
  fi
  if grep -qi '^Content-Security-Policy:.*https://hm.baidu.com' "$SMOKE_TMP/home.headers"; then
    printf '[FAIL] 未配置统计 ID 时 CSP 不应允许百度统计域名\n' >&2
    exit 1
  fi
fi
if grep -q '/assets/analytics.js' "$SMOKE_TMP/home.html"; then
  printf '[FAIL] 百度统计初始化代码应直接内联到页面\n' >&2
  exit 1
fi

if [[ "$SMOKE_BASE" == http://* ]]; then
  curl -fsS -D "$SMOKE_TMP/spoofed-proto.headers" -H 'X-Forwarded-Proto: https' "$SMOKE_BASE/" -o /dev/null
  if grep -qi '^Set-Cookie:.*; Secure' "$SMOKE_TMP/spoofed-proto.headers"; then
    printf '[FAIL] HTTP 环境中未信任的 X-Forwarded-Proto 不应启用 Secure Cookie\n' >&2
    exit 1
  fi
fi

SMOKE_CSRF="$(sed -n 's/.*name="_csrf" value="\([^"]*\)".*/\1/p' "$SMOKE_TMP/home.html" | head -n 1)"
test -n "$SMOKE_CSRF"

SMOKE_BAD_STATUS="$(curl -sS -o /dev/null -w '%{http_code}' -b "$SMOKE_JAR" -X POST "$SMOKE_BASE/identity" --data-urlencode '_csrf=invalid' --data-urlencode 'name=错误验证')"
test "$SMOKE_BAD_STATUS" = '403'

curl -fsS -D "$SMOKE_TMP/identity.headers" -b "$SMOKE_JAR" -c "$SMOKE_JAR" \
  -X POST "$SMOKE_BASE/identity" \
  --data-urlencode "_csrf=$SMOKE_CSRF" \
  --data-urlencode "name=$SMOKE_NAME" \
  -o /dev/null
SMOKE_QUESTION_PATH="$(awk 'tolower($1)=="location:" {sub(/\r$/, "", $2); print $2}' "$SMOKE_TMP/identity.headers" | tail -n 1)"
test -n "$SMOKE_QUESTION_PATH"

curl -fsS -b "$SMOKE_JAR" -c "$SMOKE_JAR" "$(smoke_url "$SMOKE_QUESTION_PATH")" -o "$SMOKE_TMP/question.html"
grep -q 'data-quiz' "$SMOKE_TMP/question.html"
grep -q 'data-question-form' "$SMOKE_TMP/question.html"
if [[ -n "${SMOKE_EXPECT_ANALYTICS_ID:-}" ]]; then
  sed -n '/<head>/,/<\/head>/p' "$SMOKE_TMP/question.html" | grep -q "https://hm.baidu.com/hm.js?${SMOKE_EXPECT_ANALYTICS_ID}"
fi
SMOKE_QUESTION_ROUTE="$SMOKE_QUESTION_PATH"
if [[ -n "$SMOKE_PREFIX" && ("$SMOKE_QUESTION_ROUTE" == "$SMOKE_PREFIX" || "$SMOKE_QUESTION_ROUTE" == "$SMOKE_PREFIX/"*) ]]; then
  SMOKE_QUESTION_ROUTE="${SMOKE_QUESTION_ROUTE#"$SMOKE_PREFIX"}"
fi
SMOKE_ATTEMPT="$(printf '%s' "$SMOKE_QUESTION_ROUTE" | sed -n 's#^/attempts/\([^/]*\)/questions/.*#\1#p')"
SMOKE_CODE="$(sed -n 's/.*name="question_code" value="\([^"]*\)".*/\1/p' "$SMOKE_TMP/question.html" | head -n 1)"
test -n "$SMOKE_ATTEMPT"
test -n "$SMOKE_CODE"

curl -fsS -D "$SMOKE_TMP/submit.headers" -b "$SMOKE_JAR" -c "$SMOKE_JAR" \
  -X POST "$SMOKE_BASE/attempts/$SMOKE_ATTEMPT/submit" \
  --data-urlencode "_csrf=$SMOKE_CSRF" \
  --data-urlencode "question_code=$SMOKE_CODE" \
  --data-urlencode 'activity_seq=1' \
  --data-urlencode 'active_seconds_delta=1' \
  -o /dev/null
SMOKE_REPORT_PATH="$(awk 'tolower($1)=="location:" {sub(/\r$/, "", $2); print $2}' "$SMOKE_TMP/submit.headers" | tail -n 1)"
test -n "$SMOKE_REPORT_PATH"

curl -fsS -b "$SMOKE_JAR" "$(smoke_url "$SMOKE_REPORT_PATH")" -o "$SMOKE_TMP/report.html"
grep -q 'GEO 能力报告' "$SMOKE_TMP/report.html"
test "$(grep -c 'class="question-detail ' "$SMOKE_TMP/report.html")" -eq 30
grep -q 'data-chart="dimension-radar"' "$SMOKE_TMP/report.html"
if [[ -n "${SMOKE_EXPECT_ANALYTICS_ID:-}" ]]; then
  sed -n '/<head>/,/<\/head>/p' "$SMOKE_TMP/report.html" | grep -q "https://hm.baidu.com/hm.js?${SMOKE_EXPECT_ANALYTICS_ID}"
fi

SMOKE_SWITCH_STATUS="$(curl -sS -o "$SMOKE_TMP/switch.html" -w '%{http_code}' -b "$SMOKE_JAR" -c "$SMOKE_JAR" \
  -X POST "$SMOKE_BASE/switch-user" \
  --data-urlencode "_csrf=$SMOKE_CSRF")"
test "$SMOKE_SWITCH_STATUS" = '422'
grep -q '请先确认切换' "$SMOKE_TMP/switch.html"
curl -fsS -b "$SMOKE_JAR" "$(smoke_url "$SMOKE_REPORT_PATH")" -o /dev/null

curl -fsS -D "$SMOKE_TMP/delete.headers" -b "$SMOKE_JAR" -c "$SMOKE_JAR" \
  -X POST "$SMOKE_BASE/me/delete" \
  --data-urlencode "_csrf=$SMOKE_CSRF" \
  --data-urlencode "confirmation_name=$SMOKE_NAME" \
  -o /dev/null
grep -q 'geo_assessment_session=' "$SMOKE_TMP/delete.headers"

printf '[OK] HTTP 冒烟测试通过：首页、身份、答题、交卷、报告、切换确认、删除与安全头\n'
