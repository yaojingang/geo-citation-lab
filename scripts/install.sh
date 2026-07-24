#!/usr/bin/env bash
set -euo pipefail

ASSET_BASE="https://github.com/yaojingang/geo-citation-lab/releases/download/v0.1.0"
INSTALL_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/geo-citation-lab/viewer"
NO_OPEN=0
ARCHIVE_NAME="geo-citation-lab-viewer.zip"
CHECKSUM_NAME="${ARCHIVE_NAME}.sha256"
INSTALL_MARKER=".geo-citation-lab-install"
CHECKSUM_MARKER=".installed-checksum"
FILES_MANIFEST="geo-citation-lab-files.sha256"
FILES_CHECKSUM_MARKER=".installed-files-checksum"

usage() {
  command cat <<'EOF'
Install the lightweight GEO Citation Lab viewer.

Usage:
  bash install.sh [--dir PATH] [--asset-base URL_OR_PATH] [--no-open]

Options:
  --dir PATH        Installation directory.
  --asset-base SRC  Release URL or local release-asset directory.
  --no-open         Install without opening the viewer in a browser.
  --help            Show this help text.
EOF
}

# Install the lightweight GEO Citation Lab viewer.
#
# Usage:
#   bash install.sh [--dir PATH] [--asset-base URL_OR_PATH] [--no-open]
#
# Options:
#   --dir PATH        Installation directory.
#   --asset-base SRC  Release URL or local release-asset directory.
#   --no-open         Install without opening the viewer in a browser.
#   --help            Show this help text.

while (($#)); do
  case "$1" in
    --dir)
      [[ $# -ge 2 ]] || { echo "error: --dir requires a path" >&2; exit 2; }
      INSTALL_DIR="$2"
      shift 2
      ;;
    --asset-base)
      [[ $# -ge 2 ]] || { echo "error: --asset-base requires a source" >&2; exit 2; }
      ASSET_BASE="$2"
      shift 2
      ;;
    --no-open)
      NO_OPEN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

install_name="$(basename "$INSTALL_DIR")"
install_parent="$(dirname "$INSTALL_DIR")"
if [[ -z "$install_name" || "$install_name" == "." || "$install_name" == ".." ]]; then
  echo "error: unsafe installation directory: $INSTALL_DIR" >&2
  exit 2
fi
mkdir -p "$install_parent"
install_parent="$(cd "$install_parent" && pwd -P)"
INSTALL_DIR="$install_parent/$install_name"

if [[ "$INSTALL_DIR" == "/" || "$INSTALL_DIR" == "$HOME" ]]; then
  echo "error: refusing to install into a broad system or home directory" >&2
  exit 2
fi
if [[ -e "$INSTALL_DIR" && ! -d "$INSTALL_DIR" ]]; then
  echo "error: installation target exists and is not a directory: $INSTALL_DIR" >&2
  exit 2
fi
if [[ -d "$INSTALL_DIR" && ! -f "$INSTALL_DIR/$INSTALL_MARKER" ]]; then
  echo "error: existing directory is not managed by this installer: $INSTALL_DIR" >&2
  exit 2
fi

temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/geo-citation-lab.XXXXXX")"
stage_dir="${INSTALL_DIR}.new.$$"
backup_dir=""

cleanup() {
  rm -rf -- "$temporary_dir"
  if [[ -d "$stage_dir" ]]; then
    rm -rf -- "$stage_dir"
  fi
}
trap cleanup EXIT

fetch_asset() {
  local name="$1"
  local destination="$2"
  local base="${ASSET_BASE%/}"
  case "$base" in
    http://*|https://*)
      command -v curl >/dev/null 2>&1 || {
        echo "error: curl is required for remote installation" >&2
        return 1
      }
      curl --fail --location --retry 3 --silent --show-error \
        "$base/$name" --output "$destination"
      ;;
    file://*)
      cp -- "${base#file://}/$name" "$destination"
      ;;
    *)
      cp -- "$base/$name" "$destination"
      ;;
  esac
}

fetch_asset "$ARCHIVE_NAME" "$temporary_dir/$ARCHIVE_NAME"
fetch_asset "$CHECKSUM_NAME" "$temporary_dir/$CHECKSUM_NAME"

if command -v sha256sum >/dev/null 2>&1; then
  HASH_TOOL="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  HASH_TOOL="shasum"
else
  echo "error: sha256sum or shasum is required to verify the download" >&2
  exit 1
fi

sha256_file() {
  local path="$1"
  if [[ "$HASH_TOOL" == "sha256sum" ]]; then
    sha256sum "$path" | awk '{ print $1 }'
  else
    shasum -a 256 "$path" | awk '{ print $1 }'
  fi
}

verify_file_tree() {
  local root="$1"
  local require_marker="$2"
  local manifest="$root/$FILES_MANIFEST"
  local marker="$root/$FILES_CHECKSUM_MARKER"
  local manifest_digest
  local marker_digest
  local checksum_line
  local expected_digest
  local relative_path
  local actual_digest
  local file_count=0

  [[ -f "$manifest" && ! -L "$manifest" ]] || return 1
  manifest_digest="$(sha256_file "$manifest")"
  if [[ "$require_marker" == "1" ]]; then
    [[ -f "$marker" && ! -L "$marker" ]] || return 1
    marker_digest="$(tr -d '[:space:]' < "$marker")"
    [[ "$marker_digest" == "$manifest_digest" ]] || return 1
  fi

  while IFS= read -r checksum_line || [[ -n "$checksum_line" ]]; do
    expected_digest="${checksum_line%%  *}"
    relative_path="${checksum_line#*  }"
    [[ "$expected_digest" =~ ^[0-9a-f]{64}$ ]] || return 1
    [[ -n "$relative_path" && "$relative_path" != "$checksum_line" ]] || return 1
    case "$relative_path" in
      /*|..|../*|*/..|*/../*|*\\*)
        return 1
        ;;
    esac
    [[ -f "$root/$relative_path" && ! -L "$root/$relative_path" ]] || return 1
    actual_digest="$(sha256_file "$root/$relative_path")"
    [[ "$actual_digest" == "$expected_digest" ]] || return 1
    ((file_count += 1))
  done < "$manifest"

  ((file_count > 0))
}

expected_checksum="$(awk 'NR == 1 { print $1 }' "$temporary_dir/$CHECKSUM_NAME")"
if [[ ! "$expected_checksum" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "error: release checksum is malformed" >&2
  exit 1
fi

actual_checksum="$(sha256_file "$temporary_dir/$ARCHIVE_NAME")"

actual_checksum="$(printf '%s' "$actual_checksum" | tr '[:upper:]' '[:lower:]')"
expected_checksum="$(printf '%s' "$expected_checksum" | tr '[:upper:]' '[:lower:]')"
if [[ "$actual_checksum" != "$expected_checksum" ]]; then
  echo "error: release checksum verification failed" >&2
  exit 1
fi

if [[ -f "$INSTALL_DIR/index.html" ]] &&
   [[ -f "$INSTALL_DIR/geo-citation-lab-manifest.json" ]] &&
   [[ -f "$INSTALL_DIR/$CHECKSUM_MARKER" ]] &&
   [[ "$(tr -d '[:space:]' < "$INSTALL_DIR/$CHECKSUM_MARKER")" == "$expected_checksum" ]] &&
   verify_file_tree "$INSTALL_DIR" 1; then
  entrypoint="$INSTALL_DIR/index.html"
  installed_version="$(
    sed -n 's/^[[:space:]]*"distribution_version":[[:space:]]*"\([^"]*\)".*/\1/p' \
      "$INSTALL_DIR/geo-citation-lab-manifest.json" | head -n 1
  )"
  echo "status=already-installed"
  echo "distribution_version=${installed_version:-unknown}"
  echo "install_path=$INSTALL_DIR"
  echo "entrypoint=$entrypoint"
  if ((NO_OPEN == 0)); then
    if command -v open >/dev/null 2>&1; then
      open "$entrypoint" || echo "open_manually=$entrypoint"
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$entrypoint" || echo "open_manually=$entrypoint"
    elif command -v cmd.exe >/dev/null 2>&1; then
      cmd.exe /C start "" "$(cygpath -w "$entrypoint" 2>/dev/null || printf '%s' "$entrypoint")" ||
        echo "open_manually=$entrypoint"
    fi
  fi
  exit 0
fi

command -v unzip >/dev/null 2>&1 || {
  echo "error: unzip is required to install the viewer" >&2
  exit 1
}

archive_size="$(unzip -l "$temporary_dir/$ARCHIVE_NAME" | awk 'END { print $1 }')"
if [[ ! "$archive_size" =~ ^[0-9]+$ ]] || ((archive_size > 15 * 1024 * 1024)); then
  echo "error: release archive exceeds the 15 MiB extraction limit" >&2
  exit 1
fi
while IFS= read -r archive_entry; do
  case "$archive_entry" in
    /*|../*|*/../*|*\\..\\*)
      echo "error: release archive contains an unsafe path: $archive_entry" >&2
      exit 1
      ;;
  esac
done < <(unzip -Z1 "$temporary_dir/$ARCHIVE_NAME")

if unzip -Z -l "$temporary_dir/$ARCHIVE_NAME" |
   awk '$1 ~ /^l/ { found = 1 } END { exit(found ? 0 : 1) }'; then
  echo "error: release archive contains a symbolic link" >&2
  exit 1
fi

if [[ -e "$stage_dir" ]]; then
  echo "error: temporary installation path already exists: $stage_dir" >&2
  exit 1
fi
mkdir -p "$stage_dir"
unzip -q "$temporary_dir/$ARCHIVE_NAME" -d "$stage_dir"

if find "$stage_dir" -type l -print -quit | grep -q .; then
  echo "error: release archive extracted a symbolic link" >&2
  exit 1
fi
if [[ ! -f "$stage_dir/index.html" ]]; then
  echo "error: release archive does not contain index.html" >&2
  exit 1
fi
if ! verify_file_tree "$stage_dir" 0; then
  echo "error: release archive file integrity verification failed" >&2
  exit 1
fi
distribution_version="$(
  sed -n 's/^[[:space:]]*"distribution_version":[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$stage_dir/geo-citation-lab-manifest.json" | head -n 1
)"
if [[ ! "$distribution_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: release manifest has no valid distribution version" >&2
  exit 1
fi
printf '%s\n' "managed-by=geo-citation-lab-installer" > "$stage_dir/$INSTALL_MARKER"
printf '%s\n' "$expected_checksum" > "$stage_dir/$CHECKSUM_MARKER"
printf '%s\n' "$(sha256_file "$stage_dir/$FILES_MANIFEST")" \
  > "$stage_dir/$FILES_CHECKSUM_MARKER"

if [[ -d "$INSTALL_DIR" ]]; then
  backup_dir="${INSTALL_DIR}.previous.$(date -u +%Y%m%dT%H%M%SZ).$$"
  mv -- "$INSTALL_DIR" "$backup_dir"
fi

if ! mv -- "$stage_dir" "$INSTALL_DIR"; then
  if [[ -n "$backup_dir" && -d "$backup_dir" ]]; then
    mv -- "$backup_dir" "$INSTALL_DIR"
  fi
  echo "error: installation failed; the previous installation was restored" >&2
  exit 1
fi

entrypoint="$INSTALL_DIR/index.html"
echo "status=installed"
echo "distribution_version=$distribution_version"
echo "install_path=$INSTALL_DIR"
echo "entrypoint=$entrypoint"
if [[ -n "$backup_dir" ]]; then
  echo "backup_path=$backup_dir"
fi

if ((NO_OPEN == 0)); then
  if command -v open >/dev/null 2>&1; then
    open "$entrypoint" || echo "open_manually=$entrypoint"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$entrypoint" || echo "open_manually=$entrypoint"
  elif command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /C start "" "$(cygpath -w "$entrypoint" 2>/dev/null || printf '%s' "$entrypoint")" ||
      echo "open_manually=$entrypoint"
  else
    echo "open_manually=$entrypoint"
  fi
fi
