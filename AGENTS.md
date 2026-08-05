# GEO Citation Lab agent instructions

## Intent routing

Map natural-language requests to one of these modes:

- `install`, `open`, `本地安装`, `打开`: install the lightweight Viewer.
- `deploy`, `发布`, `GitHub Pages`: publish the Viewer after confirming the external target.
- `full data`, `完整数据`, `分析环境`: clone the complete repository and prepare the nested Python project.
- `read`, `browse`, `看看报告`: open the canonical site when no local copy is required.

Use `deploy/manifest.json` as the machine-readable source of asset names, version, size limits,
entrypoint, and deployment workflow.

## Lightweight Viewer installation

1. Require `release_status` to be `released`, then use the version-pinned Release assets listed
   in `deploy/manifest.json`.
2. Download both the Viewer ZIP and its SHA-256 sidecar.
3. Inspect the installer before execution. Never pipe an unreviewed remote script into a shell.
4. Use `scripts/install.sh` on macOS/Linux and `scripts/install.ps1` on Windows.
5. Keep the default installation directory unless the user provides a destination.
6. Open `index.html` after installation unless the user asks for download only.
7. Report the installed path, entrypoint, checksum result, and any backup path.

The installer must remain idempotent. An update may replace only a directory containing
`.geo-citation-lab-install`. Preserve the previous installation as a timestamped backup.

If `release_status` is `pending` or the pinned Release does not exist, explain that the remote
package has not been published. A maintainer working inside a repository checkout may run
`python3 scripts/build_viewer.py --package` and install from `dist/release`.

## GitHub Pages deployment

Treat repository creation, visibility changes, Pages source changes, tags, pushes, and releases as
external mutations.

1. Resolve the authenticated GitHub identity and exact `OWNER/REPO`.
2. Read `deployment.requires_license_compliance` and `required_notice_files` from the manifest.
   Keep every required notice file in the deployed Viewer, preserve the requested attribution, and
   identify any changes made to CC BY 4.0 content.
3. Show the target repository and confirm public visibility before creating or changing it.
4. Confirm before switching an existing Pages source to GitHub Actions.
5. Run `python3 scripts/verify_distribution.py`.
6. Use `.github/workflows/publish.yml`; it publishes `dist/viewer`, not the repository root.
7. After deployment, read the workflow conclusion and open the returned Pages URL.
8. Report the public URL, workflow run, source commit, release tag when applicable, and retained
   license notices.

The canonical repository uses `https://yaojingang.github.io/geo-citation-lab/`.

## Full repository mode

The complete working tree is approximately 568 MiB and contains large research assets. State the
expected size before cloning. Read the directory-level README for the selected research line.
Python analysis for `03-cn-geo-citation-dataset` requires Python 3.11 or newer and `uv`.

Review `LICENSE`, `LICENSE-CODE`, `LICENSE-CONTENT`, and `THIRD_PARTY_NOTICES.md` before
redistribution. Code is MIT licensed, original GEO Citation Lab content is CC BY 4.0, and paper
PDF rights remain with their authors or publishers.

## Maintainer verification

Run these commands before a Pages deployment or Release:

```bash
python3 -m py_compile scripts/build_viewer.py scripts/verify_distribution.py
bash -n scripts/install.sh
python3 scripts/verify_distribution.py
```

For a release tag, require `v<distribution_version>` from `deploy/manifest.json`. The release ZIP
must stay within `modes.viewer.max_uncompressed_mib` and must exclude PDF, Parquet, DuckDB, JSONL,
and font files. Set `release_status` to `released` in the tagged commit; leave it as `pending`
until the release is ready.

## GEO Assessment subproject

`geo-assessment/` is an independent PHP and SQLite application. Keep runtime compatibility with
PHP 7.3.5 and SQLite 3.24 while running development tests on PHP 8.2 or newer. Public source must
leave `GEO_BAIDU_ANALYTICS_ID` empty; deployment examples may enable a disclosed site-specific ID.

Before changing or releasing the assessment, run from `geo-assessment/`:

```bash
composer validate --strict
composer verify
bash tools/build-release.sh "assessment-v$(cat VERSION)"
bash tools/verify-release.sh "dist/geo-assessment-$(cat VERSION).zip"
```

Assessment releases use `assessment-v*` tags so they do not trigger the Viewer `v*` workflow.
The release archive is generated from `tools/release-files.txt`, includes production Composer
autoloading, and excludes tests, development dependencies, databases, keys, logs, backups, paper
PDFs and research datasets. Preserve `LICENSE`, `LICENSE-CODE`, `LICENSE-CONTENT` and
`THIRD_PARTY_NOTICES.md` in every archive.
