---
name: geo-citation-lab
description: Install, open, download, or deploy GEO Citation Lab from its GitHub repository. Use when a user asks to install the lightweight offline viewer, open the research reports locally, download the full GEO citation dataset, prepare the analysis environment, or publish the viewer to GitHub Pages.
---

# GEO Citation Lab

Use the repository distribution contract to choose the smallest safe delivery mode and report the
result in plain language.

## Route the request

- For “安装”, “打开”, or “本地阅读”, install the lightweight Viewer.
- For “部署” or “GitHub Pages”, confirm the GitHub target before making external changes.
- For “完整数据” or “分析环境”, state the approximate size and clone the full repository.
- For reading only, open `https://yaojingang.github.io/geo-citation-lab/`.

Read `deploy/manifest.json` from the repository root. If no checkout is available, fetch
`https://raw.githubusercontent.com/yaojingang/geo-citation-lab/main/deploy/manifest.json`.
Treat the manifest as the source of asset names, versions, size limits, and entrypoints.

## Install the Viewer

1. Require `release_status: released`, then download the version-pinned asset and matching
   installer. The raw repository copies of `scripts/install.sh` and `scripts/install.ps1` are the
   source fallback.
2. Inspect the installer before execution.
3. Run `scripts/install.sh` on macOS/Linux or `scripts/install.ps1` on Windows.
4. Let the installer download and verify the latest ZIP and SHA-256 sidecar.
5. Open the returned `index.html` unless the user requested download only.
6. Report `status`, `install_path`, `entrypoint`, and `backup_path` when present.

Never pipe an unreviewed remote script into a shell. Preserve an existing installation unless its
directory contains `.geo-citation-lab-install`.

If `release_status` is `pending` or the pinned Release is unavailable, report that remote
installation is not published. A maintainer inside a checkout may build local assets with
`python3 scripts/build_viewer.py --package`.

## Deploy to GitHub Pages

1. Resolve the authenticated GitHub identity and exact target repository.
2. Read `deployment.requires_license_compliance` and `required_notice_files`. Preserve every
   required notice file, the requested attribution, and an indication of changes to CC BY 4.0
   content.
3. Confirm the target and public visibility.
4. Run `python3 scripts/verify_distribution.py`.
5. Confirm before changing an existing Pages publishing source.
6. Use `.github/workflows/publish.yml` to deploy the generated Viewer.
7. Read the completed workflow state and return its public URL.

Repository creation, visibility changes, Pages source changes, tags, pushes, and releases require
explicit user approval.

## Download full data

State that the complete working tree is approximately 568 MiB. Clone the repository only after the
user requests full data or analysis. For `03-cn-geo-citation-dataset`, follow its README and use
Python 3.11 or newer with `uv`.

Read `LICENSE`, `LICENSE-CODE`, `LICENSE-CONTENT`, and `THIRD_PARTY_NOTICES.md` before
redistribution. Code is MIT licensed, original GEO Citation Lab content is CC BY 4.0, and
third-party materials retain their source terms.
