#!/usr/bin/env python3
"""Verify viewer contents, release assets, links, and local installation."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import stat
import subprocess
import tempfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

from build_viewer import _javascript_string_literal


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_viewer.py"
MANIFEST_PATH = REPO_ROOT / "deploy" / "manifest.json"
REQUIRED_FILES = {
    "index.html",
    ".nojekyll",
    "geo-citation-lab-manifest.json",
    "geo-citation-lab-files.sha256",
    "LICENSE",
    "LICENSE-CODE",
    "LICENSE-CONTENT",
    "THIRD_PARTY_NOTICES.md",
    "01-geo-experiment-data-report/04-repet/final_report.html",
    "02-geo-aeo-ai-search-papers/index.html",
    "02-geo-aeo-ai-search-papers/00_资料说明/论文清单.csv",
    "03-cn-geo-citation-dataset/reports/final/CN-GEO_多维数据分析报告.html",
}
BANNED_SUFFIXES = {".pdf", ".parquet", ".duckdb", ".jsonl", ".ttf"}


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.links.append(value)


def _local_link_target(html_file: Path, value: str) -> Path | None:
    if not value or value.startswith(("#", "data:", "mailto:", "javascript:")):
        return None
    if "${" in value:
        return None
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc or value.startswith("/"):
        return None
    path_text = unquote(parsed.path)
    if not path_text:
        return None
    target = (html_file.parent / path_text).resolve()
    if path_text.endswith("/"):
        target = target / "index.html"
    return target


def _verify_links(viewer: Path) -> None:
    missing: list[str] = []
    for html_file in viewer.rglob("*.html"):
        parser = LinkCollector()
        parser.feed(html_file.read_text(encoding="utf-8"))
        for value in parser.links:
            target = _local_link_target(html_file, value)
            if target is not None and not target.exists():
                missing.append(f"{html_file.relative_to(viewer)} -> {value}")
    if missing:
        raise AssertionError("Missing local viewer links:\n" + "\n".join(missing))


def _verify_archive(viewer: Path, package_dir: Path, manifest: dict) -> None:
    package_files = {
        path.name for path in package_dir.iterdir() if path.is_file()
    }
    assert package_files == set(manifest["release_assets"]), (
        "Release package files differ from manifest release_assets"
    )
    archive = package_dir / "geo-citation-lab-viewer.zip"
    checksum_file = package_dir / "geo-citation-lab-viewer.zip.sha256"
    expected_checksum = checksum_file.read_text(encoding="utf-8").split()[0]
    actual_checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert actual_checksum == expected_checksum, "Release archive checksum mismatch"

    viewer_files = {
        path.relative_to(viewer).as_posix()
        for path in viewer.rglob("*")
        if path.is_file()
    }
    assert REQUIRED_FILES <= viewer_files, "Viewer is missing required files"
    banned = sorted(
        path for path in viewer_files if Path(path).suffix.lower() in BANNED_SUFFIXES
    )
    assert not banned, f"Viewer contains excluded binary assets: {banned}"
    noise = sorted(
        path
        for path in viewer_files
        if "__pycache__" in Path(path).parts
        or Path(path).name in {".DS_Store", ".pytest_cache"}
        or Path(path).suffix == ".pyc"
    )
    assert not noise, f"Viewer contains local cache files: {noise}"

    total_bytes = sum(
        path.stat().st_size for path in viewer.rglob("*") if path.is_file()
    )
    max_bytes = manifest["modes"]["viewer"]["max_uncompressed_mib"] * 1024 * 1024
    assert total_bytes <= max_bytes, (
        f"Viewer exceeds size limit: {total_bytes} > {max_bytes}"
    )

    with zipfile.ZipFile(archive) as bundle:
        archive_files = {name for name in bundle.namelist() if not name.endswith("/")}
        assert archive_files == viewer_files, "Archive and viewer file lists differ"
        for name in archive_files:
            parts = Path(name).parts
            assert not Path(name).is_absolute() and ".." not in parts, (
                f"Unsafe archive path: {name}"
            )

    file_manifest = viewer / "geo-citation-lab-files.sha256"
    listed_files: set[str] = set()
    for line in file_manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        assert separator and len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)
        assert relative not in listed_files
        listed_files.add(relative)
        target = viewer / relative
        assert target.is_file() and not target.is_symlink()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    assert listed_files == viewer_files - {"geo-citation-lab-files.sha256"}


def _verify_catalog(viewer: Path) -> None:
    catalog = (
        viewer / "02-geo-aeo-ai-search-papers" / "index.html"
    ).read_text(encoding="utf-8")
    assert "window.GEO_VIEWER_CSV =" in catalog, "Paper CSV was not embedded"
    assert "window.GEO_VIEWER_USE_SOURCE_PDF_URLS = true;" in catalog
    assert "const EMBEDDED_CSV = window.GEO_VIEWER_CSV || \"\";" in catalog
    assert "const USE_SOURCE_PDF_URLS =" in catalog
    assert "let csv = EMBEDDED_CSV;" in catalog
    assert "function safeExternalUrl(value)" in catalog
    assert 'parsed.protocol === "https:" || parsed.protocol === "http:"' in catalog
    assert 'escapeHtml(encodeURI(paper.filePath))' in catalog
    assert 'rel="noopener noreferrer"' in catalog

    csv_path = (
        viewer
        / "02-geo-aeo-ai-search-papers"
        / "00_资料说明"
        / "论文清单.csv"
    )
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert rows, "Paper catalog CSV is empty"
    for row in rows:
        parsed = urlparse(row["URL"])
        assert parsed.scheme in {"http", "https"} and parsed.netloc, (
            f"Paper catalog has an unsafe source URL: {row['URL']}"
        )

    hostile_literal = _javascript_string_literal(
        '</script><script>alert("x")</script>&>\u2028\u2029'
    )
    assert "</script" not in hostile_literal.lower()
    assert "\\u003c/script\\u003e" in hostile_literal
    assert "\\u0026" in hostile_literal
    assert "\\u2028" in hostile_literal and "\\u2029" in hostile_literal


def _verify_licenses(viewer: Path, manifest: dict) -> None:
    licensing = manifest["licensing"]
    license_files = {
        licensing["scope_file"],
        licensing["code"]["file"],
        licensing["original_content"]["file"],
        licensing["third_party_notices"],
    }
    assert license_files == set(manifest["deployment"]["required_notice_files"])
    for name in license_files:
        source = REPO_ROOT / name
        bundled = viewer / name
        assert source.is_file() and bundled.is_file()
        assert bundled.read_bytes() == source.read_bytes()

    mit_text = (REPO_ROOT / licensing["code"]["file"]).read_text(encoding="utf-8")
    assert mit_text.startswith("MIT License\n")
    assert "Permission is hereby granted, free of charge" in mit_text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in mit_text

    content_text = (REPO_ROOT / licensing["original_content"]["file"]).read_text(
        encoding="utf-8"
    )
    assert "Creative Commons Attribution 4.0 International" in content_text
    assert "https://creativecommons.org/licenses/by/4.0/" in content_text
    assert licensing["original_content"]["attribution"] in content_text.replace(
        '"', ""
    ).replace("\n> ", " ").replace("\n", " ")

    root_index = (viewer / "index.html").read_text(encoding="utf-8")
    catalog = (
        viewer / "02-geo-aeo-ai-search-papers" / "index.html"
    ).read_text(encoding="utf-8")
    for html in (root_index, catalog):
        assert 'rel="license"' in html
        assert "https://creativecommons.org/licenses/by/4.0/" in html


def _verify_skill() -> None:
    skill_dir = REPO_ROOT / "skills" / "geo-citation-lab"
    skill_text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    assert skill_text.startswith("---\nname: geo-citation-lab\n")
    assert "\ndescription: " in skill_text.split("---", 2)[1]
    assert "TODO" not in skill_text
    assert "deployment.requires_license_compliance" in skill_text
    assert "required_notice_files" in skill_text
    assert "release_status: released" in skill_text

    agent_instructions = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "deployment.requires_license_compliance" in agent_instructions
    assert "required_notice_files" in agent_instructions
    assert "release_status" in agent_instructions

    interface_text = (skill_dir / "agents" / "openai.yaml").read_text(
        encoding="utf-8"
    )
    assert 'display_name: "GEO Citation Lab"' in interface_text
    assert "$geo-citation-lab" in interface_text


def _run_shell_install(package_dir: Path, install_dir: Path) -> None:
    command = [
        "bash",
        str(REPO_ROOT / "scripts" / "install.sh"),
        "--asset-base",
        str(package_dir),
        "--dir",
        str(install_dir),
        "--no-open",
    ]
    first = subprocess.run(command, check=True, text=True, capture_output=True)
    assert "status=installed" in first.stdout
    assert "distribution_version=0.1.0" in first.stdout
    assert (install_dir / "index.html").is_file()

    second = subprocess.run(command, check=True, text=True, capture_output=True)
    assert "status=already-installed" in second.stdout
    assert "distribution_version=0.1.0" in second.stdout
    assert not list(install_dir.parent.glob(f"{install_dir.name}.previous.*"))

    report_relative = Path(
        "01-geo-experiment-data-report/04-repet/final_report.html"
    )
    (install_dir / report_relative).unlink()
    third = subprocess.run(command, check=True, text=True, capture_output=True)
    assert "status=installed" in third.stdout
    assert (install_dir / "index.html").is_file()
    assert (install_dir / report_relative).is_file()
    backups = list(install_dir.parent.glob(f"{install_dir.name}.previous.*"))
    assert len(backups) == 1
    assert (backups[0] / ".geo-citation-lab-install").is_file()
    assert (backups[0] / "index.html").is_file()
    assert not (backups[0] / report_relative).exists()


def _run_shell_rejection_tests(package_dir: Path, root: Path) -> None:
    command_prefix = [
        "bash",
        str(REPO_ROOT / "scripts" / "install.sh"),
        "--no-open",
    ]

    unmanaged = root / "unmanaged-target"
    unmanaged.mkdir()
    sentinel = unmanaged / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    unmanaged_result = subprocess.run(
        command_prefix
        + [
            "--asset-base",
            str(package_dir),
            "--dir",
            str(unmanaged),
        ],
        text=True,
        capture_output=True,
    )
    assert unmanaged_result.returncode != 0
    assert "is not managed by this installer" in unmanaged_result.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve"

    corrupt_assets = root / "corrupt-assets"
    shutil.copytree(package_dir, corrupt_assets)
    (corrupt_assets / "geo-citation-lab-viewer.zip.sha256").write_text(
        "0" * 64 + "  geo-citation-lab-viewer.zip\n",
        encoding="utf-8",
    )
    corrupt_target = root / "corrupt-target"
    corrupt_result = subprocess.run(
        command_prefix
        + [
            "--asset-base",
            str(corrupt_assets),
            "--dir",
            str(corrupt_target),
        ],
        text=True,
        capture_output=True,
    )
    assert corrupt_result.returncode != 0
    assert "checksum verification failed" in corrupt_result.stderr
    assert not corrupt_target.exists()

    unsafe_assets = root / "unsafe-assets"
    unsafe_assets.mkdir()
    unsafe_archive = unsafe_assets / "geo-citation-lab-viewer.zip"
    with zipfile.ZipFile(unsafe_archive, "w") as bundle:
        bundle.writestr("../escape.txt", "blocked")
        bundle.writestr("index.html", "<!doctype html>")
    unsafe_digest = hashlib.sha256(unsafe_archive.read_bytes()).hexdigest()
    (unsafe_assets / "geo-citation-lab-viewer.zip.sha256").write_text(
        f"{unsafe_digest}  geo-citation-lab-viewer.zip\n",
        encoding="utf-8",
    )
    unsafe_target = root / "unsafe-target"
    unsafe_result = subprocess.run(
        command_prefix
        + [
            "--asset-base",
            str(unsafe_assets),
            "--dir",
            str(unsafe_target),
        ],
        text=True,
        capture_output=True,
    )
    assert unsafe_result.returncode != 0
    assert "contains an unsafe path" in unsafe_result.stderr
    assert not (root / "escape.txt").exists()

    symlink_assets = root / "symlink-assets"
    symlink_assets.mkdir()
    symlink_archive = symlink_assets / "geo-citation-lab-viewer.zip"
    outside_marker = root / "outside-install-marker"
    outside_marker.write_text("preserve", encoding="utf-8")
    link_entry = zipfile.ZipInfo(".geo-citation-lab-install")
    link_entry.create_system = 3
    link_entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink_archive, "w") as bundle:
        bundle.writestr(link_entry, str(outside_marker))
        bundle.writestr("index.html", "<!doctype html>")
        bundle.writestr(
            "geo-citation-lab-manifest.json",
            '{"distribution_version":"0.1.0"}',
        )
    symlink_digest = hashlib.sha256(symlink_archive.read_bytes()).hexdigest()
    (symlink_assets / "geo-citation-lab-viewer.zip.sha256").write_text(
        f"{symlink_digest}  geo-citation-lab-viewer.zip\n",
        encoding="utf-8",
    )
    symlink_target = root / "symlink-target"
    symlink_result = subprocess.run(
        command_prefix
        + [
            "--asset-base",
            str(symlink_assets),
            "--dir",
            str(symlink_target),
        ],
        text=True,
        capture_output=True,
    )
    assert symlink_result.returncode != 0
    assert "contains a symbolic link" in symlink_result.stderr
    assert outside_marker.read_text(encoding="utf-8") == "preserve"
    assert not symlink_target.exists()


def _run_powershell_install_if_available(
    package_dir: Path, install_dir: Path
) -> bool:
    executable = shutil.which("pwsh")
    if not executable:
        return False
    command = [
        executable,
        "-NoProfile",
        "-File",
        str(REPO_ROOT / "scripts" / "install.ps1"),
        "-AssetBase",
        str(package_dir),
        "-InstallDir",
        str(install_dir),
        "-NoOpen",
    ]
    first = subprocess.run(command, check=True, text=True, capture_output=True)
    assert "status=installed" in first.stdout
    assert "distribution_version=0.1.0" in first.stdout
    second = subprocess.run(command, check=True, text=True, capture_output=True)
    assert "status=already-installed" in second.stdout
    assert "distribution_version=0.1.0" in second.stdout
    report_relative = Path(
        "01-geo-experiment-data-report/04-repet/final_report.html"
    )
    (install_dir / report_relative).unlink()
    third = subprocess.run(command, check=True, text=True, capture_output=True)
    assert "status=installed" in third.stdout
    assert (install_dir / "index.html").is_file()
    assert (install_dir / report_relative).is_file()
    assert len(list(install_dir.parent.glob(f"{install_dir.name}.previous.*"))) == 1
    return True


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["default_mode"] == "viewer"
    assert manifest["distribution_version"] == "0.1.0"
    assert manifest["release_status"] in {"pending", "released"}
    assert manifest["modes"]["viewer"]["runtime_requirements"] == []
    version = manifest["distribution_version"]
    release_base = (
        "https://github.com/yaojingang/geo-citation-lab/releases/download/"
        f"v{version}"
    )
    assert manifest["modes"]["viewer"]["asset"].startswith(release_base + "/")
    assert manifest["modes"]["viewer"]["checksum"].startswith(release_base + "/")
    assert manifest["licensing"]["code"]["spdx"] == "MIT"
    assert manifest["licensing"]["original_content"]["spdx"] == "CC-BY-4.0"
    assert manifest["deployment"]["requires_redistribution_authorization"] is False
    assert manifest["deployment"]["requires_license_compliance"] is True
    assert set(manifest["deployment"]["required_notice_files"]) <= REQUIRED_FILES
    shell_installer = (REPO_ROOT / "scripts" / "install.sh").read_text(
        encoding="utf-8"
    )
    powershell_installer = (REPO_ROOT / "scripts" / "install.ps1").read_text(
        encoding="utf-8"
    )
    assert f'ASSET_BASE="{release_base}"' in shell_installer
    assert f'[string]$AssetBase = "{release_base}"' in powershell_installer
    max_mib = manifest["modes"]["viewer"]["max_uncompressed_mib"]
    assert f"archive_size > {max_mib} * 1024 * 1024" in shell_installer
    assert f"$uncompressedSize -gt ({max_mib} * 1024 * 1024)" in powershell_installer
    _verify_skill()

    with tempfile.TemporaryDirectory(prefix="geo-citation-lab-verify-") as temporary:
        root = Path(temporary)
        viewer = root / "viewer"
        package_dir = root / "release"
        subprocess.run(
            [
                "python3",
                str(BUILD_SCRIPT),
                "--output",
                str(viewer),
                "--package",
                "--package-dir",
                str(package_dir),
            ],
            check=True,
        )

        _verify_links(viewer)
        _verify_catalog(viewer)
        _verify_licenses(viewer, manifest)
        _verify_archive(viewer, package_dir, manifest)
        _run_shell_install(package_dir, root / "shell-install")
        _run_shell_rejection_tests(package_dir, root)
        powershell_verified = _run_powershell_install_if_available(
            package_dir, root / "powershell-install"
        )

        total_bytes = sum(
            path.stat().st_size for path in viewer.rglob("*") if path.is_file()
        )
        print("viewer_links=pass")
        print("viewer_catalog=pass")
        print("viewer_licenses=pass")
        print("agent_skill=pass")
        print("viewer_archive=pass")
        print("shell_install=pass")
        print("shell_rejection_tests=pass")
        print(
            "powershell_install=pass"
            if powershell_verified
            else "powershell_install=skipped-pwsh-unavailable"
        )
        print(f"viewer_uncompressed_bytes={total_bytes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
