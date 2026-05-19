#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert final_report.md to a single self-contained HTML file with images as base64."""
from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_vendor = _repo_root / ".vendor"
if _vendor.is_dir():
    sys.path.insert(0, str(_vendor))

try:
    import markdown
except ImportError:
    print("Missing dependency: pip install markdown", file=sys.stderr)
    raise SystemExit(1)


def data_uri_for_file(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "application/octet-stream"
    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def resolve_local_asset(md_dir: Path, ref: str) -> Path | None:
    ref = ref.strip().replace("\\", "/")
    if not ref or ref.startswith(("http://", "https://", "data:", "#")):
        return None
    md_dir = md_dir.resolve()
    candidates: list[Path] = [
        md_dir / ref,
        md_dir / ref.lstrip("/"),
    ]
    if ref.startswith("report/"):
        candidates.append(md_dir / ref[len("report/") :])
    candidates.append(md_dir / "image" / Path(ref).name)
    for c in candidates:
        try:
            c = c.resolve()
        except OSError:
            continue
        if not c.is_file():
            continue
        try:
            c.relative_to(md_dir)
        except ValueError:
            continue
        return c
    return None


def inline_images_in_markdown(md: str, md_dir: Path) -> tuple[str, list[str]]:
    """Replace local image refs with data URIs. Returns (new_md, warnings)."""
    cache: dict[str, str] = {}
    warnings: list[str] = []

    def to_data_uri(ref: str) -> str:
        ref = ref.strip()
        if ref.startswith(("http://", "https://", "data:")):
            return ref
        if ref in cache:
            return cache[ref]
        p = resolve_local_asset(md_dir, ref)
        if p is None:
            warnings.append(f"missing file for ref: {ref!r}")
            cache[ref] = ref
            return ref
        cache[ref] = data_uri_for_file(p)
        return cache[ref]

    def repl_mdimg(m: re.Match[str]) -> str:
        alt, url = m.group(1), m.group(2)
        return f"![{alt}]({to_data_uri(url)})"

    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl_mdimg, md)

    def repl_img_src(m: re.Match[str]) -> str:
        pre, url, post = m.group(1), m.group(2), m.group(3)
        return f'{pre}{to_data_uri(url)}{post}'

    md = re.sub(
        r'(\bsrc\s*=\s*")([^"]+)(")',
        repl_img_src,
        md,
        flags=re.IGNORECASE,
    )
    return md, warnings


def wrap_block_tables(rendered_html: str) -> str:
    """Wrap tables in a scroll container for small screens."""
    return re.sub(
        r"(<table>.*?</table>)",
        r'<div class="table-wrap">\1</div>',
        rendered_html,
        flags=re.DOTALL,
    )


HTML_SHELL = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --paper: #f7f3ea;
      --sheet: #fffdfa;
      --ink: #1a1d21;
      --muted: #5c6773;
      --rule: #d9d2c4;
      --accent: #1b365d;
      --accent-soft: #e8eef5;
      --code-bg: #f1ede5;
      --quote-bg: #f6f2ea;
    }}
    @page {{
      size: A4;
      margin: 18mm 18mm 20mm 18mm;
    }}
    html {{
      background: var(--paper);
      max-width: 100%;
      overflow-x: hidden;
    }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua",
        Georgia, "Noto Serif CJK SC", "Source Han Serif SC", "Songti SC", serif;
      line-height: 1.78;
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizeLegibility;
      max-width: 100%;
      overflow-x: hidden;
    }}
    main {{
      width: 100%;
      max-width: 980px;
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    article {{
      width: 100%;
      max-width: 100%;
      background: var(--sheet);
      border: 1px solid rgba(217, 210, 196, 0.9);
      box-shadow: 0 20px 60px rgba(36, 41, 46, 0.08);
      border-radius: 18px;
      padding: 40px 48px 52px;
    }}
    * {{
      box-sizing: border-box;
    }}
    img {{
      max-width: 100%;
      height: auto;
      display: block;
      margin: 1.2rem auto;
      border-radius: 10px;
    }}
    .table-wrap {{
      margin: 1.2em 0 1.5em;
      overflow-x: auto;
      overflow-y: hidden;
      -webkit-overflow-scrolling: touch;
      border: 1px solid rgba(217, 210, 196, 0.92);
      border-radius: 12px;
      background: white;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 640px;
      margin: 0;
      font-size: 0.95em;
      background: white;
    }}
    th, td {{
      border: 1px solid var(--rule);
      padding: 0.58em 0.72em;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
    }}
    tr:nth-child(even) td {{
      background: rgba(250, 248, 242, 0.85);
    }}
    pre, code {{
      font-family: ui-monospace, "Cascadia Code", "SF Mono", Menlo, monospace;
      font-size: 0.88em;
    }}
    pre {{
      background: var(--code-bg);
      padding: 1em 1.1em;
      overflow-x: auto;
      border-radius: 10px;
      border: 1px solid var(--rule);
    }}
    code {{
      background: rgba(27, 54, 93, 0.08);
      padding: 0.12em 0.32em;
      border-radius: 4px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    pre code {{
      background: transparent;
      padding: 0;
    }}
    blockquote {{
      margin: 1.2em 0 1.4em;
      padding: 0.45em 1em 0.45em 1.2em;
      border-left: 4px solid var(--accent);
      color: var(--muted);
      background: var(--quote-bg);
      border-radius: 0 8px 8px 0;
    }}
    hr {{
      border: 0;
      border-top: 1px solid var(--rule);
      margin: 2rem 0;
    }}
    h1, h2, h3, h4 {{
      color: var(--ink);
      line-height: 1.3;
      page-break-after: avoid;
    }}
    h1 {{
      font-size: 2rem;
      color: var(--accent);
      margin: 0 0 0.9rem;
      padding-bottom: 0.55rem;
      border-bottom: 2px solid var(--rule);
      letter-spacing: 0.01em;
    }}
    h2 {{
      font-size: 1.42rem;
      margin-top: 2.1rem;
      margin-bottom: 0.9rem;
      padding-left: 0.7rem;
      border-left: 4px solid var(--accent);
    }}
    h3 {{
      font-size: 1.12rem;
      margin-top: 1.5rem;
      margin-bottom: 0.55rem;
      color: var(--accent);
    }}
    h4 {{
      font-size: 1rem;
      margin-top: 1.2rem;
      margin-bottom: 0.45rem;
    }}
    p {{
      margin: 0.75rem 0;
    }}
    ul, ol {{
      margin: 0.8rem 0 1rem 1.4rem;
      padding: 0;
    }}
    li {{
      margin: 0.34rem 0;
    }}
    strong {{
      color: var(--ink);
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      border-bottom: 1px solid rgba(27, 54, 93, 0.28);
    }}
    p, li, blockquote, td, th, h1, h2, h3, h4 {{
      overflow-wrap: anywhere;
      word-break: break-word;
      white-space: normal;
    }}
    @media (max-width: 900px) {{
      main {{
        padding: 18px 14px 34px;
      }}
      article {{
        padding: 28px 24px 34px;
      }}
    }}
    @media (max-width: 640px) {{
      body {{
        font-size: 15.5px;
        line-height: 1.72;
      }}
      main {{
        padding: 0;
      }}
      article {{
        border: 0;
        box-shadow: none;
        border-radius: 0;
        padding: 18px 16px 28px;
      }}
      article > *:not(.table-wrap) {{
        max-width: calc(100vw - 32px);
      }}
      h1 {{
        font-size: 1.56rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.45rem;
      }}
      h2 {{
        font-size: 1.18rem;
        margin-top: 1.7rem;
        margin-bottom: 0.72rem;
        padding-left: 0.58rem;
      }}
      h3 {{
        font-size: 1.02rem;
        margin-top: 1.25rem;
      }}
      h4 {{
        font-size: 0.96rem;
      }}
      pre {{
        margin-left: 0;
        margin-right: 0;
        padding: 0.88em 0.92em;
        border-radius: 8px;
      }}
      blockquote {{
        margin: 1rem 0 1.1rem;
        padding: 0.42em 0.88em 0.42em 1em;
      }}
      ul, ol {{
        margin-left: 1.1rem;
      }}
      .table-wrap {{
        margin: 1rem -4px 1.2rem;
        border-radius: 10px;
      }}
      table {{
        min-width: 560px;
        font-size: 0.91em;
      }}
      th, td {{
        padding: 0.52em 0.58em;
      }}
    }}
    @media print {{
      html, body {{
        background: white;
      }}
      main {{
        max-width: none;
        padding: 0;
      }}
      article {{
        border: 0;
        box-shadow: none;
        border-radius: 0;
        padding: 0;
      }}
      .table-wrap {{
        overflow: visible;
        border: 0;
        box-shadow: none;
        border-radius: 0;
        margin: 1.2em 0 1.5em;
      }}
      table, pre, blockquote, img {{
        break-inside: avoid;
      }}
      table {{
        min-width: 0;
      }}
    }}
  </style>
</head>
<body>
<main>
  <article>
{body}
  </article>
</main>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Build self-contained HTML from Markdown.")
    p.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "final_report.md")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "final_report.html")
    args = p.parse_args()

    md_path = args.input.resolve()
    if not md_path.is_file():
        raise SystemExit(f"Input not found: {md_path}")

    md_dir = md_path.parent
    raw = md_path.read_text(encoding="utf-8")
    md_inlined, warns = inline_images_in_markdown(raw, md_dir)
    for w in warns:
        print(f"[warn] {w}", file=sys.stderr)

    body = markdown.markdown(
        md_inlined,
        extensions=["tables", "fenced_code", "nl2br"],
        output_format="html",
    )
    body = wrap_block_tables(body)
    title = "Report"
    m = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    if m:
        title = m.group(1).strip()

    out = HTML_SHELL.format(title=html.escape(title), body=body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out, encoding="utf-8")
    print(f"Wrote {args.output} ({len(out):,} bytes)")


if __name__ == "__main__":
    main()
