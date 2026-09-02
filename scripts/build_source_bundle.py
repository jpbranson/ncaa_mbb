"""Regenerate docs/cbbwp-source.md, the disaster-recovery source bundle.

The bundle is a mirror: the folder plus its git history is the source of truth.
It exists so the whole project can be rebuilt from the project docs alone if a
machine is lost. That only works if it is regenerated when the source changes,
which is why this is a script and not a manual paste.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "cbbwp-source.md"

INCLUDE = [
    ("pyproject.toml", "toml"),
    ("README.md", "markdown"),
]
TREES = [("src/cbbwp", "py"), ("scripts", "py"), ("tests", "py")]
SKIP = {"__pycache__", "build_source_bundle.py"}


def files() -> list[tuple[Path, str]]:
    out = [(ROOT / n, lang) for n, lang in INCLUDE if (ROOT / n).exists()]
    for tree, lang in TREES:
        for p in sorted((ROOT / tree).rglob("*.py")):
            if any(part in SKIP for part in p.parts) or p.name in SKIP:
                continue
            out.append((p, lang))
    return out


def main() -> None:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        head = "unknown"
    parts = [
        "# `cbbwp` source bundle",
        # The absolute path is deliberately not recorded: this bundle is copied
        # between machines, and a path from whichever one last regenerated it is
        # noise at best and misleading at worst.
        f"Complete source. Regenerated {time.strftime('%Y-%m-%d %H:%M')} from the "
        f"`{ROOT.name}` working folder, at commit `{head}`.",
        "",
        "State rules v2, model v2. This bundle is a mirror for disaster recovery; the "
        "folder is the source of truth (it also holds the data, the fitted model and the "
        "git history). Regenerate with `python3 scripts/build_source_bundle.py` whenever "
        "the source changes.",
        "",
        "See `cbbwp-EXPLAIN.md` for what every piece does and why.",
        "",
        "---",
        "",
    ]
    for path, lang in files():
        rel = path.relative_to(ROOT)
        parts += [f"## `{rel}`", "", f"```{lang}", path.read_text().rstrip(), "```", ""]
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT} — {len(files())} files, {OUT.stat().st_size/1000:.0f} KB")


if __name__ == "__main__":
    main()
