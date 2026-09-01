# docs/

| File | What it is |
|---|---|
| `cbbwp-EXPLAIN.md` | **Read this first.** The whole model in plain language: a glossary of every mathematical term, every decision with the alternative that was rejected, the known weaknesses, and a Q&A section. Written to be presented from. |
| `cbbwp-progress-and-results.md` | Build log, current results, what to do next. The re-entry point. |
| `cbb-win-probability-model-plan.md` | The original design plan, kept verbatim. Where the build departed from it, EXPLAIN §7 says why. |
| `cbbwp-source.md` | Every source file in one document, for disaster recovery. **Generated** — do not edit by hand; regenerate with the snippet below whenever the code changes meaningfully, and copy it to the project. |

## Mirrored to the Claude project

These four are mirrored into the project (`claude/…`) so a fresh session can
read them without the folder mounted. **This folder is the source of truth**;
the project copies are the mirror. When you change a doc here, push it there.

Two project docs are *not* mirrored down here: `cbbwp-report-page.md` and
`cbbwp-report-data.md`. They are the source of the published results artifact
rather than working files, and the artifact is updated from the project.
`scripts/evaluate.py` regenerates every number in them.

## Regenerating the source bundle

```bash
python3 - <<'EOF'
import pathlib, datetime
ROOT = pathlib.Path(__file__).resolve().parent if False else pathlib.Path.cwd()
ORDER = (sorted((ROOT/"scripts").glob("*.py"))
         + [ROOT/"src/cbbwp/__init__.py"]
         + sorted(p for p in (ROOT/"src/cbbwp").glob("*.py") if p.name != "__init__.py")
         + sorted((ROOT/"src/cbbwp/adapters").glob("*.py"))
         + sorted((ROOT/"tests").glob("*.py"))
         + [ROOT/"pyproject.toml", ROOT/"README.md"])
LANG = {".py": "python", ".toml": "toml", ".md": "markdown"}
out = [f"# `cbbwp` source bundle",
       f"Regenerated {datetime.datetime.now():%Y-%m-%d %H:%M}.", "", "---", ""]
for p in ORDER:
    if not p.exists(): continue
    rel = p.relative_to(ROOT)
    out += [f"## `{rel}`", "", f"```{LANG.get(p.suffix,'')}",
            p.read_text().rstrip(), "```", "", "---", ""]
pathlib.Path("docs/cbbwp-source.md").write_text("\n".join(out))
print("wrote docs/cbbwp-source.md")
EOF
```

Run it from the repo root.

**Keep EXPLAIN.md current.** When a number changes, change it there too — it is
the document people will actually read.
