"""Write an immutable, pinned model artifact into the registry.

    python3 scripts/publish_model.py v3
    python3 scripts/publish_model.py v3 --from artifacts/gbm_v1.txt
    python3 scripts/publish_model.py v2 --force        # deliberate re-publish

Two things this refuses to do, because "immutable and pinned" has to be enforced
somewhere or it is only a description:

  * overwrite an existing version without --force. A registry directory is what
    a running deployment loads by name; silently replacing one means the same
    version string refers to two different models, and every number ever
    recorded against it becomes ambiguous.
  * publish a source file it cannot find, or one whose hash it cannot record.

The version label is just a label: it does NOT constrain which fit gets copied.
`--from` is explicit so the binding between a label and a particular fit is at
least visible in the shell history, and the manifest records the source path and
the file's modification time so a published artifact can be traced back.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cbbwp.schemas import FEATURE_NAMES, STATE_RULES_VERSION  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("version", nargs="?", default="v1", help="registry version to write")
ap.add_argument("--from", dest="source", default=str(ROOT / "artifacts/gbm_v1.txt"),
                help="the fitted booster to publish (default: artifacts/gbm_v1.txt)")
ap.add_argument("--force", action="store_true",
                help="replace an existing registry version (it is meant to be immutable)")
ap.add_argument("--notes", default="", help="free text recorded in the manifest")
a = ap.parse_args()

src = pathlib.Path(a.source)
if not src.exists():
    raise SystemExit(f"no such model file: {src}\n"
                     "  run scripts/fit_models.py first, or pass --from")

dest = ROOT / "registry" / a.version
model_path = dest / "model.txt"
if model_path.exists() and not a.force:
    existing = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    incoming = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    if existing == incoming:
        print(f"{a.version} already holds this exact model ({existing}); nothing to do")
        raise SystemExit(0)
    raise SystemExit(
        f"refusing to overwrite registry/{a.version}\n"
        f"  it holds {existing}, you are publishing {incoming}\n"
        "  A version is a name a deployment loads; two models under one name makes\n"
        "  every recorded number against it ambiguous. Publish a new version, or\n"
        "  pass --force if replacing it is genuinely what you mean.")

dest.mkdir(parents=True, exist_ok=True)
shutil.copy(src, model_path)
digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
manifest = {
    "version": a.version, "kind": "lightgbm", "features": FEATURE_NAMES,
    "state_rules_version": STATE_RULES_VERSION,
    "sha256": digest,
    "created": datetime.datetime.now(datetime.UTC).isoformat(),
    # Provenance: which file this came from, and when that file was written.
    # Without it, "publish_model.py v3" records nothing about WHICH fit it took.
    "source_path": str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
    "source_mtime": datetime.datetime.fromtimestamp(
        src.stat().st_mtime, datetime.UTC).isoformat(),
    "train_seasons": [2016, 2017, 2018, 2019, 2021, 2022, 2023],
    "calibration_season": 2024, "test_seasons": [2025, 2026],
}
if a.notes:
    manifest["notes"] = a.notes
(dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("published", dest, digest, "from", src)
