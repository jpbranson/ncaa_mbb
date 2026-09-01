"""Write an immutable, pinned model artifact into the registry."""
import sys, pathlib, json, shutil, hashlib, datetime
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cbbwp.schemas import FEATURE_NAMES, STATE_RULES_VERSION

ROOT = pathlib.Path(__file__).resolve().parents[1]
version = sys.argv[1] if len(sys.argv) > 1 else "v1"
dest = ROOT / "registry" / version
dest.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / "artifacts/gbm_v1.txt", dest / "model.txt")
digest = hashlib.sha256((dest / "model.txt").read_bytes()).hexdigest()[:16]
(dest / "manifest.json").write_text(json.dumps({
    "version": version, "kind": "lightgbm", "features": FEATURE_NAMES,
    "state_rules_version": STATE_RULES_VERSION,
    "sha256": digest, "created": datetime.datetime.now(datetime.UTC).isoformat(),
    "train_seasons": [2016, 2017, 2018, 2019, 2021, 2022, 2023],
    "calibration_season": 2024, "test_seasons": [2025, 2026],
}, indent=2))
print("published", dest, digest)
