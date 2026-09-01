import sys, pathlib, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp.calibration import TimeBucketedCalibrator
from cbbwp import endgame
from cbbwp.evaluate import log_loss, brier, accuracy, ece, calibration_table

ROOT = pathlib.Path(__file__).resolve().parents[1]
ca = np.load(ROOT / "artifacts/calib_preds.npz")
te = np.load(ROOT / "artifacts/test_preds.npz")
y, secs, espn = te["y"], te["secs"], te["espn"]

cal = TimeBucketedCalibrator().fit(ca["p_gbm"], ca["y"], ca["secs"])
with open(ROOT / "artifacts/calibrator_v1.pkl", "wb") as f:
    pickle.dump(cal, f)

raw = te["p_gbm"]
calibrated = cal.transform(raw, secs)
# endgame overrides need the margin, recovered from the saved feature column
import polars as pl
margin = pl.concat([pl.scan_parquet(ROOT / f"data/proc/states/states_{s}.parquet")
                    .select("margin") for s in (2025, 2026)], how="diagonal").collect()["margin"].to_numpy()
assert len(margin) == len(y)
final = endgame.apply(calibrated, margin, secs)

rows = {
    "lightgbm raw": raw,
    "+ time-bucket calib": calibrated,
    "+ endgame overrides": final,
    "logistic baseline": te["p_lr"],
    "ESPN (deployed)": espn,
}
print(f"{'model':<22}{'logloss':>9}{'brier':>9}{'acc':>8}{'ECE':>8}")
for k, p in rows.items():
    print(f"{k:<22}{log_loss(y,p):>9.4f}{brier(y,p):>9.4f}{accuracy(y,p):>8.4f}{ece(y,p):>8.4f}")

print(f"\nlog loss by time bucket\n{'bucket':<12}{'n':>10}" + "".join(f"{k[:11]:>13}" for k in rows))
for name, lo, hi in [("40-20 min",1200,2400),("20-10 min",600,1200),("10-5 min",300,600),
                     ("5-2 min",120,300),("2-1 min",60,120),("1-0 min",0,60)]:
    m = (secs >= lo) & (secs < hi)
    print(f"{name:<12}{m.sum():>10,}" + "".join(f"{log_loss(y[m],p[m]):>13.4f}" for p in rows.values()))

print("\nfinal-2-minute calibration, after correction")
m = secs < 120
print(f"{'bin':<12}{'n':>10}{'pred':>8}{'obs':>8}{'gap':>8}")
for r in calibration_table(y[m], final[m], n_bins=10):
    print(f"{r['bin']:<12}{r['n']:>10,}{r['pred']:>8.3f}{r['obs']:>8.3f}{r['obs']-r['pred']:>+8.3f}")

np.savez_compressed(ROOT / "artifacts/final_preds.npz", y=y, p=final, secs=secs,
                    espn=espn, margin=margin, game_id=te["game_id"], season=te["season"])
