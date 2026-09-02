import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cbbwp.evaluate import by_time_bucket, log_loss, brier, calibration_table, ece, accuracy

ROOT = pathlib.Path(__file__).resolve().parents[1]
# fit_models.py writes test_preds.npz; a later step exported the same columns
# as eval_preds.parquet for the results report. artifacts/ is gitignored, so a
# restored working copy can have either, or only one. Accept both -- otherwise
# "scripts/evaluate.py regenerates every number" is a claim that fails on any
# machine that happens to have the other file.
_npz = ROOT / "artifacts/test_preds.npz"
_pq = ROOT / "artifacts/eval_preds.parquet"
if _npz.exists():
    d = np.load(_npz)
    print(f"source: {_npz.name} (float64 as fitted)\n")
elif _pq.exists():
    import polars as pl
    _t = pl.read_parquet(_pq)
    # The parquet stores predictions as float32. Metrics agree with the float64
    # originals to three decimals; accuracy and ECE can differ in the fourth.
    d = {c: _t[c].to_numpy().astype(np.float64) for c in _t.columns}
    print(f"source: {_pq.name} (float32 export -- acc/ECE may differ in the 4th decimal)\n")
else:
    raise SystemExit(
        "no predictions found. Expected artifacts/test_preds.npz (written by "
        "fit_models.py) or artifacts/eval_preds.parquet. Re-run scripts/fit_models.py.")
y, secs, espn = d["y"], d["secs"], d["espn"]
preds = {"logistic": d["p_lr"], "lightgbm": d["p_gbm"], "espn": espn}

ok = np.isfinite(espn)
print(f"test rows {len(y):,}   ESPN win prob present on {ok.mean()*100:.1f}%\n")
print(f"{'model':<10} {'logloss':>8} {'brier':>8} {'acc':>7} {'ECE':>7}   (rows where ESPN also present)")
for name, p in preds.items():
    print(f"{name:<10} {log_loss(y[ok],p[ok]):>8.4f} {brier(y[ok],p[ok]):>8.4f} "
          f"{accuracy(y[ok],p[ok]):>7.4f} {ece(y[ok],p[ok]):>7.4f}")

print(f"\n{'bucket':<12}{'n':>10}" + "".join(f"{k:>12}" for k in preds))
for name, lo, hi in [("40-20 min",1200,2400),("20-10 min",600,1200),("10-5 min",300,600),
                     ("5-2 min",120,300),("2-1 min",60,120),("1-0 min",0,60)]:
    m = ok & (secs >= lo) & (secs < hi)
    if m.sum() == 0: continue
    print(f"{name:<12}{m.sum():>10,}" + "".join(f"{log_loss(y[m],p[m]):>12.4f}" for p in preds.values()))
print("(cells are log loss; lower is better)")

print("\ncalibration of lightgbm on the test seasons")
print(f"{'bin':<12}{'n':>10}{'pred':>8}{'obs':>8}{'gap':>8}")
for r in calibration_table(y, d["p_gbm"]):
    print(f"{r['bin']:<12}{r['n']:>10,}{r['pred']:>8.3f}{r['obs']:>8.3f}{r['obs']-r['pred']:>+8.3f}")
