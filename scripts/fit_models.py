"""Fit the logistic baseline and the LightGBM model. Split by season, never randomly."""
import sys, pathlib, json, pickle, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from cbbwp.schemas import FEATURE_NAMES
from cbbwp.features import mirror_features
from cbbwp.evaluate import log_loss, brier, accuracy

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN = [2016, 2017, 2018, 2019, 2021, 2022, 2023]
CALIB = [2024]
TEST  = [2025, 2026]

# Monotonic constraints (plan 4.2): more points, the ball, and a better team can
# never *lower* the home side's win probability.
MONOTONE = {
    "margin": 1, "sqrt_time": 0, "margin_per_sqrt_time": 1, "possession": 1,
    "pregame_exp_margin": 1, "pregame_exp_margin_decayed": 1,
    "is_ot": 0, "timeout_diff": 0, "bonus_diff": 0, "ft_pct_diff": 0,
    "margin_per_sqrt_points_left": 1,
}


def load(seasons, cols=None):
    files = [ROOT / f"data/proc/states/states_{y}.parquet" for y in seasons]
    lf = pl.concat([pl.scan_parquet(f) for f in files], how="diagonal")
    return lf.select(cols).collect() if cols else lf.collect()


def xy(df):
    X = df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    y = df["home_win"].to_numpy().astype(np.int8)
    return X, y


t0 = time.time()
need = FEATURE_NAMES + ["home_win", "game_seconds_remaining", "game_id", "season"]
tr = load(TRAIN, need)
print(f"train rows {tr.height:,}  games {tr['game_id'].n_unique():,}  ({time.time()-t0:.0f}s)")
Xtr, ytr = xy(tr)
Xtr, ytr = mirror_features(Xtr.astype(np.float64), ytr)
Xtr = Xtr.astype(np.float32); ytr = ytr.astype(np.int8)
print(f"after symmetry mirroring: {Xtr.shape[0]:,} rows")
del tr

ca = load(CALIB, need); Xca, yca = xy(ca)
te = load(TEST, need + ["espn_wp"]); Xte, yte = xy(te)
print(f"calib {Xca.shape[0]:,}   test {Xte.shape[0]:,}")

# ---- 1. logistic regression baseline ------------------------------------
t0 = time.time()
sc = StandardScaler().fit(Xtr)
lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
lr.fit(sc.transform(Xtr), ytr)
p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
print(f"\nLOGISTIC  fit {time.time()-t0:.0f}s   test logloss {log_loss(yte,p_lr):.4f}  "
      f"brier {brier(yte,p_lr):.4f}  acc {accuracy(yte,p_lr):.4f}")
print("  coefs:", {n: round(float(c), 3) for n, c in zip(FEATURE_NAMES, lr.coef_[0])})

# ---- 2. LightGBM --------------------------------------------------------
t0 = time.time()
params = dict(
    objective="binary", metric="binary_logloss", learning_rate=0.05,
    num_leaves=160, min_data_in_leaf=1500, feature_fraction=0.9,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1,
    monotone_constraints=[MONOTONE[n] for n in FEATURE_NAMES],
    monotone_constraints_method="advanced", num_threads=2,
    # Pinned so a refit reproduces registry/v1 exactly rather than approximately.
    seed=20260831, bagging_seed=20260831, feature_fraction_seed=20260831,
    data_random_seed=20260831, deterministic=True,
)
dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURE_NAMES)
dca = lgb.Dataset(Xca, label=yca, feature_name=FEATURE_NAMES, reference=dtr)
gbm = lgb.train(params, dtr, num_boost_round=2500, valid_sets=[dca],
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(200)])
p_gbm = gbm.predict(Xte, num_iteration=gbm.best_iteration)
print(f"\nLIGHTGBM  fit {time.time()-t0:.0f}s  best_iter {gbm.best_iteration}   "
      f"test logloss {log_loss(yte,p_gbm):.4f}  brier {brier(yte,p_gbm):.4f}  acc {accuracy(yte,p_gbm):.4f}")

art = ROOT / "artifacts"; art.mkdir(exist_ok=True)
with open(art / "lr_v1.pkl", "wb") as f:
    pickle.dump({"scaler": sc, "model": lr, "features": FEATURE_NAMES}, f)
gbm.save_model(str(art / "gbm_v1.txt"), num_iteration=gbm.best_iteration)
np.savez_compressed(art / "test_preds.npz",
                    y=yte, p_lr=p_lr, p_gbm=p_gbm,
                    secs=te["game_seconds_remaining"].to_numpy(),
                    espn=te["espn_wp"].to_numpy().astype(np.float64),
                    game_id=te["game_id"].to_numpy(), season=te["season"].to_numpy())
# calibration-season predictions, needed to fit the calibrators
np.savez_compressed(art / "calib_preds.npz", y=yca,
                    p_lr=lr.predict_proba(sc.transform(Xca))[:, 1],
                    p_gbm=gbm.predict(Xca, num_iteration=gbm.best_iteration),
                    secs=ca["game_seconds_remaining"].to_numpy())
print("\nsaved artifacts")
