"""Exhaustive endgame solver: the last 60 seconds, by backward induction.

Endgame plan, Phase 3. The plan's design decision was "a table, not a live
simulation", for three reasons: serving becomes a lookup, monotonicity can be
ENFORCED across the whole table rather than hoped for, and a person can read a
row and check it against their own judgement.

This module goes one step further than the plan asked and replaces Monte Carlo
with **backward induction**. The endgame state space is small, discrete and
acyclic in time -- every transition burns at least one second -- so the exact
value of every state can be computed directly. That removes simulation noise
entirely, which matters because two identical states must never disagree, and
it makes the monotonicity checks meaningful: a violation is then a statement
about the model, not about how many samples were drawn.

STATE, from the point of view of the team WITH THE BALL
    t   seconds remaining, 0..60
    m   that team's margin, clamped to -12..+12
    fo  that team's own team fouls this half, 0..10 (10 = "10 or more")
    fd  the defending team's team fouls, same encoding
    bo  that team's free-throw ability bucket, 0=poor 1=average 2=good
    bd  the defending team's bucket

V[t][m, fo, fd, bo, bd] = P(the team with the ball wins).

Symmetry is structural rather than fitted. When possession changes, the value
of the state to the team that just lost the ball is

    1 - V[t'][-m, fd, fo, bd, bo]

so the table cannot disagree with itself about which side of a game it is
describing, and a mirrored state cannot drift from its twin.

Every probability comes from artifacts/endgame_params.json and
artifacts/endgame_possessions.json, both measured on 2016-2024 only. Nothing
here is hand-set except the state-space bounds and the smoothing window, and
both are declared as constants below.

WHAT THIS DELIBERATELY DOES NOT MODEL, and why
  * Timeouts. The plan lists them in the state space, but no separable effect
    was measured -- the possession-length and foul-rate cells already average
    over however teams actually used their timeouts. Adding a state dimension
    with an invented coefficient would add sampling error to sparse states for
    no measured gain. See `docs/cbbwp-endgame-phase2.md`.
  * Optimal play. The fouling rule is OBSERVED behaviour. The model predicts
    real games, in which coaches foul later and less often than optimally.
  * Team strength. The table is deliberately team-agnostic apart from free-throw
    ability. Strength enters at blend time, where the model already carries it.
    A tie at 0:00 is therefore 0.5 here, not a rating-adjusted overtime number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- state space ------------------------------------------------------------
T_MAX = 60
MARGIN_MIN, MARGIN_MAX = -12, 12
FOUL_MAX = 10           # 10 encodes "10 or more"
N_FT_BUCKETS = 3

MARGINS = np.arange(MARGIN_MIN, MARGIN_MAX + 1)
NM = len(MARGINS)
NF = FOUL_MAX + 1
NB = N_FT_BUCKETS
SHAPE = (NM, NF, NF, NB, NB)

BONUS_FOULS = 7          # the 7th team foul of a half starts the one-and-one
DOUBLE_BONUS_FOULS = 10

# Possession lengths are measured as means; a single deterministic length would
# put artificial parity structure into the table ("exactly three possessions
# left"). Spread each mean over three adjacent seconds instead.
DURATION_SMOOTHING = (0.25, 0.50, 0.25)
MAX_DURATION = 30


def _mi(m: np.ndarray | int):
    """Margin -> index, clamped."""
    return np.clip(m, MARGIN_MIN, MARGIN_MAX) - MARGIN_MIN


REV = np.arange(NM)[::-1]     # index of -m


@dataclass(frozen=True)
class Params:
    """Everything the solver needs, already smoothed onto the state grid."""
    p_foul: np.ndarray        # (T_MAX+1, NM)  defence sends the offence to the line
    p_to: np.ndarray          # (T_MAX+1, NM)  turnover, conditional on not fouled
    p3a: np.ndarray           # (NM,) share of shots that are threes
    p3: np.ndarray            # (NM,)
    p2: np.ndarray            # (NM,)
    dur: np.ndarray           # (T_MAX+1, NM) mean seconds, played out
    dur_foul: np.ndarray      # (T_MAX+1, NM) mean seconds to the foul
    ft1_bonus: np.ndarray     # (NB,) front end of a one-and-one
    ft2_bonus: np.ndarray     # (NB,) the bonus shot
    ft1_two: np.ndarray       # (NB,) first of two
    ft2_two: np.ndarray       # (NB,) second of two
    oreb_ft: float
    oreb_3: float
    oreb_2: float


# --- parameter assembly -----------------------------------------------------
def _cell_grid(cells: dict, field: str, default: float) -> np.ndarray:
    """The (m, 10s-bucket) measurements, interpolated onto (t, m)."""
    buckets = sorted({float(k.split("|")[1]) for k in cells})
    raw = np.full((len(buckets), NM), np.nan)
    for key, v in cells.items():
        ms, tbs = key.split("|")
        m = int(float(ms))
        if not (MARGIN_MIN <= m <= MARGIN_MAX):
            continue
        val = v.get(field)
        if val is None:
            continue
        raw[buckets.index(float(tbs)), _mi(m)] = val
    # Fill margin gaps by nearest neighbour along m, then interpolate in t.
    for r in range(raw.shape[0]):
        row = raw[r]
        if np.all(np.isnan(row)):
            row[:] = default
        else:
            idx = np.arange(NM)
            good = ~np.isnan(row)
            row[~good] = np.interp(idx[~good], idx[good], row[good])
    centres = np.array(buckets) + 5.0
    out = np.empty((T_MAX + 1, NM))
    for j in range(NM):
        out[:, j] = np.interp(np.arange(T_MAX + 1), centres, raw[:, j])
    return out


def load_params(params_path: Path, poss_path: Path, ft_bucket_offsets) -> Params:
    P = json.loads(Path(params_path).read_text())
    Q = json.loads(Path(poss_path).read_text())
    cells = Q["cells"]

    p_foul = np.clip(_cell_grid(cells, "p_fouled_to_line", 0.15), 0.0, 0.95)
    p_to_raw = np.clip(_cell_grid(cells, "p_turnover", 0.13), 0.0, 0.9)
    # Measured turnover share is unconditional; the solver needs it conditional
    # on the possession not having ended at the foul line.
    p_to = np.clip(p_to_raw / np.maximum(1e-6, 1.0 - p_foul), 0.0, 0.95)

    dur = np.clip(_cell_grid(cells, "mean_dur", 6.0), 1.0, MAX_DURATION)
    dur_foul = np.clip(_cell_grid(cells, "mean_dur_when_fouled", 5.0), 1.0, MAX_DURATION)

    shots = P["late_shots_by_actor_margin"]
    p3a = np.empty(NM); p3 = np.empty(NM); p2 = np.empty(NM)
    for j, m in enumerate(MARGINS):
        key = str(int(np.clip(m, -6, 6)))
        s = shots.get(key)
        p3a[j] = s["p3a"]; p3[j] = s["p3"]; p2[j] = s["p2"]

    ft = P["free_throws"]["last_60s"]
    off = np.asarray(ft_bucket_offsets, dtype=float)

    def band(key, fallback):
        base = ft[key]["p"] if key in ft else fallback
        return np.clip(base + off, 0.30, 0.99)

    reb = P["rebounds"]
    return Params(
        p_foul=p_foul, p_to=p_to, p3a=p3a, p3=p3, p2=p2, dur=dur, dur_foul=dur_foul,
        ft1_bonus=band("one_and_one_1", 0.709),
        ft2_bonus=band("one_and_one_2", 0.769),
        ft1_two=band("two_shot_1", 0.718),
        ft2_two=band("two_shot_2", 0.770),
        oreb_ft=reb["oreb_after_missed_ft_last60"]["p"],
        oreb_3=reb["oreb_after_missed_3_last60"]["p"],
        oreb_2=reb["oreb_after_missed_2_last60"]["p"],
    )


# --- the solver -------------------------------------------------------------
def _terminal() -> np.ndarray:
    """t = 0. The team with the ball wins iff it is ahead; a tie is overtime."""
    v = np.empty(SHAPE)
    win = np.where(MARGINS > 0, 1.0, np.where(MARGINS < 0, 0.0, 0.5))
    v[:] = win.reshape(NM, 1, 1, 1, 1)
    return v


def _flip(v: np.ndarray) -> np.ndarray:
    """Value of a state to the team that has just LOST the ball."""
    return 1.0 - v[REV][:, :, :, :, :].transpose(0, 2, 1, 4, 3)


def _shift_margin(v: np.ndarray, pts: int) -> np.ndarray:
    """v evaluated at margin m + pts, clamped at the ends of the grid."""
    if pts == 0:
        return v
    idx = np.clip(np.arange(NM) + pts, 0, NM - 1)
    return v[idx]


def _foul_index() -> np.ndarray:
    """fd -> fd + 1, capped at FOUL_MAX."""
    return np.minimum(np.arange(NF) + 1, FOUL_MAX)


def solve(p: Params) -> np.ndarray:
    """V[t] for t = 0..T_MAX. Exact, no sampling."""
    V = np.empty((T_MAX + 1,) + SHAPE)
    V[0] = _terminal()

    inc = _foul_index()
    # shots awarded by the foul that takes the defence to fd+1
    after = np.minimum(np.arange(NF) + 1, FOUL_MAX + 1)
    shots_for = np.where(after >= DOUBLE_BONUS_FOULS, 2, np.where(after >= BONUS_FOULS, 1, 0))

    # Slices below are (margin, own_fouls, own_bucket, opp_bucket): the free-throw
    # rate varies on the SHOOTING team's bucket, which is axis 2 of that slice.
    ft1b = p.ft1_bonus.reshape(1, 1, NB, 1)
    ft2b = p.ft2_bonus.reshape(1, 1, NB, 1)
    ft1t = p.ft1_two.reshape(1, 1, NB, 1)
    ft2t = p.ft2_two.reshape(1, 1, NB, 1)

    for t in range(1, T_MAX + 1):

        def look(tau_grid: np.ndarray, transform):
            """Expected value after burning `tau_grid` seconds, per margin.

            tau_grid is (NM,) of mean seconds; each is spread over three
            adjacent whole seconds so the table carries no parity artefacts.
            """
            acc = np.zeros(SHAPE)
            base = np.rint(tau_grid).astype(int)
            for w, d in zip(DURATION_SMOOTHING, (-1, 0, 1)):
                tau = np.clip(base + d, 1, MAX_DURATION)
                for u in np.unique(tau):
                    sel = tau == u
                    nxt = V[max(0, t - int(u))]
                    contrib = transform(nxt)
                    acc[sel] += w * contrib[sel]
            return acc

        # ---- branch A: the defence fouls -------------------------------
        def fouled(nxt: np.ndarray) -> np.ndarray:
            keep = nxt                      # offence still has the ball
            lost = _flip(nxt)               # offence has given it up
            out = np.empty(SHAPE)

            for f in range(NF):
                fd_new = inc[f]
                s = shots_for[f]
                k = keep[:, :, fd_new, :, :]
                l = lost[:, :, fd_new, :, :]

                def at(arr, pts):
                    return _shift_margin(arr, pts)

                if s == 0:
                    # No shots: the ball goes back in from the side.
                    out[:, :, f, :, :] = k
                    continue

                if s == 1:
                    p1 = ft1b; p2_ = ft2b
                    # miss the front end -> live rebound
                    miss1 = (1 - p1) * (p.oreb_ft * at(k, 0) + (1 - p.oreb_ft) * at(l, 0))
                    # make it, then the bonus shot
                    make2 = p2_ * at(l, 2)
                    miss2 = (1 - p2_) * (p.oreb_ft * at(k, 1) + (1 - p.oreb_ft) * at(l, 1))
                    out[:, :, f, :, :] = miss1 + p1 * (make2 + miss2)
                    continue

                # Two shots. The first only adds a point; the trip -- and so the
                # possession -- is decided by the second.
                p1 = ft1t; p2_ = ft2t
                def trip(made1: int):
                    return (
                        p2_ * at(l, made1 + 1)
                        + (1 - p2_) * (p.oreb_ft * at(k, made1) + (1 - p.oreb_ft) * at(l, made1))
                    )
                out[:, :, f, :, :] = p1 * trip(1) + (1 - p1) * trip(0)
            return out

        # ---- branch B: turnover ----------------------------------------
        def turned_over(nxt: np.ndarray) -> np.ndarray:
            return _flip(nxt)

        # ---- branch C: a shot goes up -----------------------------------
        def shot(nxt: np.ndarray) -> np.ndarray:
            keep, lost = nxt, _flip(nxt)
            three = (
                p.p3[:, None, None, None, None] * _shift_margin(lost, 3)
                + (1 - p.p3[:, None, None, None, None])
                * (p.oreb_3 * keep + (1 - p.oreb_3) * lost)
            )
            two = (
                p.p2[:, None, None, None, None] * _shift_margin(lost, 2)
                + (1 - p.p2[:, None, None, None, None])
                * (p.oreb_2 * keep + (1 - p.oreb_2) * lost)
            )
            a = p.p3a[:, None, None, None, None]
            return a * three + (1 - a) * two

        pf = p.p_foul[t][:, None, None, None, None]
        pt = p.p_to[t][:, None, None, None, None]

        v_foul = look(p.dur_foul[t], fouled)
        v_to = look(p.dur[t], turned_over)
        v_shot = look(p.dur[t], shot)

        V[t] = pf * v_foul + (1 - pf) * (pt * v_to + (1 - pt) * v_shot)
        np.clip(V[t], 0.0, 1.0, out=V[t])

    return V


# --- monotonicity -----------------------------------------------------------
def monotonicity_report(V: np.ndarray) -> dict:
    """Every violation the endgame plan names, checked exhaustively."""
    d_margin = np.diff(V, axis=1)                     # scoring must not hurt
    d_bo = np.diff(V, axis=4)                         # better own FT shooting
    d_bd = np.diff(V, axis=5)                         # better opponent FT shooting
    own_foul = np.diff(V, axis=2)                     # more fouls of your own
    opp_foul = np.diff(V, axis=3)                     # more fouls by the opponent
    have_ball = V - _flip_all(V)
    return {
        "margin_min_increment": float(d_margin.min()),
        "margin_violations": int((d_margin < -1e-12).sum()),
        "own_ft_bucket_min_increment": float(d_bo.min()),
        "own_ft_bucket_violations": int((d_bo < -1e-12).sum()),
        "opp_ft_bucket_max_increment": float(d_bd.max()),
        "opp_ft_bucket_violations": int((d_bd > 1e-12).sum()),
        "own_fouls_max_increment": float(own_foul.max()),
        "opp_fouls_min_increment": float(opp_foul.min()),
        "possession_min_advantage": float(have_ball.min()),
        "possession_violations": int((have_ball < -1e-12).sum()),
    }


def _flip_all(V: np.ndarray) -> np.ndarray:
    return 1.0 - V[:, REV].transpose(0, 1, 3, 2, 5, 4)


def enforce_margin_monotonicity(V: np.ndarray) -> tuple[np.ndarray, float]:
    """Isotonic projection along margin; returns the largest correction made.

    Parameters are estimated from finite samples, so a cell can end up a
    fraction below its neighbour. Projecting is honest as long as the size of
    the correction is reported -- a large one would mean the model is wrong,
    not merely noisy.
    """
    out = np.maximum.accumulate(V, axis=1)
    return out, float(np.abs(out - V).max())


# --- serving ----------------------------------------------------------------
def ft_bucket(pct, bucket_means) -> np.ndarray:
    """Nearest free-throw ability bucket for a team's season FT%."""
    means = np.asarray(bucket_means, dtype=float)
    p = np.asarray(pct, dtype=float)
    return np.abs(p[..., None] - means).argmin(axis=-1).astype(np.int64)


def lookup_home(
    table: np.ndarray,
    seconds_remaining,
    margin_home,
    possession,
    home_fouls,
    away_fouls,
    home_bucket,
    away_bucket,
) -> np.ndarray:
    """P(home wins), from a table stored in the ball-holder's point of view.

    `possession` is the pipeline's convention: 1.0 home, 0.0 away, 0.5 unknown.
    An unknown possession is averaged rather than guessed, which is the only
    answer that cannot be wrong in a way that shows up as a discontinuity.
    """
    t = np.clip(np.rint(np.asarray(seconds_remaining)).astype(int), 0, T_MAX)
    # margin arrives as a float in the state frame; the table is indexed by
    # whole points, so round rather than truncate.
    m = np.rint(np.asarray(margin_home, dtype=float)).astype(int)
    fh = np.clip(np.asarray(home_fouls), 0, FOUL_MAX).astype(int)
    fa = np.clip(np.asarray(away_fouls), 0, FOUL_MAX).astype(int)
    bh = np.asarray(home_bucket).astype(int)
    ba = np.asarray(away_bucket).astype(int)

    home_ball = table[t, _mi(m), fh, fa, bh, ba]
    away_ball = 1.0 - table[t, _mi(-m), fa, fh, ba, bh]
    poss = np.asarray(possession, dtype=float)
    return np.where(poss >= 0.75, home_ball,
                    np.where(poss <= 0.25, away_ball, 0.5 * (home_ball + away_ball)))
