from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.linear_model import Ridge


class ChannelSpec(BaseModel):
    name: str = Field(..., min_length=1, description="Channel name without spend_ prefix, e.g. 'search'")
    decay: float = Field(0.5, ge=0.0, le=0.99, description="Geometric adstock decay (carryover).")
    half_saturation: float = Field(1.0, gt=0.0, description="Hill function half-saturation parameter (scale).")
    hill_alpha: float = Field(1.3, gt=0.1, le=5.0, description="Hill function shape parameter.")


@dataclass(frozen=True)
class FitResult:
    model: Ridge
    feature_names: List[str]
    channel_feature_names: Dict[str, List[str]]
    channels: List[ChannelSpec]
    y_mean: float
    diagnostics: Dict[str, float]


def _ensure_weekly_sorted(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """
    Geometric adstock: y[t] = x[t] + decay * y[t-1]
    """
    if x.ndim != 1:
        raise ValueError("adstock expects a 1D array")
    y = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x.astype(float)):
        carry = float(v) + float(decay) * carry
        y[i] = carry
    return y


def hill_saturation(x: np.ndarray, half_saturation: float, alpha: float) -> np.ndarray:
    """
    Hill saturation (0..1): f(x) = x^a / (x^a + s^a)
    """
    x = np.maximum(0.0, x.astype(float))
    a = float(alpha)
    s = float(half_saturation)
    # Avoid numerical issues at x=0
    xa = np.power(x + 1e-12, a)
    sa = math.pow(s, a)
    return xa / (xa + sa)


def fourier_seasonality(dates: Sequence[pd.Timestamp], period_weeks: float = 52.0, K: int = 2) -> pd.DataFrame:
    """
    Simple Fourier seasonality terms for weekly data.
    """
    t = np.arange(len(dates), dtype=float)
    cols: Dict[str, np.ndarray] = {}
    for k in range(1, int(K) + 1):
        cols[f"sin_{k}"] = np.sin(2 * math.pi * k * t / float(period_weeks))
        cols[f"cos_{k}"] = np.cos(2 * math.pi * k * t / float(period_weeks))
    return pd.DataFrame(cols)


def infer_spend_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("spend_")]


def build_design_matrix(
    df: pd.DataFrame,
    *,
    channels: Sequence[ChannelSpec],
    control_columns: Sequence[str],
    add_trend: bool = True,
    add_seasonality: bool = True,
    seasonality_K: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Returns X and a mapping from channel name -> feature columns used.
    """
    df = _ensure_weekly_sorted(df)
    X_parts: List[pd.DataFrame] = []
    channel_feature_names: Dict[str, List[str]] = {}

    # Trend + seasonality
    if add_trend:
        X_parts.append(pd.DataFrame({"trend": np.arange(len(df), dtype=float)}))
    if add_seasonality:
        X_parts.append(fourier_seasonality(list(df["date"]), K=seasonality_K))

    # Controls (as-is)
    for c in control_columns:
        if c not in df.columns:
            raise ValueError(f"Control column missing: {c}")
    if control_columns:
        X_parts.append(df[list(control_columns)].astype(float))

    # Channels: adstock then saturation
    for ch in channels:
        col = f"spend_{ch.name}"
        if col not in df.columns:
            raise ValueError(f"Missing spend column for channel '{ch.name}': expected '{col}'")
        raw = df[col].fillna(0.0).astype(float).to_numpy()
        ad = geometric_adstock(raw, decay=ch.decay)
        sat = hill_saturation(ad, half_saturation=ch.half_saturation, alpha=ch.hill_alpha)
        fname = f"ch__{ch.name}"
        X_parts.append(pd.DataFrame({fname: sat}))
        channel_feature_names[ch.name] = [fname]

    X = pd.concat(X_parts, axis=1)
    return X, channel_feature_names


def fit_mmm(
    df: pd.DataFrame,
    *,
    channels: Sequence[ChannelSpec],
    control_columns: Sequence[str],
    ridge_alpha: float = 2.0,
    add_trend: bool = True,
    add_seasonality: bool = True,
    seasonality_K: int = 2,
) -> FitResult:
    df = _ensure_weekly_sorted(df)
    if "kpi" not in df.columns or "date" not in df.columns:
        raise ValueError("CSV must include 'date' and 'kpi' columns.")

    y = df["kpi"].astype(float).to_numpy()
    y_mean = float(np.mean(y))
    y_centered = y - y_mean

    X, channel_feature_names = build_design_matrix(
        df,
        channels=list(channels),
        control_columns=list(control_columns),
        add_trend=add_trend,
        add_seasonality=add_seasonality,
        seasonality_K=seasonality_K,
    )

    model = Ridge(alpha=float(ridge_alpha), fit_intercept=True, random_state=0)
    model.fit(X.to_numpy(), y_centered)

    y_hat = model.predict(X.to_numpy()) + y_mean
    resid = y - y_hat
    rmse = float(np.sqrt(np.mean(np.square(resid))))
    mae = float(np.mean(np.abs(resid)))
    r2 = float(model.score(X.to_numpy(), y_centered))

    return FitResult(
        model=model,
        feature_names=list(X.columns),
        channel_feature_names=channel_feature_names,
        channels=list(channels),
        y_mean=y_mean,
        diagnostics={"rmse": rmse, "mae": mae, "r2_centered": r2},
    )


def contributions_over_time(
    df: pd.DataFrame,
    *,
    fit: FitResult,
    control_columns: Sequence[str],
    add_trend: bool = True,
    add_seasonality: bool = True,
    seasonality_K: int = 2,
) -> pd.DataFrame:
    """
    Decomposes predictions into baseline (intercept + mean) + each feature contribution.
    Returns a dataframe with date, actual, predicted, and contribution columns.
    """
    df = _ensure_weekly_sorted(df)
    X, _ = build_design_matrix(
        df,
        channels=fit.channels,
        control_columns=list(control_columns),
        add_trend=add_trend,
        add_seasonality=add_seasonality,
        seasonality_K=seasonality_K,
    )
    # Ensure same column order
    X = X[fit.feature_names]
    coefs = fit.model.coef_.reshape(-1)
    intercept = float(fit.model.intercept_)

    contrib = X.to_numpy() * coefs[None, :]
    contrib_df = pd.DataFrame(contrib, columns=[f"contrib__{c}" for c in X.columns])

    out = pd.DataFrame(
        {
            "date": df["date"].to_numpy(),
            "kpi_actual": df["kpi"].astype(float).to_numpy(),
        }
    )
    out = pd.concat([out, contrib_df], axis=1)
    out["baseline"] = fit.y_mean + intercept
    out["kpi_pred"] = out["baseline"] + contrib_df.sum(axis=1)
    return out


def channel_summary(contrib_df: pd.DataFrame, channels: Sequence[ChannelSpec]) -> pd.DataFrame:
    """
    Summarize total contribution by channel.
    """
    rows: List[Dict[str, float | str]] = []
    for ch in channels:
        col = f"contrib__ch__{ch.name}"
        if col not in contrib_df.columns:
            continue
        total = float(contrib_df[col].sum())
        rows.append({"channel": ch.name, "total_contribution": total})
    return pd.DataFrame(rows).sort_values("total_contribution", ascending=False)


def _response_curve(
    spend: np.ndarray,
    *,
    channel: ChannelSpec,
    coef: float,
    horizon_weeks: int = 1,
) -> float:
    """
    Approximate next-period contribution from a constant spend level over horizon.
    Uses adstock steady-state approximation for geometric decay.
    """
    s = float(np.mean(spend))
    # steady-state adstock for constant spend: s / (1 - decay)
    ad = s / max(1e-6, (1.0 - float(channel.decay)))
    sat = float(hill_saturation(np.array([ad]), channel.half_saturation, channel.hill_alpha)[0])
    # contribution in centered-y space; intercept/mean handled elsewhere
    return float(coef) * sat * float(horizon_weeks)


def optimize_budget(
    *,
    fit: FitResult,
    current_spend: Dict[str, float],
    total_budget: float,
    min_spend: Optional[Dict[str, float]] = None,
    max_spend: Optional[Dict[str, float]] = None,
    horizon_weeks: int = 1,
    step: float = 50.0,
) -> Dict[str, float]:
    """
    Simple greedy allocator using estimated response curves.
    This avoids heavy solver dependencies while producing sensible results for demos.
    """
    min_spend = dict(min_spend or {})
    max_spend = dict(max_spend or {})

    channels = [c.name for c in fit.channels]
    for ch in channels:
        min_spend.setdefault(ch, 0.0)
        max_spend.setdefault(ch, float("inf"))

    # Start at mins
    alloc = {ch: float(min_spend[ch]) for ch in channels}
    remaining = float(total_budget) - sum(alloc.values())
    if remaining < -1e-6:
        raise ValueError("Total budget is below the sum of minimum spends.")

    # Pull coefficients for channel features
    coef_by_channel: Dict[str, float] = {}
    for ch in fit.channels:
        fcols = fit.channel_feature_names.get(ch.name, [])
        if not fcols:
            coef_by_channel[ch.name] = 0.0
            continue
        idx = fit.feature_names.index(fcols[0])
        coef_by_channel[ch.name] = float(fit.model.coef_.reshape(-1)[idx])

    # Greedy increments by step: allocate to channel with best marginal gain at current level
    step = float(step)
    if step <= 0:
        raise ValueError("step must be > 0")

    # Use current_spend to anchor starting level shape (scale)
    spend_anchor = {ch: float(current_spend.get(ch, 0.0)) for ch in channels}

    while remaining > 1e-6:
        best_ch = None
        best_gain = -float("inf")
        increment = min(step, remaining)

        for ch in fit.channels:
            name = ch.name
            if alloc[name] + increment > max_spend[name] + 1e-9:
                continue

            # Estimate gain by comparing response at alloc+inc vs alloc
            coef = coef_by_channel.get(name, 0.0)
            base = _response_curve(
                np.array([spend_anchor[name] + alloc[name]]),
                channel=ch,
                coef=coef,
                horizon_weeks=horizon_weeks,
            )
            nxt = _response_curve(
                np.array([spend_anchor[name] + alloc[name] + increment]),
                channel=ch,
                coef=coef,
                horizon_weeks=horizon_weeks,
            )
            gain = nxt - base
            if gain > best_gain:
                best_gain = gain
                best_ch = name

        if best_ch is None:
            break
        alloc[best_ch] += increment
        remaining -= increment

    return alloc


def generate_synthetic_mmm_data(
    *,
    n_weeks: int = 104,
    channels: Sequence[str] = ("search", "social", "display", "email"),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic dataset for demo purposes.
    Produces weekly spends, controls, and a KPI with carryover + saturation-like effects.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    start = pd.Timestamp("2024-01-07")
    dates = pd.date_range(start=start, periods=int(n_weeks), freq="W-SUN")

    df = pd.DataFrame({"date": dates})
    df["promo_flag"] = (np_rng.random(int(n_weeks)) < 0.12).astype(int)
    df["price_index"] = 1.0 + np.cumsum(np_rng.normal(0, 0.003, size=int(n_weeks)))

    # Base trend + seasonality
    t = np.arange(int(n_weeks), dtype=float)
    base = 1200 + 2.5 * t + 80 * np.sin(2 * math.pi * t / 52.0)

    # Channel spends
    for ch in channels:
        # Spend pulses + noise
        pulse = (np_rng.random(int(n_weeks)) < 0.18).astype(float) * np_rng.uniform(2000, 8000, size=int(n_weeks))
        spend = np_rng.gamma(shape=2.0, scale=800.0, size=int(n_weeks)) + pulse
        df[f"spend_{ch}"] = np.maximum(0.0, spend)

    # True underlying effects (unknown to model)
    true = base + 250 * df["promo_flag"].to_numpy() - 180 * (df["price_index"].to_numpy() - 1.0)

    # Add carryover-like channel effects with diminishing returns
    for ch in channels:
        x = df[f"spend_{ch}"].to_numpy()
        decay = rng.uniform(0.25, 0.75)
        ad = geometric_adstock(x, decay)
        sat = 1.0 - np.exp(-ad / (np.percentile(ad, 75) + 1e-6))
        coef = rng.uniform(180, 520)
        true += coef * sat

    noise = np_rng.normal(0, 65, size=int(n_weeks))
    df["kpi"] = true + noise
    return df

