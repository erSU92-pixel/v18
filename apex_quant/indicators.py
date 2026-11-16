"""Causal, zero-lookahead technical indicators and statistical regime estimators."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, lfilter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class Indicators:
    """Statistically pure, strictly causal indicator calculations."""

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> float:
        s = Indicators.rsi_series(series, period)
        return float(s.iloc[-1]) if not s.empty and np.isfinite(s.iloc[-1]) else 50.0

    @staticmethod
    def true_range(df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    @staticmethod
    def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = Indicators.true_range(df)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def atr_last(df: pd.DataFrame, period: int = 14) -> float:
        atr_s = Indicators.atr_series(df, period)
        if atr_s.empty:
            return 0.0
        val = float(atr_s.iloc[-1])
        return val if np.isfinite(val) else 0.0

    @staticmethod
    def adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low = df["high"], df["low"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = Indicators.true_range(df)
        atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
        pdm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        mdm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
        plus_di = 100.0 * (pdm_s / (atr + 1e-9))
        minus_di = 100.0 * (mdm_s / (atr + 1e-9))
        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
        return dx.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def adx_last(df: pd.DataFrame, period: int = 14) -> float:
        s = Indicators.adx_series(df, period)
        if s.empty:
            return 0.0
        val = float(s.iloc[-1])
        return val if np.isfinite(val) else 0.0

    @staticmethod
    def macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[float, float, float]:
        ef = Indicators.ema(series, fast)
        es = Indicators.ema(series, slow)
        macd_line = ef - es
        sig_line = Indicators.ema(macd_line, signal)
        hist = macd_line - sig_line
        return float(macd_line.iloc[-1]), float(sig_line.iloc[-1]), float(hist.iloc[-1])

    @staticmethod
    def bollinger_bands(
        series: pd.Series, period: int = 20, num_std: float = 2.0
    ) -> Tuple[float, float, float]:
        sma = series.rolling(period).mean()
        std = series.rolling(period).std(ddof=0)
        upper = sma + num_std * std
        lower = sma - num_std * std
        return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])

    @staticmethod
    def stochastic(
        df: pd.DataFrame, period: int = 14, smooth_k: int = 3
    ) -> Tuple[float, float]:
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()
        k_fast = 100.0 * ((df["close"] - low_min) / (high_max - low_min + 1e-9))
        k = k_fast.rolling(smooth_k).mean()
        d = k.rolling(smooth_k).mean()
        k_val = float(k.iloc[-1]) if np.isfinite(k.iloc[-1]) else 50.0
        d_val = float(d.iloc[-1]) if np.isfinite(d.iloc[-1]) else 50.0
        return k_val, d_val

    @staticmethod
    def supertrend(
        df: pd.DataFrame, period: int = 10, mult: float = 3.0
    ) -> Tuple[int, float, float]:
        """
        True ratcheting Supertrend indicator.
        Returns (direction: +1/-1, active_band_price, atr_last).
        """
        n = len(df)
        if n < period + 2:
            return 0, float(df["close"].iloc[-1]), 0.0

        atr = Indicators.atr_series(df, period).values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        hl2 = (high + low) / 2.0

        basic_upper = hl2 + mult * atr
        basic_lower = hl2 - mult * atr

        final_upper = np.copy(basic_upper)
        final_lower = np.copy(basic_lower)
        direction = np.ones(n, dtype=int)

        for i in range(1, n):
            if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i - 1]

            if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i - 1]

            if direction[i - 1] == 1:
                direction[i] = -1 if close[i] < final_lower[i] else 1
            else:
                direction[i] = 1 if close[i] > final_upper[i] else -1

        dir_last = int(direction[-1])
        active_band = float(final_lower[-1] if dir_last == 1 else final_upper[-1])
        return dir_last, active_band, float(atr[-1])

    @staticmethod
    def kalman_filter(series: pd.Series, atr_series: pd.Series) -> pd.Series:
        """
        Scale-invariant 1D Kalman Filter calibrated dynamically to Gold's ATR variance.
        Avoids fixed hardcoded Q/R unit mismatch across different Gold price regimes.
        """
        prices = series.values.astype(float)
        n = len(prices)
        if n == 0:
            return pd.Series([], dtype=float)

        valid_atr = atr_series.dropna().values
        med_atr = float(np.median(valid_atr)) if len(valid_atr) > 0 else 1.0
        r_var = max(med_atr ** 2, 1e-6)
        q_var = 0.05 * r_var

        x_hat = prices[0]
        p = r_var
        estimates = np.empty(n, dtype=float)

        for i in range(n):
            p = p + q_var
            k = p / (p + r_var)
            x_hat = x_hat + k * (prices[i] - x_hat)
            p = (1.0 - k) * p
            estimates[i] = x_hat

        return pd.Series(estimates, index=series.index)

    @staticmethod
    def causal_lowpass(series: pd.Series, cutoff: float = 0.15) -> pd.Series:
        """
        Strictly causal low-pass filter (no look-ahead / no filtfilt backward pass).
        Uses SciPy causal lfilter when available, otherwise double-EMA zero-lag causal smoothing.
        """
        vals = series.values.astype(float)
        if len(vals) < 10:
            return series.copy()

        if HAS_SCIPY:
            b, a = butter(2, cutoff)
            # Initialize filter state at first sample to prevent massive startup transient
            zi = np.zeros(max(len(a), len(b)) - 1)
            centered = vals - vals[0]
            filtered, _ = lfilter(b, a, centered, zi=zi)
            return pd.Series(filtered + vals[0], index=series.index)

        ema1 = series.ewm(span=8, adjust=False).mean()
        ema2 = ema1.ewm(span=8, adjust=False).mean()
        return 2.0 * ema1 - ema2  # Causal DEMA

    @staticmethod
    def hurst(series: pd.Series, max_lags: int = 20) -> float:
        """
        Computes Generalized Hurst Exponent on log-prices with zero-variance protection.
        H > 0.52 indicates persistent trend; H < 0.48 indicates mean-reversion.
        """
        clean = series.dropna()
        ts = clean.values.astype(float)
        ts = ts[ts > 0]
        if len(ts) < max_lags * 2:
            return 0.5

        log_ts = np.log(ts)
        lags = np.arange(2, max_lags)
        tau = np.array([np.std(log_ts[lag:] - log_ts[:-lag]) for lag in lags], dtype=float)
        tau = np.maximum(tau, 1e-12)
        if not np.all(np.isfinite(tau)):
            return 0.5

        slope, _ = np.polyfit(np.log(lags.astype(float)), np.log(tau), 1)
        return float(np.clip(slope, 0.0, 1.0)) if np.isfinite(slope) else 0.5

    @staticmethod
    def shannon_entropy(series: pd.Series, window: int = 100, bins: int = 10) -> float:
        """Measures local rolling Shannon Entropy of recent log-returns (0 = deterministic, 1 = max noise)."""
        clean = series.dropna()
        if len(clean) > window + 1:
            clean = clean.iloc[-(window + 1) :]
        returns = np.log(clean / clean.shift(1)).dropna()
        if len(returns) < bins * 2:
            return 0.5

        counts, _ = np.histogram(returns.values, bins=bins)
        counts = counts[counts > 0]
        total = np.sum(counts)
        if total <= 0:
            return 0.5
        probs = counts / total
        ent = -np.sum(probs * np.log2(probs))
        max_ent = np.log2(bins)
        if max_ent <= 0:
            return 0.5
        return float(np.clip(ent / max_ent, 0.0, 1.0))

    @staticmethod
    def session_vwap(df: pd.DataFrame) -> pd.Series:
        """Computes UTC-session-anchored VWAP (or TWAP when feed volume is constant)."""
        if "datetime" not in df.columns or df.empty:
            return df["close"].copy()

        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        vol = df["volume"].where(df["volume"] > 0, 1.0) if "volume" in df.columns else pd.Series(1.0, index=df.index)
        session_dates = pd.to_datetime(df["datetime"], utc=True).dt.date
        pv = typical * vol
        cum_pv = pv.groupby(session_dates).cumsum()
        cum_v = vol.groupby(session_dates).cumsum()
        return cum_pv / (cum_v + 1e-9)
