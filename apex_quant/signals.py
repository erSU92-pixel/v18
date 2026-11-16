"""14-model quantitative parliament, SMC order-block/FVG detector, and purged walk-forward ML."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .config import Config
from .indicators import Indicators


@dataclass
class SignalResult:
    direction: str = "NEUTRAL"
    verdict: str = "CANCEL"
    buy_ratio: float = 0.0
    sell_ratio: float = 0.0
    buy_score: float = 0.0
    sell_score: float = 0.0
    active_models: int = 0
    total_models: int = 0
    votes_detail: List[Dict[str, Any]] = field(default_factory=list)
    regime: str = "UNKNOWN"
    hurst: float = 0.5
    entropy: float = 0.5
    adx: float = 0.0
    atr: float = 0.0
    htf_bias: str = "NEUTRAL"
    skip_reason: str = ""


class SignalGenerator:
    """Quantitative Parliament consensus engine composed of 14 causal market models."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._registry = self._build_model_registry()

    # Model 1: EMA Cross + Momentum
    def _ema_cross(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        if atr <= 0 or len(df) < 25:
            return 0, 0.0
        c = df["close"]
        e9, e21 = Indicators.ema(c, 9), Indicators.ema(c, 21)
        diff = float(e9.iloc[-1] - e21.iloc[-1])
        prev_diff = float(e9.iloc[-2] - e21.iloc[-2])
        conf = min(1.0, abs(diff) / atr)
        if diff > 0 and prev_diff <= 0:
            return 1, min(1.0, conf + 0.3)
        if diff < 0 and prev_diff >= 0:
            return -1, min(1.0, conf + 0.3)
        if diff > 0:
            return 1, conf
        if diff < 0:
            return -1, conf
        return 0, 0.0

    # Model 2: Triple EMA Alignment
    def _ema_trend(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        if len(df) < 200:
            return 0, 0.0
        c = df["close"]
        e20 = float(Indicators.ema(c, 20).iloc[-1])
        e50 = float(Indicators.ema(c, 50).iloc[-1])
        e200 = float(Indicators.ema(c, 200).iloc[-1])
        last = float(c.iloc[-1])
        if last > e20 > e50 > e200:
            return 1, 0.85
        if last < e20 < e50 < e200:
            return -1, 0.85
        return 0, 0.0

    # Model 3: True Ratcheting Supertrend
    def _supertrend(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        trend, band, st_atr = Indicators.supertrend(df, period=10, mult=3.0)
        if trend == 0 or st_atr <= 0:
            return 0, 0.0
        last_close = float(df["close"].iloc[-1])
        conf = min(1.0, abs(last_close - band) / (st_atr * 2.0 + 1e-9))
        return trend, max(0.35, conf)

    # Model 4: Donchian Channel Breakout
    def _donchian(self, df: pd.DataFrame, atr: float, period: int = 20) -> Tuple[int, float]:
        if atr <= 0 or len(df) < period + 2:
            return 0, 0.0
        high_max = float(df["high"].rolling(period).max().iloc[-2])
        low_min = float(df["low"].rolling(period).min().iloc[-2])
        last = float(df["close"].iloc[-1])
        if last > high_max:
            return 1, min(1.0, (last - high_max) / atr)
        if last < low_min:
            return -1, min(1.0, (low_min - last) / atr)
        return 0, 0.0

    # Model 5: ATR-Calibrated Scale-Invariant Kalman Filter
    def _kalman(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        if atr <= 0 or len(df) < 30:
            return 0, 0.0
        atr_s = Indicators.atr_series(df, self.cfg.atr_period)
        k_est = Indicators.kalman_filter(df["close"], atr_s)
        slope = float(k_est.diff(3).iloc[-1])
        conf = min(1.0, abs(slope) / (atr * 0.25))
        if slope > 0.05 * atr:
            return 1, conf
        if slope < -0.05 * atr:
            return -1, conf
        return 0, 0.0

    # Model 6: MACD Momentum
    def _macd(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        if atr <= 0 or len(df) < 35:
            return 0, 0.0
        macd, sig, hist = Indicators.macd(df["close"])
        conf = min(1.0, abs(hist) / (atr * 0.5 + 1e-9))
        if macd > sig and hist > 0:
            return 1, conf
        if macd < sig and hist < 0:
            return -1, conf
        return 0, 0.0

    # Model 7: RSI Momentum / Mean-Reversion
    def _rsi(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        if len(df) < 20:
            return 0, 0.0
        r = Indicators.rsi(df["close"], 14)
        if r < 30:
            return 1, (30.0 - r) / 30.0
        if r > 70:
            return -1, (r - 70.0) / 30.0
        if r > 55:
            return 1, (r - 50.0) / 50.0
        if r < 45:
            return -1, (50.0 - r) / 50.0
        return 0, 0.0

    # Model 8: Stochastic Momentum
    def _stochastic(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        if len(df) < 22:
            return 0, 0.0
        k, d = Indicators.stochastic(df)
        if k < 20 and k > d:
            return 1, (20.0 - k) / 20.0
        if k > 80 and k < d:
            return -1, (k - 80.0) / 20.0
        return 0, 0.0

    # Model 9: Bollinger Bands
    def _bollinger(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        if len(df) < 25:
            return 0, 0.0
        upper, _, lower = Indicators.bollinger_bands(df["close"])
        last = float(df["close"].iloc[-1])
        band_w = upper - lower
        if band_w <= 0:
            return 0, 0.0
        if last <= lower:
            return 1, min(1.0, (lower - last) / band_w + 0.5)
        if last >= upper:
            return -1, min(1.0, (last - upper) / band_w + 0.5)
        return 0, 0.0

    # Model 10: Ichimoku Kinko Hyo (with Chikou Span confirmation)
    def _ichimoku(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        if len(df) < 78:
            return 0, 0.0
        c = df["close"]
        h = df["high"]
        l = df["low"]
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2.0
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2.0
        span_a = ((tenkan + kijun) / 2.0).shift(26)
        span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2.0).shift(26)

        last_c = float(c.iloc[-1])
        past_c26 = float(c.iloc[-27])
        k_val = float(kijun.iloc[-1])
        t_val = float(tenkan.iloc[-1])
        sa = float(span_a.iloc[-1])
        sb = float(span_b.iloc[-1])
        if not (np.isfinite(sa) and np.isfinite(sb)):
            return 0, 0.0

        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)
        if last_c > cloud_top and t_val > k_val and last_c > past_c26:
            return 1, 0.85
        if last_c < cloud_bottom and t_val < k_val and last_c < past_c26:
            return -1, 0.85
        return 0, 0.0

    # Model 11: Unmitigated Institutional SMC Order Block
    def _smc_orderblock(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        if atr <= 0:
            return 0, 0.0
        ob_b = self.detect_order_block(df, "BUY", atr=atr)
        ob_s = self.detect_order_block(df, "SELL", atr=atr)
        last = float(df["close"].iloc[-1])
        if ob_b and ob_b["low"] <= last <= ob_b["high"] + 0.35 * atr:
            return 1, 0.90
        if ob_s and ob_s["low"] - 0.35 * atr <= last <= ob_s["high"]:
            return -1, 0.90
        return 0, 0.0

    # Model 12: Strictly Causal Low-Pass Filter (Replaces non-causal filtfilt)
    def _causal_filter(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        if atr <= 0 or len(df) < 50:
            return 0, 0.0
        smooth = Indicators.causal_lowpass(df["close"], cutoff=0.15)
        ef = smooth.ewm(span=10, adjust=False).mean()
        es = smooth.ewm(span=30, adjust=False).mean()
        diff = float(ef.iloc[-1] - es.iloc[-1])
        slope = float(ef.diff(3).iloc[-1])
        conf = min(1.0, abs(diff) / atr)
        if diff > 0 and slope > 0:
            return 1, conf
        if diff < 0 and slope < 0:
            return -1, conf
        return 0, 0.0

    # Model 13: Session Anchored VWAP Bias
    def _session_vwap(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        if atr <= 0 or len(df) < 30:
            return 0, 0.0
        vwap = Indicators.session_vwap(df)
        last = float(df["close"].iloc[-1])
        v_last = float(vwap.iloc[-1])
        v_slope = float(vwap.diff(3).iloc[-1])
        dist = last - v_last
        conf = min(1.0, abs(dist) / (1.5 * atr))
        if dist > 0.1 * atr and v_slope > 0:
            return 1, max(0.3, conf)
        if dist < -0.1 * atr and v_slope < 0:
            return -1, max(0.3, conf)
        return 0, 0.0

    # Model 14: Purged Walk-Forward Symmetric Triple-Barrier ML Ensemble
    def _ml_ensemble(self, df: pd.DataFrame, atr: float) -> Tuple[int, float]:
        if not (HAS_SKLEARN and self.cfg.enable_ml_models) or len(df) < 320:
            return 0, 0.0
        try:
            c = df["close"]
            atr_s = Indicators.atr_series(df, self.cfg.atr_period)
            features = pd.DataFrame(index=df.index)
            features["ret_1"] = c.pct_change(1)
            features["ret_5"] = c.pct_change(5)
            features["ret_12"] = c.pct_change(12)
            features["atr_norm"] = atr_s / (c + 1e-9)
            features["rsi_14"] = Indicators.rsi_series(c, 14) / 100.0
            features["ema20_dist"] = (c - Indicators.ema(c, 20)) / (atr_s + 1e-9)
            features["ema50_dist"] = (c - Indicators.ema(c, 50)) / (atr_s + 1e-9)
            features = features.dropna()

            # Symmetric barriers (+1.5 ATR vs -1.5 ATR) prevent random-walk directional bias!
            horizon = 20
            barrier_mult = 1.5
            atr_v = atr_s.values
            closes = c.values
            highs = df["high"].values
            lows = df["low"].values
            n = len(closes)

            y_arr = np.full(n, -1, dtype=int)
            for i in range(30, n - horizon):
                if not np.isfinite(atr_v[i]) or atr_v[i] <= 0:
                    continue
                tp_up = closes[i] + barrier_mult * atr_v[i]
                tp_dn = closes[i] - barrier_mult * atr_v[i]
                hit_up = np.where(highs[i + 1 : i + horizon + 1] >= tp_up)[0]
                hit_dn = np.where(lows[i + 1 : i + horizon + 1] <= tp_dn)[0]
                t_up = int(hit_up[0]) if len(hit_up) else 999
                t_dn = int(hit_dn[0]) if len(hit_dn) else 999
                if t_up < t_dn:
                    y_arr[i] = 1
                elif t_dn < t_up:
                    y_arr[i] = 0
                else:
                    # Time-expiry sign fallback
                    y_arr[i] = 1 if closes[i + horizon] > closes[i] else 0

            y = pd.Series(y_arr, index=df.index)
            y = y[y >= 0]
            valid_idx = features.index.intersection(y.index)
            X_clean, y_clean = features.loc[valid_idx], y.loc[valid_idx]

            # Purged split: remove `horizon` embargo gap between train & validation to prevent leakage
            total_n = len(X_clean)
            split_tr_end = int(total_n * 0.75)
            split_val_start = split_tr_end + horizon
            if split_tr_end < 120 or (total_n - split_val_start) < 30:
                return 0, 0.0

            X_tr, y_tr = X_clean.iloc[:split_tr_end], y_clean.iloc[:split_tr_end]
            X_val, y_val = X_clean.iloc[split_val_start:], y_clean.iloc[split_val_start:]

            if len(np.unique(y_tr.values)) < 2 or len(np.unique(y_val.values)) < 2:
                return 0, 0.0

            clf = RandomForestClassifier(
                n_estimators=80,
                max_depth=4,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            )
            clf.fit(X_tr.values, y_tr.values)
            val_preds = clf.predict(X_val.values)
            bal_acc = float(balanced_accuracy_score(y_val.values, val_preds))
            if bal_acc < self.cfg.ml_min_val_acc:
                return 0, 0.0

            proba = clf.predict_proba(features.iloc[-1:].values)[0]
            classes = list(clf.classes_)
            prob_up = float(proba[classes.index(1)]) if 1 in classes else 0.5
            if prob_up > 0.60:
                return 1, min(1.0, (prob_up - 0.5) * 2.0)
            if prob_up < 0.40:
                return -1, min(1.0, (0.5 - prob_up) * 2.0)
            return 0, 0.0
        except Exception:
            return 0, 0.0

    def _build_model_registry(self) -> List[Tuple[str, Callable[[pd.DataFrame, float], Tuple[int, float]], float]]:
        registry: List[Tuple[str, Callable[[pd.DataFrame, float], Tuple[int, float]], float]] = [
            ("EMA_Cross", self._ema_cross, 1.0),
            ("EMA_Trend", self._ema_trend, 1.0),
            ("Supertrend", self._supertrend, 1.2),
            ("Donchian", self._donchian, 0.8),
            ("Kalman", self._kalman, 1.0),
            ("MACD", self._macd, 1.2),
            ("RSI", self._rsi, 1.0),
            ("Stochastic", self._stochastic, 0.9),
            ("BollingerBB", self._bollinger, 0.9),
            ("Ichimoku", self._ichimoku, 1.3),
            ("SMC_OB", self._smc_orderblock, 1.5),
            ("CausalFilter", self._causal_filter, 0.8),
            ("SessionVWAP", self._session_vwap, 1.1),
        ]
        if self.cfg.enable_ml_models:
            registry.append(("ML_Ensemble", self._ml_ensemble, 1.0))
        return registry

    def _classify_regime(self, h1: pd.DataFrame) -> Tuple[str, float, float]:
        hurst_val = Indicators.hurst(h1["close"])
        entropy_val = Indicators.shannon_entropy(h1["close"])

        if entropy_val >= self.cfg.entropy_chaos_max:
            return "CHAOTIC", hurst_val, entropy_val
        if hurst_val >= self.cfg.hurst_trend_min:
            return "TRENDING", hurst_val, entropy_val
        if hurst_val <= self.cfg.hurst_meanrev_max:
            return "MEAN_REVERTING", hurst_val, entropy_val
        return "RANGING", hurst_val, entropy_val

    @staticmethod
    def _htf_bias(h4: pd.DataFrame) -> str:
        if len(h4) < 50:
            return "NEUTRAL"
        c = h4["close"]
        e20, e50 = Indicators.ema(c, 20), Indicators.ema(c, 50)
        if float(c.iloc[-1]) > float(e50.iloc[-1]) and float(e20.iloc[-1]) > float(e50.iloc[-1]):
            return "BULLISH"
        if float(c.iloc[-1]) < float(e50.iloc[-1]) and float(e20.iloc[-1]) < float(e50.iloc[-1]):
            return "BEARISH"
        return "NEUTRAL"

    def generate(
        self, m5: pd.DataFrame, m15: pd.DataFrame, h1: pd.DataFrame, h4: pd.DataFrame
    ) -> SignalResult:
        res = SignalResult(total_models=len(self._registry))

        atr_val = Indicators.atr_last(m5, self.cfg.atr_period)
        adx_val = Indicators.adx_last(m5, self.cfg.atr_period)
        res.atr = round(atr_val, 4)
        res.adx = round(adx_val, 2)

        regime, H, E = self._classify_regime(h1)
        res.regime, res.hurst, res.entropy = regime, H, E

        h4_trend = self._htf_bias(h4)
        res.htf_bias = h4_trend

        # Evaluate Quantitative Parliament
        buy_weight = 0.0
        sell_weight = 0.0
        active_count = 0
        details: List[Dict[str, Any]] = []

        for name, model_fn, weight in self._registry:
            try:
                vote, conf = model_fn(m5, atr_val)
            except Exception as e:
                logging.debug(f"Model {name} encountered error: {e}")
                vote, conf = 0, 0.0

            conf = float(np.clip(conf, 0.0, 1.0))
            if vote == 0 or conf <= 0.0:
                details.append(
                    {"model": name, "vote": "NEUTRAL", "weight": weight, "conf": 0.0, "score": 0.0}
                )
                continue

            active_count += 1
            score = weight * conf
            if vote > 0:
                buy_weight += score
                v_label = "BUY"
            else:
                sell_weight += score
                v_label = "SELL"

            details.append(
                {
                    "model": name,
                    "vote": v_label,
                    "weight": weight,
                    "conf": round(conf, 2),
                    "score": round(score, 2),
                }
            )

        res.buy_score = round(buy_weight, 2)
        res.sell_score = round(sell_weight, 2)
        res.active_models = active_count
        res.votes_detail = details

        total_weight = buy_weight + sell_weight
        if total_weight > 1e-6:
            res.buy_ratio = buy_weight / total_weight
            res.sell_ratio = sell_weight / total_weight

        # Safety Gates
        if atr_val < self.cfg.min_atr_usd:
            res.skip_reason = f"Dead market ATR ${atr_val:.2f} < ${self.cfg.min_atr_usd:.2f} floor"
            return res

        if atr_val > self.cfg.max_atr_usd:
            res.skip_reason = f"Extreme flash-spike ATR ${atr_val:.2f} > ${self.cfg.max_atr_usd:.2f} cap"
            return res

        if adx_val < self.cfg.adx_min:
            res.skip_reason = f"ADX {adx_val:.1f} < {self.cfg.adx_min} (Low Trend Strength)"
            return res

        if regime == "CHAOTIC":
            res.skip_reason = f"Chaotic Market (Entropy={E:.2f} >= {self.cfg.entropy_chaos_max:.2f})"
            return res

        if regime == "RANGING" and not self.cfg.allow_ranging_regime:
            res.skip_reason = f"Ranging Regime (Hurst={H:.2f}, Entropy={E:.2f})"
            return res

        if total_weight < 1e-6:
            res.skip_reason = "No directional quantitative signal"
            return res

        if active_count < self.cfg.parliament_min_active:
            res.skip_reason = (
                f"Active models ({active_count}) below quorum ({self.cfg.parliament_min_active})"
            )
            return res

        dom_ratio = max(res.buy_ratio, res.sell_ratio)
        if dom_ratio >= self.cfg.parliament_full_ratio:
            res.verdict = "EXECUTE_FULL"
        elif dom_ratio >= self.cfg.parliament_half_ratio:
            res.verdict = "EXECUTE_HALF"
        elif dom_ratio >= self.cfg.parliament_min_ratio:
            res.verdict = "EXECUTE_MICRO"
        else:
            res.skip_reason = (
                f"Consensus ({dom_ratio:.0%}) below minimum ({self.cfg.parliament_min_ratio:.0%})"
            )
            return res

        # Multi-Timeframe H4 Alignment + M15 Confirmation Check
        proposed_dir = "BUY" if res.buy_ratio > res.sell_ratio else "SELL"
        if proposed_dir == "BUY" and h4_trend == "BEARISH" and self.cfg.enforce_htf_filter:
            if dom_ratio >= self.cfg.counter_trend_allow_ratio:
                res.direction = "BUY"
                if res.verdict == "EXECUTE_FULL":
                    res.verdict = "EXECUTE_HALF"
            else:
                res.verdict = "CANCEL"
                res.skip_reason = (
                    f"BUY blocked by H4 BEARISH HTF filter "
                    f"(Consensus {dom_ratio:.0%} < {self.cfg.counter_trend_allow_ratio:.0%})"
                )
                return res
        elif proposed_dir == "SELL" and h4_trend == "BULLISH" and self.cfg.enforce_htf_filter:
            if dom_ratio >= self.cfg.counter_trend_allow_ratio:
                res.direction = "SELL"
                if res.verdict == "EXECUTE_FULL":
                    res.verdict = "EXECUTE_HALF"
            else:
                res.verdict = "CANCEL"
                res.skip_reason = (
                    f"SELL blocked by H4 BULLISH HTF filter "
                    f"(Consensus {dom_ratio:.0%} < {self.cfg.counter_trend_allow_ratio:.0%})"
                )
                return res
        else:
            res.direction = proposed_dir

        return res

    @staticmethod
    def detect_order_block(
        df: pd.DataFrame, side: str, lookback: int = 40, atr: Optional[float] = None
    ) -> Optional[dict]:
        """Detects unmitigated Smart Money Concepts (SMC) Order Block zones."""
        if len(df) < lookback + 5:
            return None
        tail = df.tail(lookback + 5).reset_index(drop=True)
        atr_val = atr if (atr is not None and atr > 0) else Indicators.atr_last(df)
        if atr_val <= 0:
            return None

        for i in range(len(tail) - 2, 2, -1):
            body = abs(float(tail["close"].iloc[i]) - float(tail["open"].iloc[i]))
            if body < 1.2 * atr_val:
                continue
            is_bull = float(tail["close"].iloc[i]) > float(tail["open"].iloc[i])
            if side == "BUY" and not is_bull:
                continue
            if side == "SELL" and is_bull:
                continue

            for j in range(i - 1, max(i - 10, -1), -1):
                c_j = float(tail["close"].iloc[j])
                o_j = float(tail["open"].iloc[j])
                is_opposing = (c_j < o_j) if side == "BUY" else (c_j > o_j)
                if not is_opposing:
                    continue

                ob_high = float(tail["high"].iloc[j])
                ob_low = float(tail["low"].iloc[j])

                # Check if subsequent bars after impulse i already broke/invalidated the OB
                after = tail.iloc[i + 1 :]
                if not after.empty:
                    if side == "BUY" and float(after["low"].min()) < ob_low:
                        break  # Demand OB swept/mitigated
                    if side == "SELL" and float(after["high"].max()) > ob_high:
                        break  # Supply OB swept/mitigated

                return {"high": ob_high, "low": ob_low}
        return None

    def compute_levels(
        self, m5: pd.DataFrame, side: str, ob: Optional[dict]
    ) -> Tuple[float, float, float, float, float]:
        """Calculates Entry (Limit), SL (with minimum ATR noise floor), and TP1/TP2/TP3."""
        atr = Indicators.atr_last(m5, self.cfg.atr_period)
        if atr <= 0:
            raise ValueError("ATR cannot be zero or negative")

        price = float(m5["close"].iloc[-1])
        dec = self.cfg.price_decimals
        spread_buf = self.cfg.spread_buffer * 2.0
        min_sl_dist = max(atr * self.cfg.min_sl_atr_mult, 1.0)

        ob_high = ob["high"] if ob else price
        ob_low = ob["low"] if ob else price

        if side == "BUY":
            ob_candidate = (
                (ob_high - (ob_high - ob_low) * 0.5) if (ob and ob_high < price) else None
            )
            if ob_candidate is not None and (price - ob_candidate) <= self.cfg.entry_max_chase_atr * atr:
                entry = ob_candidate
                sl_ref = ob_low
            else:
                entry = price - atr * self.cfg.entry_pullback_atr
                sl_ref = entry - atr * self.cfg.atr_sl_mult

            entry = min(entry, price - atr * 0.05)
            if (price - entry) > self.cfg.entry_max_chase_atr * atr:
                entry = price - atr * self.cfg.entry_pullback_atr

            sl = min(sl_ref, entry - min_sl_dist) - spread_buf
            if entry <= sl:
                sl = entry - atr * self.cfg.atr_sl_mult - spread_buf

            tp1 = entry + atr * self.cfg.atr_tp1_mult
            tp2 = entry + atr * self.cfg.atr_tp2_mult
            tp3 = entry + atr * self.cfg.atr_tp3_mult

            entry_r, sl_r = round(entry, dec), round(sl, dec)
            tp1_r, tp2_r, tp3_r = round(tp1, dec), round(tp2, dec), round(tp3, dec)
            if not (sl_r < entry_r < tp1_r < tp2_r < tp3_r):
                raise ValueError("Invalid BUY price geometry after rounding")
            return entry_r, sl_r, tp1_r, tp2_r, tp3_r

        else:
            ob_candidate = (
                (ob_low + (ob_high - ob_low) * 0.5) if (ob and ob_low > price) else None
            )
            if ob_candidate is not None and (ob_candidate - price) <= self.cfg.entry_max_chase_atr * atr:
                entry = ob_candidate
                sl_ref = ob_high
            else:
                entry = price + atr * self.cfg.entry_pullback_atr
                sl_ref = entry + atr * self.cfg.atr_sl_mult

            entry = max(entry, price + atr * 0.05)
            if (entry - price) > self.cfg.entry_max_chase_atr * atr:
                entry = price + atr * self.cfg.entry_pullback_atr

            sl = max(sl_ref, entry + min_sl_dist) + spread_buf
            if sl <= entry:
                sl = entry + atr * self.cfg.atr_sl_mult + spread_buf

            tp1 = entry - atr * self.cfg.atr_tp1_mult
            tp2 = entry - atr * self.cfg.atr_tp2_mult
            tp3 = entry - atr * self.cfg.atr_tp3_mult

            entry_r, sl_r = round(entry, dec), round(sl, dec)
            tp1_r, tp2_r, tp3_r = round(tp1, dec), round(tp2, dec), round(tp3, dec)
            if not (tp3_r < tp2_r < tp1_r < entry_r < sl_r):
                raise ValueError("Invalid SELL price geometry after rounding")
            return entry_r, sl_r, tp1_r, tp2_r, tp3_r
