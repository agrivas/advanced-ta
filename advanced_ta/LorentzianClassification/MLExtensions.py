import numpy as np
import pandas as pd
from ta.momentum import rsi as RSI
from ta.volatility import average_true_range as ATR
from ta.trend import cci as CCI, adx as ADX, ema_indicator as EMA, sma_indicator as SMA

# ==========================
# ==== Helper Functions ====
# ==========================

def log_return_zscore(src: pd.Series, window=200) -> np.array:
    """
    Standardizes a price series using a rolling Z-score of log-returns.
    This eliminates look-ahead bias and ensures geometric stationarity.
    """
    log_ret = np.log(src / src.shift(1))
    rolling_mean = log_ret.rolling(window=window).mean()
    rolling_std = log_ret.rolling(window=window).std()
    
    z_score = (log_ret - rolling_mean) / rolling_std
    return z_score.fillna(0).values

def rolling_zscore(src: pd.Series, window=200) -> np.array:
    """
    Standardizes an oscillator (which may cross zero) using a rolling Z-score.
    """
    rolling_mean = src.rolling(window=window).mean()
    rolling_std = src.rolling(window=window).std()
    
    z_score = (src - rolling_mean) / rolling_std
    return z_score.fillna(0).values

def rescale_to_unit(src: np.array) -> np.array:
    """
    Squashes a Z-score into a [0, 1] range using a Sigmoid function.
    Handles outliers much better than Min-Max scaling.
    """
    return 1 / (1 + np.exp(-src))

# ==========================
# ==== Feature Functions ====
# ==========================

def n_rsi(src: pd.Series, n1=14, n2=1) -> np.array:
    """
    Normalized RSI based on Log-Price momentum.
    """
    log_src = np.log(src)
    rsi_val = EMA(RSI(log_src, n1), n2)
    # RSI is strictly bounded [0, 100], so linear scaling to [0, 1] is safe
    return rsi_val.fillna(50).values / 100.0

def n_cci(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1=20, n2=1) -> np.array:
    """
    Normalized CCI. Uses log-prices and Z-score standardization.
    """
    log_h = np.log(highSrc)
    log_l = np.log(lowSrc)
    log_c = np.log(closeSrc)
    
    cci_val = CCI(log_h, log_l, log_c, n1)
    smoothed_cci = EMA(cci_val, n2)
    
    # CCI is unbounded and crosses zero; apply rolling Z-score then Sigmoid
    z_cci = rolling_zscore(smoothed_cci, window=200)
    return rescale_to_unit(z_cci)

def n_wt(src: pd.Series, n1=10, n2=11) -> np.array:
    """
    Log-Stationary WaveTrend. Uses log-price to ensure percentage-based cycles.
    """
    log_src = np.log(src)
    ema1 = EMA(log_src, n1)
    ema2 = EMA(np.abs(log_src - ema1), n1)
    ci = (log_src - ema1) / (0.015 * ema2)
    wt1 = EMA(ci, n2) 
    wt2 = SMA(wt1, 4)
    
    wave_trend = wt1 - wt2
    # WaveTrend crosses zero; apply rolling Z-score then Sigmoid
    z_wt = rolling_zscore(wave_trend, window=200)
    return rescale_to_unit(z_wt)

def n_adx(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1=14) -> np.array:
    """
    Normalized ADX. Measures trend strength independent of price scale.
    """
    log_h = np.log(highSrc)
    log_l = np.log(lowSrc)
    log_c = np.log(closeSrc)
    
    adx_val = ADX(log_h, log_l, log_c, n1)
    # ADX is bounded [0, 100]
    return adx_val.fillna(0).values / 100.0

# ==========================
# ==== Advanced Filters ====
# ==========================

def regime_filter(src: pd.Series, high: pd.Series, low: pd.Series, useRegimeFilter: bool, threshold: float) -> np.array:
    """
    Log-Velocity Regime Filter. 
    Detects when the 'Percentage Velocity' of the market exceeds historical norms.
    """
    if not useRegimeFilter: 
        return np.array([True] * len(src))

    log_close = np.log(src)
    log_high = np.log(high)
    log_low = np.log(low)

    def klmf(c_src, h, l):
        v1 = np.zeros_like(c_src)
        v2 = np.zeros_like(c_src)
        out = np.zeros_like(c_src)
        for i in range(1, len(c_src)):
            # Velocity of log-price change
            v1[i] = 0.2 * (c_src[i] - c_src[i-1]) + 0.8 * v1[i-1]
            # Volatility (Log High-Low Range)
            v2[i] = 0.1 * (h[i] - l[i]) + 0.8 * v2[i-1]
            
            omega = np.abs(v1[i] / v2[i]) if v2[i] != 0 else 0
            alpha = (-(omega**2) + np.sqrt(omega**4 + 16*omega**2)) / 8
            out[i] = alpha * c_src[i] + (1 - alpha) * out[i-1]
        return out

    kf_curve = klmf(log_close.values, log_high.values, log_low.values)
    
    # Slope of the log-filtered curve (Percentage Change Velocity)
    slope = np.abs(np.diff(kf_curve, prepend=kf_curve[0]))
    
    # Use Rolling Mean of slope to create a dynamic threshold
    ema_slope = EMA(pd.Series(slope), 200).values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_slope = np.nan_to_num((slope - ema_slope) / ema_slope, nan=0.0)
    
    return normalized_slope >= threshold

def filter_adx(src: pd.Series, high: pd.Series, low: pd.Series, adxThreshold: float, useAdxFilter: bool, length=14) -> np.array:
    """
    ADX Filter applied to log prices to maintain geometric symmetry.
    """
    if not useAdxFilter: 
        return np.array([True] * len(src))
        
    log_h = np.log(high)
    log_l = np.log(low)
    log_c = np.log(src)
    
    adx_series = ADX(log_h, log_l, log_c, length).values
    return np.nan_to_num(adx_series, nan=0.0) >= adxThreshold

def filter_volatility(high: pd.Series, low: pd.Series, close: pd.Series, useVolatilityFilter: bool, minLength=1, maxLength=10) -> np.array:
    """
    Volatility filter using Log-ATR. Ensures that short-term percentage volatility
    is compared correctly against long-term percentage volatility.
    """
    if not useVolatilityFilter: 
        return np.array([True] * len(close))
        
    log_h = np.log(high)
    log_l = np.log(low)
    log_c = np.log(close)
    
    recent_atr = ATR(log_h, log_l, log_c, minLength).values
    historical_atr = ATR(log_h, log_l, log_c, maxLength).values
    
    return np.nan_to_num(recent_atr, nan=0.0) > np.nan_to_num(historical_atr, nan=0.0)