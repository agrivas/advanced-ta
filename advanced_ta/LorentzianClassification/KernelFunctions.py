import numpy as np
import pandas as pd

def rational_quadratic(src: pd.Series, lookback: int, relative_weight: float, window: int) -> np.ndarray:
    """
    True vectorized Rational Quadratic filter using 1D Convolution.
    Replaces the O(N*K) shift loop with O(N log K) convolution.
    """
    # 1. Create distance array (equivalent to 'i' in the original loop)
    i = np.arange(window + 2)
    
    # 2. Calculate the Rational Quadratic weights
    w = (1 + (i ** 2 / (lookback ** 2 * 2 * relative_weight))) ** -relative_weight
    
    # 3. Normalize weights so they sum to 1 (replaces cumulativeWeight)
    w = w / np.sum(w)
    
    # 4. Apply Causal Convolution
    # mode='full' automatically reverses the kernel to look backward (no look-ahead bias)
    # We slice [:len(src)] to trim the padding added by 'full' mode
    smoothed = np.convolve(src.values, w, mode='full')[:len(src)]
    
    # 5. Mask the warmup period (matching original val[:startAtBar + 1] = 0.0)
    smoothed[:window + 1] = 0.0
    
    return smoothed

def gaussian(src: pd.Series, lookback: int, window: int) -> np.ndarray:
    """
    True vectorized Gaussian filter using 1D Convolution.
    """
    i = np.arange(window + 2)
    
    # Calculate Gaussian weights (using numpy's vectorized exp instead of math.exp)
    w = np.exp(-(i ** 2) / (2 * lookback ** 2))
    w = w / np.sum(w)
    
    smoothed = np.convolve(src.values, w, mode='full')[:len(src)]
    smoothed[:window + 1] = 0.0
    
    return smoothed