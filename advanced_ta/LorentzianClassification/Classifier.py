import pandas as pd
import numpy as np
from ta.trend import ema_indicator as EMA, sma_indicator as SMA
from . import MLExtensions as ml
from . import KernelFunctions as kernels
from .Types import Direction, Feature, Settings, FilterSettings, KernelFilter, Filter

class LorentzianClassification:
    # Expose types as class attributes for backward compatibility
    Feature = Feature
    Settings = Settings
    FilterSettings = FilterSettings
    KernelFilter = KernelFilter
    Filter = Filter
    Direction = Direction

    def __init__(self, data: pd.DataFrame, features: list = None, settings: Settings = None, filterSettings: FilterSettings = None):
        self.df = data.copy()
        self.settings = settings or Settings(source='close')
        self.filterSettings = filterSettings or FilterSettings()

        # 1. Feature Generation (using the updated MLExtensions.py logic)
        self.feature_data = self._generate_features(features)

        # 2. Kernel Smoother (using the updated KernelFunctions.py logic)
        k = self.filterSettings.kernelFilter
        self.yhat1 = kernels.rational_quadratic(self.df['close'], k.lookbackWindow, k.relativeWeight, k.regressionLevel)
        self.yhat2 = kernels.gaussian(self.df['close'], k.lookbackWindow - k.crossoverLag, k.regressionLevel)

        # 3. Main Classification Execution
        self._classify()

    def _generate_features(self, features):
        """Generates the standardized feature matrix."""
        feature_list = []
        # Default features if none provided
        features = features or [
            Feature("RSI", 14, 2),
            Feature("WT", 10, 11),
            Feature("CCI", 20, 2),
            Feature("ADX", 20, 2)
        ]

        for f in features:
            if isinstance(f, pd.Series):
                feature_list.append(f.values)
            else:
                match f.type:
                    case "RSI": feature_list.append(ml.n_rsi(self.df['close'], f.param1, f.param2))
                    case "WT":  feature_list.append(ml.n_wt(self.df['close'], f.param1, f.param2))
                    case "CCI": feature_list.append(ml.n_cci(self.df['high'], self.df['low'], self.df['close'], f.param1, f.param2))
                    case "ADX": feature_list.append(ml.n_adx(self.df['high'], self.df['low'], self.df['close'], f.param1))

        return np.array(feature_list).T # Shape: (Bars, Features)

    def _classify(self):
        """Core ML Logic using optimized Lorentzian Distance."""
        # Create Training Labels: 1 for Long, -1 for Short (based on 4-bar lookahead)
        # CRITICAL: Clean NaN values from shift(-4) to prevent NaN poisoning
        y_train = np.where(self.df['close'].shift(-4) > self.df['close'], 1, -1)
        y_train = np.nan_to_num(y_train, nan=0).astype(int)

        # Vectorized Lorentzian Distance Calculation
        predictions = []
        max_bars = self.settings.maxBarsBack

        for i in range(len(self.df)):
            if i < 200: # Warmup period for Z-scores/Kernels
                predictions.append(0)
                continue

            # Only look back at the historical window (prevents future leakage)
            start_idx = max(0, i - max_bars)
            historical_features = self.feature_data[start_idx:i]
            current_features = self.feature_data[i]

            # Lorentzian Distance: sum(log(1 + |feat_hist - feat_curr|))
            distances = np.sum(np.log(1 + np.abs(historical_features - current_features)), axis=1)

            # Find K-Nearest Neighbors
            k_indices = np.argsort(distances)[:self.settings.neighborsCount]
            prediction = np.sum(y_train[start_idx + k_indices])
            predictions.append(prediction)

        self.df['prediction'] = predictions
        self._apply_filters()

    def _apply_filters(self):
        """Apply Regime, Volatility, and Kernel filters to generate signals."""
        # Volatility & Regime Filters from ml
        vol_filter = ml.filter_volatility(self.df['high'], self.df['low'], self.df['close'], self.filterSettings.useVolatilityFilter)
        regime_filter = ml.regime_filter(self.df['close'], self.df['high'], self.df['low'], self.filterSettings.useRegimeFilter, self.filterSettings.regimeThreshold)

        # ADX Filter
        adx_filter = ml.filter_adx(self.df['close'], self.df['high'], self.df['low'], self.filterSettings.adxThreshold, self.filterSettings.useAdxFilter, self.filterSettings.adxLength)

        # Kernel Logic: Bullish if RQ is rising or GC crosses above RQ
        is_bullish_kernel = (self.yhat1 > np.roll(self.yhat1, 1)) | (self.yhat2 > self.yhat1)
        is_bearish_kernel = (self.yhat1 < np.roll(self.yhat1, 1)) | (self.yhat2 < self.yhat1)

        # Combine everything into a clean signal
        self.df['signal'] = 0
        long_cond = (self.df['prediction'] > 0) & vol_filter & regime_filter & adx_filter & is_bullish_kernel
        short_cond = (self.df['prediction'] < 0) & vol_filter & regime_filter & adx_filter & is_bearish_kernel

        self.df.loc[long_cond, 'signal'] = Direction.LONG
        self.df.loc[short_cond, 'signal'] = Direction.SHORT

        # EMA Filter
        if self.settings.useEmaFilter:
            ema = EMA(self.df['close'], self.settings.emaPeriod)
            self.df['isEmaUptrend'] = self.df['close'] > ema
            self.df['isEmaDowntrend'] = self.df['close'] < ema
            self.df.loc[~self.df['isEmaUptrend'], 'signal'] = 0
            self.df.loc[~self.df['isEmaDowntrend'], 'signal'] = 0
        else:
            self.df['isEmaUptrend'] = True
            self.df['isEmaDowntrend'] = True

        # SMA Filter
        if self.settings.useSmaFilter:
            sma = SMA(self.df['close'], self.settings.smaPeriod)
            self.df['isSmaUptrend'] = self.df['close'] > sma
            self.df['isSmaDowntrend'] = self.df['close'] < sma
            self.df.loc[~self.df['isSmaUptrend'], 'signal'] = 0
            self.df.loc[~self.df['isSmaDowntrend'], 'signal'] = 0
        else:
            self.df['isSmaUptrend'] = True
            self.df['isSmaDowntrend'] = True

        # Signal state tracking
        self._compute_signal_states()

    def _compute_signal_states(self):
        """Compute derived signal state columns for trade logic."""
        df = self.df
        signal = df['signal']

        # isLastSignalBuy / isLastSignalSell
        df['isLastSignalBuy'] = signal == Direction.LONG
        df['isLastSignalSell'] = signal == Direction.SHORT

        # isNewBuySignal: signal flips to LONG from non-LONG
        prev_signal = signal.shift(1).fillna(0).astype(int)
        df['isNewBuySignal'] = (signal == Direction.LONG) & (prev_signal != Direction.LONG)
        df['isNewSellSignal'] = (signal == Direction.SHORT) & (prev_signal != Direction.SHORT)

        # isEarlySignalFlip: signal flipped within last 3 bars
        df['isEarlySignalFlip'] = df['isNewBuySignal'] | df['isNewSellSignal']
        for i in range(1, 3):
            df['isEarlySignalFlip'] |= df['isNewBuySignal'].shift(i).astype(bool).fillna(False) | df['isNewSellSignal'].shift(i).astype(bool).fillna(False)

        # barsHeld: count of consecutive bars with same non-zero signal
        df['barsHeld'] = 0
        for i in range(len(df)):
            if signal.iloc[i] == 0:
                df.iloc[i, df.columns.get_loc('barsHeld')] = 0
            elif i == 0 or signal.iloc[i] != signal.iloc[i-1]:
                df.iloc[i, df.columns.get_loc('barsHeld')] = 1
            else:
                df.iloc[i, df.columns.get_loc('barsHeld')] = df.iloc[i-1, df.columns.get_loc('barsHeld')] + 1

        # Trade tracking columns
        df['startLongTrade'] = np.nan
        df['startShortTrade'] = np.nan
        df['endLongTrade'] = np.nan
        df['endShortTrade'] = np.nan

        in_long = False
        in_short = False
        entry_price = None

        for i in range(len(df)):
            if df['isNewBuySignal'].iloc[i]:
                if in_short:
                    df.iloc[i, df.columns.get_loc('endShortTrade')] = df['close'].iloc[i]
                    in_short = False
                in_long = True
                entry_price = df['close'].iloc[i]
                df.iloc[i, df.columns.get_loc('startLongTrade')] = entry_price

            elif df['isNewSellSignal'].iloc[i]:
                if in_long:
                    df.iloc[i, df.columns.get_loc('endLongTrade')] = df['close'].iloc[i]
                    in_long = False
                in_short = True
                entry_price = df['close'].iloc[i]
                df.iloc[i, df.columns.get_loc('startShortTrade')] = entry_price

            elif in_long and signal.iloc[i] != Direction.LONG:
                df.iloc[i, df.columns.get_loc('endLongTrade')] = df['close'].iloc[i]
                in_long = False

            elif in_short and signal.iloc[i] != Direction.SHORT:
                df.iloc[i, df.columns.get_loc('endShortTrade')] = df['close'].iloc[i]
                in_short = False

    @property
    def data(self) -> pd.DataFrame:
        return self.df

    def dump(self, path: str):
        """Save the result DataFrame to CSV."""
        self.df.to_csv(path)
