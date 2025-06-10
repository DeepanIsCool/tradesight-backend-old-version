import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # Use StandardScaler as in training
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D,
    BatchNormalization, Bidirectional, MultiHeadAttention,
    GlobalAveragePooling1D, GRU
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from datetime import datetime, timedelta
import uuid
import warnings
import os
import joblib # <-- Import joblib for saving scaler/pca
import pytz # <-- Import pytz for timezone handling in prediction
import json # <-- Import json for potential output formatting

warnings.filterwarnings('ignore')

# --- Class for calculating the technical indicators (Keep as is) ---
class CustomTechnicalIndicators:
    @staticmethod
    def calculate_adx(high, low, close, window=14):
        # ... (implementation unchanged) ...
        if len(high) != len(low) or len(low) != len(close):
            raise ValueError("Input arrays must have the same length")
        # Add robust check for sufficient data length BEFORE calculations
        min_len_adx = window * 2 # Need at least 2*window - 1, using window*2 is safer buffer
        if len(high) < min_len_adx:
             # Return zeros or NaNs of appropriate length instead of raising error immediately
             # This allows processing shorter recent data in prediction, though indicators might be less reliable
             print(f"Warning: Insufficient data for ADX (need {min_len_adx}, got {len(high)}). Returning zeros.")
             return np.zeros_like(high, dtype=float)
             # raise ValueError(f"Input length for ADX must be at least {min_len_adx}, got {len(high)}")

        # Handle potential NaNs before calculation if not handled by caller
        high = np.nan_to_num(np.array(high, dtype=float))
        low = np.nan_to_num(np.array(low, dtype=float))
        close = np.nan_to_num(np.array(close, dtype=float))

        # Ensure no NaNs remain after conversion
        if np.any(np.isnan(high)) or np.any(np.isnan(low)) or np.any(np.isnan(close)):
             print("Warning: NaNs detected in ADX input even after nan_to_num. Returning zeros.")
             return np.zeros_like(high, dtype=float)
             # raise ValueError("Input arrays contain NaN values after nan_to_num in ADX")

        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr[0] = tr1[0] # Avoid using rolled value for the first element

        dmplus = np.where(high - np.roll(high, 1) > np.roll(low, 1) - low,
                        np.maximum(high - np.roll(high, 1), 0), 0)
        dmminus = np.where(np.roll(low, 1) - low > high - np.roll(high, 1),
                        np.maximum(np.roll(low, 1) - low, 0), 0)
        dmplus[0] = dmminus[0] = 0 # Avoid rolled value for first element

        tr_ema = np.zeros_like(tr)
        dmplus_ema = np.zeros_like(dmplus)
        dmminus_ema = np.zeros_like(dmminus)

        # Check again for sufficient length for the initial mean calculation
        if len(tr) < window:
            print(f"Warning: Length {len(tr)} less than window {window} before ADX EMA loop. Returning zeros.")
            return np.zeros_like(high, dtype=float)

        # Calculate initial mean safely
        tr_ema[window-1] = np.mean(tr[1:window]) # Use 1:window to avoid index 0 issues after roll
        dmplus_ema[window-1] = np.mean(dmplus[1:window])
        dmminus_ema[window-1] = np.mean(dmminus[1:window])

        # Calculate EMAs
        alpha = 1.0 / window # Correct alpha calculation
        for i in range(window, len(tr)):
            tr_ema[i] = tr[i] * alpha + tr_ema[i-1] * (1 - alpha)
            dmplus_ema[i] = dmplus[i] * alpha + dmplus_ema[i-1] * (1 - alpha)
            dmminus_ema[i] = dmminus[i] * alpha + dmminus_ema[i-1] * (1 - alpha)

        # Fill initial values
        tr_ema[:window-1] = tr_ema[window-1]
        dmplus_ema[:window-1] = dmplus_ema[window-1]
        dmminus_ema[:window-1] = dmminus_ema[window-1]


        # Calculate DIs safely checking for division by zero
        plus_di = np.zeros_like(tr_ema)
        minus_di = np.zeros_like(tr_ema)
        # Use a small epsilon to avoid division by zero
        epsilon = 1e-10
        nonzero_indices = tr_ema > epsilon
        plus_di[nonzero_indices] = 100 * (dmplus_ema[nonzero_indices] / (tr_ema[nonzero_indices] + epsilon))
        minus_di[nonzero_indices] = 100 * (dmminus_ema[nonzero_indices] / (tr_ema[nonzero_indices] + epsilon))


        # Calculate DX safely
        dx = np.zeros_like(plus_di)
        di_sum = plus_di + minus_di
        nonzero_sum_indices = di_sum > epsilon
        dx[nonzero_sum_indices] = 100 * np.abs(plus_di[nonzero_sum_indices] - minus_di[nonzero_sum_indices]) / (di_sum[nonzero_sum_indices] + epsilon)

        # Calculate ADX EMA
        adx = np.zeros_like(dx)
        adx_start_index = window - 1 + window -1 # Index for first ADX value (2*window - 2)
        if len(dx) < adx_start_index + 1:
             print(f"Warning: Insufficient length for final ADX calculation (need {adx_start_index + 1}, got {len(dx)}). Returning zeros.")
             return np.zeros_like(high, dtype=float)

        # Initial ADX is the average of the first 'window' DX values starting from dx[window-1]
        adx[adx_start_index] = np.mean(dx[window-1 : adx_start_index+1])

        alpha_adx = 1.0/window # Same alpha
        for i in range(adx_start_index + 1, len(dx)):
            adx[i] = (adx[i-1] * (window - 1) + dx[i]) / window # Smoothed Moving Average formula often used for ADX

        # Fill initial part of ADX array
        adx[:adx_start_index] = adx[adx_start_index] # Fill initial values

        return np.nan_to_num(adx, nan=0.0) # Final safety check


    @staticmethod
    def calculate_rsi(close, window=14):
        # ... (implementation unchanged, ensure sufficient length check) ...
        if not isinstance(close, np.ndarray):
             close = np.array(close, dtype=float)
        close = np.squeeze(close)
        if close.ndim != 1:
             raise ValueError(f"Expected 1D array for close, got shape {close.shape}")

        # Add robust check for sufficient data length BEFORE calculations
        if len(close) <= window: # Need strictly more than window
             print(f"Warning: Insufficient data for RSI (need > {window}, got {len(close)}). Returning zeros.")
             return np.zeros_like(close, dtype=float)
             # raise ValueError(f"Input length {len(close)} must be greater than {window} for RSI")

        # Handle potential NaNs before calculation
        close = np.nan_to_num(close)
        if np.any(np.isnan(close)):
             print("Warning: NaNs detected in RSI input even after nan_to_num. Returning zeros.")
             return np.zeros_like(close, dtype=float)
             # raise ValueError("Input array contains NaN values after nan_to_num in RSI")


        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # Use Simple Moving Average for initial values
        avg_gain = np.zeros_like(close, dtype=float)
        avg_loss = np.zeros_like(close, dtype=float)

        # Check sufficient length for initial mean
        if len(gain) < window: # Need 'window' number of gains/losses
             print(f"Warning: Insufficient gains/losses ({len(gain)}) for initial RSI average (window {window}). Returning zeros.")
             return np.zeros_like(close, dtype=float)

        avg_gain[window] = np.mean(gain[:window])
        avg_loss[window] = np.mean(loss[:window])

        # Use Wilder's Smoothing Method (similar to EMA)
        for i in range(window + 1, len(close)):
            avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i - 1]) / window
            avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i - 1]) / window

        # Calculate RS
        rs = np.zeros_like(close, dtype=float)
        # Avoid division by zero
        epsilon = 1e-10
        non_zero_loss_indices = avg_loss > epsilon
        rs[non_zero_loss_indices] = avg_gain[non_zero_loss_indices] / avg_loss[non_zero_loss_indices]

        # Calculate RSI
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Set RSI=100 where avg_loss is zero but avg_gain is positive
        rsi[ (avg_loss <= epsilon) & (avg_gain > epsilon) ] = 100.0
        # Set RSI=0 where both are zero (or negligible) - debatable, 50 might be better? Let's use 50.
        rsi[ (avg_loss <= epsilon) & (avg_gain <= epsilon) ] = 50.0

        # Fill initial values
        rsi[:window] = rsi[window] # Fill the start with the first calculated value

        return np.nan_to_num(rsi, nan=50.0) # Return 50 for any remaining NaNs (neutral)

    # ... (other indicator methods - ADD similar length and NaN checks if needed) ...
    @staticmethod
    def calculate_obv(close, volume):
        """Calculate On-Balance Volume (OBV)"""
        # Ensure numpy arrays and handle NaNs
        close = np.nan_to_num(np.array(close, dtype=float))
        volume = np.nan_to_num(np.array(volume, dtype=float))

        if len(close) == 0 or len(volume) == 0:
            return np.array([], dtype=float)
        if len(close) != len(volume):
            raise ValueError("Close and Volume must have the same length for OBV")

        obv = np.zeros_like(close, dtype=float)
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv

    @staticmethod
    def calculate_macd(close, short_window=12, long_window=26, signal_window=9):
        """Calculate Moving Average Convergence Divergence (MACD)"""
        close = np.nan_to_num(np.array(close, dtype=float))
        if len(close) < long_window:
             print(f"Warning: Insufficient data for MACD (need {long_window}, got {len(close)}). Returning zeros.")
             return np.zeros_like(close), np.zeros_like(close)

        # Calculate short and long EMAs
        ema_short = pd.Series(close).ewm(span=short_window, adjust=False).mean().values
        ema_long = pd.Series(close).ewm(span=long_window, adjust=False).mean().values

        macd_line = ema_short - ema_long

        # Calculate signal line (EMA of MACD line)
        signal_line = pd.Series(macd_line).ewm(span=signal_window, adjust=False).mean().values

        # Fill initial NaNs generated by EWM (optional, can let them be handled later)
        # macd_line = np.nan_to_num(macd_line)
        # signal_line = np.nan_to_num(signal_line)

        return macd_line, signal_line

    @staticmethod
    def calculate_stochastic(high, low, close, window=14, smooth_window=3):
        """Calculate Stochastic Oscillator"""
        high = np.nan_to_num(np.array(high, dtype=float))
        low = np.nan_to_num(np.array(low, dtype=float))
        close = np.nan_to_num(np.array(close, dtype=float))
        min_len = window + smooth_window - 1
        if len(close) < min_len:
            print(f"Warning: Insufficient data for Stochastic (need {min_len}, got {len(close)}). Returning zeros.")
            return np.zeros_like(close), np.zeros_like(close)

        # Rolling min/max calculation
        lowest_low = pd.Series(low).rolling(window=window).min()
        highest_high = pd.Series(high).rolling(window=window).max()

        # Calculate %K
        k_line = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10)) # Add epsilon

        # Calculate %D (SMA of %K)
        d_line = k_line.rolling(window=smooth_window).mean()

        # Fill initial NaNs
        k_line = k_line.fillna(50.0).values # Fill NaNs with 50 (neutral)
        d_line = d_line.fillna(50.0).values # Fill NaNs with 50 (neutral)

        return k_line, d_line


    @staticmethod
    def calculate_bollinger_bands(close, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        close = np.nan_to_num(np.array(close, dtype=float))
        if len(close) < window:
             print(f"Warning: Insufficient data for Bollinger Bands (need {window}, got {len(close)}). Returning zeros.")
             return np.zeros_like(close), np.zeros_like(close)

        series = pd.Series(close)
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        # Fill initial NaNs with close price or first valid band value
        first_valid_idx = window - 1
        upper_band[:first_valid_idx] = upper_band[first_valid_idx] if len(upper_band) > first_valid_idx else close[0]
        lower_band[:first_valid_idx] = lower_band[first_valid_idx] if len(lower_band) > first_valid_idx else close[0]
        upper_band = upper_band.fillna(method='ffill').fillna(method='bfill').values
        lower_band = lower_band.fillna(method='ffill').fillna(method='bfill').values


        return upper_band, lower_band

    @staticmethod
    def detect_support_resistance(high, low, window=20):
        """Simple Support and Resistance Detection using rolling min/max"""
        high = np.nan_to_num(np.array(high, dtype=float))
        low = np.nan_to_num(np.array(low, dtype=float))
        if len(low) < window + 1: # Need window + current point
            print(f"Warning: Insufficient data for Support/Resistance (need {window+1}, got {len(low)}). Returning zeros.")
            return np.zeros_like(low), np.zeros_like(high)

        support = np.zeros_like(low)
        resistance = np.zeros_like(high)

        # A point is support if it's the minimum of the *next* 'window' points
        # A point is resistance if it's the maximum of the *next* 'window' points
        # This requires looking ahead, which might not be ideal for causal prediction features.
        # Let's use a simpler definition: Is the current low the lowest in the last 'window' periods?
        # Is the current high the highest in the last 'window' periods?

        rolling_low = pd.Series(low).rolling(window=window, closed='left').min() # Min of previous 'window'
        rolling_high = pd.Series(high).rolling(window=window, closed='left').max() # Max of previous 'window'

        # Support: current low equals the rolling low of the *past* window
        support = np.where(low <= rolling_low, 1, 0)
        # Resistance: current high equals the rolling high of the *past* window
        resistance = np.where(high >= rolling_high, 1, 0)

        # Fill initial NaNs caused by rolling window
        support[:window] = 0
        resistance[:window] = 0

        return support, resistance


    @staticmethod
    def detect_candle_patterns(open_price, high, low, close):
        """Basic Candle Pattern Detection (Doji example)"""
        open_price = np.nan_to_num(np.array(open_price, dtype=float))
        high = np.nan_to_num(np.array(high, dtype=float))
        low = np.nan_to_num(np.array(low, dtype=float))
        close = np.nan_to_num(np.array(close, dtype=float))

        if not all(len(arr) == len(close) for arr in [open_price, high, low]):
             raise ValueError("OHLC must have the same length for candle patterns")
        if len(close) == 0:
            return np.array([], dtype=int)


        patterns = np.zeros_like(close, dtype=int) # Use int for flags
        epsilon = 1e-6 # Avoid division by zero

        body_size = np.abs(close - open_price)
        total_range = high - low

        # Define threshold relative to total range
        doji_threshold_factor = 0.1
        is_doji = body_size <= (total_range + epsilon) * doji_threshold_factor

        patterns[is_doji] = 1 # Assign 1 if it's a Doji

        return patterns


    @staticmethod
    def calculate_supertrend(high, low, close, period=14, multiplier=3):
        """Calculate SuperTrend indicator using pandas for ATR"""
        high = np.nan_to_num(np.array(high, dtype=float))
        low = np.nan_to_num(np.array(low, dtype=float))
        close = np.nan_to_num(np.array(close, dtype=float))
        min_len = period # ATR needs 'period'
        if len(close) < min_len:
            print(f"Warning: Insufficient data for Supertrend ATR (need {min_len}, got {len(close)}). Returning zeros.")
            return np.zeros_like(close), np.zeros_like(close, dtype=int)

        # Calculate True Range (TR)
        tr1 = pd.Series(high - low)
        tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
        tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Average True Range (ATR) using RMA/Wilder's Smoothing
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        # Fill initial NaNs in ATR
        atr = atr.fillna(method='bfill').fillna(0) # Backfill first, then fill remaining with 0

        # Calculate HL2 (High + Low) / 2
        hl2 = (high + low) / 2

        # Calculate Upper and Lower Bands
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        # Initialize SuperTrend arrays
        supertrend = np.zeros_like(close, dtype=float)
        uptrend = np.ones_like(close, dtype=int) # Start assuming uptrend

        # Check length again before loop
        if len(close) < 1:
            return np.zeros_like(close), np.zeros_like(close, dtype=int)

        # Set initial Supertrend value (e.g., based on first lower band)
        supertrend[0] = lower_band.iloc[0] if not lower_band.empty else close[0]

        # Iterate to calculate SuperTrend
        for i in range(1, len(close)):
            curr_close = close[i]
            prev_close = close[i-1]
            prev_supertrend = supertrend[i-1]
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]

            # If previous trend was up (SuperTrend == previous lower band)
            if prev_supertrend == lower_band.iloc[i-1]:
                if curr_close > prev_supertrend:
                    supertrend[i] = max(prev_supertrend, curr_lower) # Trend continues up
                    uptrend[i] = 1
                else:
                    supertrend[i] = curr_upper # Trend reverses down
                    uptrend[i] = 0
            # If previous trend was down (SuperTrend == previous upper band)
            else: # prev_supertrend == upper_band.iloc[i-1]
                if curr_close < prev_supertrend:
                    supertrend[i] = min(prev_supertrend, curr_upper) # Trend continues down
                    uptrend[i] = 0
                else:
                    supertrend[i] = curr_lower # Trend reverses up
                    uptrend[i] = 1

        return np.nan_to_num(supertrend, nan=close[0]), uptrend

    @staticmethod
    def calculate_ichimoku(high, low, close, tenkan_period=9, kijun_period=26, senkou_span_b_period=52, displacement=26):
        """Calculate Ichimoku Cloud components using pandas"""
        high_s = pd.Series(np.nan_to_num(np.array(high, dtype=float)))
        low_s = pd.Series(np.nan_to_num(np.array(low, dtype=float)))
        close_s = pd.Series(np.nan_to_num(np.array(close, dtype=float)))

        min_len = max(tenkan_period, kijun_period, senkou_span_b_period, displacement + 1)
        if len(close) < min_len:
            print(f"Warning: Insufficient data for Ichimoku (need {min_len}, got {len(close)}). Returning zero arrays.")
            zeros = np.zeros_like(close_s.values)
            return zeros, zeros, zeros, zeros, zeros, zeros

        # Tenkan-sen (Conversion Line)
        tenkan_high = high_s.rolling(window=tenkan_period).max()
        tenkan_low = low_s.rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        kijun_high = high_s.rolling(window=kijun_period).max()
        kijun_low = low_s.rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A) - shifted forward
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

        # Senkou Span B (Leading Span B) - shifted forward
        senkou_b_high = high_s.rolling(window=senkou_span_b_period).max()
        senkou_b_low = low_s.rolling(window=senkou_span_b_period).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(displacement)

        # Chikou Span (Lagging Span) - shifted backward
        # chikou_span = close_s.shift(-displacement) # Not typically used as a direct feature for future prediction

        # Cloud Strength (difference between spans) - use non-shifted for current comparison?
        # Let's calculate current cloud boundaries for breakout signal
        current_senkou_a = ((tenkan_sen + kijun_sen) / 2)
        current_senkou_b = ((high_s.rolling(window=senkou_span_b_period).max() + low_s.rolling(window=senkou_span_b_period).min()) / 2)
        cloud_strength = current_senkou_a - current_senkou_b # Strength of the current cloud projection

        # Cloud Breakout Signal (based on *current* price vs *current* cloud projection)
        cloud_breakout = np.zeros_like(close)
        for i in range(1, len(close)):
            # Use .iloc for safe access even with NaNs from rolling/shifting
            prev_close = close_s.iloc[i-1]
            curr_close = close_s.iloc[i]
            prev_a = current_senkou_a.iloc[i-1]
            prev_b = current_senkou_b.iloc[i-1]
            curr_a = current_senkou_a.iloc[i]
            curr_b = current_senkou_b.iloc[i]

            # Check if values are valid numbers before comparison
            if pd.notna(prev_close) and pd.notna(curr_close) and \
               pd.notna(prev_a) and pd.notna(prev_b) and pd.notna(curr_a) and pd.notna(curr_b):

                prev_cloud_top = max(prev_a, prev_b)
                prev_cloud_bottom = min(prev_a, prev_b)
                curr_cloud_top = max(curr_a, curr_b)
                curr_cloud_bottom = min(curr_a, curr_b)

                # Breakout above cloud
                if prev_close <= prev_cloud_top and curr_close > curr_cloud_top:
                    cloud_breakout[i] = 1
                # Breakdown below cloud
                elif prev_close >= prev_cloud_bottom and curr_close < curr_cloud_bottom:
                    cloud_breakout[i] = -1

        # Fill NaNs (important for features) - use ffill then bfill
        fill_value_price = close_s.iloc[0] # Use first close price as a fallback
        tenkan_sen = tenkan_sen.fillna(method='ffill').fillna(method='bfill').fillna(fill_value_price).values
        kijun_sen = kijun_sen.fillna(method='ffill').fillna(method='bfill').fillna(fill_value_price).values
        senkou_span_a = senkou_span_a.fillna(method='ffill').fillna(method='bfill').fillna(fill_value_price).values
        senkou_span_b = senkou_span_b.fillna(method='ffill').fillna(method='bfill').fillna(fill_value_price).values
        cloud_strength = cloud_strength.fillna(0).values # Fill strength NaNs with 0

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, cloud_strength, cloud_breakout

    @staticmethod
    def calculate_elder_ray(high, low, close, ema_period=13):
        """Calculate Elder Ray indicator using pandas"""
        high_s = pd.Series(np.nan_to_num(np.array(high, dtype=float)))
        low_s = pd.Series(np.nan_to_num(np.array(low, dtype=float)))
        close_s = pd.Series(np.nan_to_num(np.array(close, dtype=float)))
        min_len = ema_period
        if len(close) < min_len:
            print(f"Warning: Insufficient data for Elder Ray EMA (need {min_len}, got {len(close)}). Returning zeros.")
            zeros = np.zeros_like(close_s.values)
            return zeros, zeros, zeros

        # Calculate EMA of close price
        ema = close_s.ewm(span=ema_period, adjust=False).mean()

        # Calculate Bull Power and Bear Power
        bull_power = high_s - ema
        bear_power = low_s - ema

        # Calculate Elder Ray Signal (simplified version)
        # Buy signal: Bear Power rising (less negative) and above zero, Bull Power positive
        # Sell signal: Bull Power falling (less positive) and below zero, Bear Power negative
        elder_signal = np.zeros_like(close)
        bear_power_diff = bear_power.diff()
        bull_power_diff = bull_power.diff()

        # Iterate for signals (ensure index alignment and NaN check)
        for i in range(1, len(close)):
             # Check for valid numbers
             if pd.notna(bear_power_diff.iloc[i]) and pd.notna(bear_power.iloc[i]) and pd.notna(bull_power.iloc[i]):
                 # Buy Signal: Bear power rising and negative, Bull power positive
                 if bear_power_diff.iloc[i] > 0 and bear_power.iloc[i] < 0 and bull_power.iloc[i] > 0:
                     elder_signal[i] = 1
                 # Sell Signal: Bull power falling and positive, Bear power negative
                 elif pd.notna(bull_power_diff.iloc[i]) and bull_power_diff.iloc[i] < 0 and bull_power.iloc[i] > 0 and bear_power.iloc[i] < 0 :
                      elder_signal[i] = -1


        # Fill NaNs - Bull/Bear power can be NaN if EMA is NaN
        fill_value = 0.0
        bull_power = bull_power.fillna(fill_value).values
        bear_power = bear_power.fillna(fill_value).values

        return bull_power, bear_power, elder_signal

    @staticmethod
    def calculate_vwap(high, low, close, volume, period=14):
        """Calculate Volume Weighted Average Price (VWAP) using rolling window"""
        high = np.nan_to_num(np.array(high, dtype=float))
        low = np.nan_to_num(np.array(low, dtype=float))
        close = np.nan_to_num(np.array(close, dtype=float))
        volume = np.nan_to_num(np.array(volume, dtype=float))
        min_len = period
        if len(close) < min_len:
            print(f"Warning: Insufficient data for VWAP (need {min_len}, got {len(close)}). Returning zeros.")
            zeros = np.zeros_like(close)
            return zeros, zeros

        typical_price = (high + low + close) / 3
        pv = typical_price * volume

        # Rolling sum of PV and Volume
        rolling_pv_sum = pd.Series(pv).rolling(window=period).sum()
        rolling_volume_sum = pd.Series(volume).rolling(window=period).sum()

        # Calculate VWAP, handle division by zero
        epsilon = 1e-10
        vwap = rolling_pv_sum / (rolling_volume_sum + epsilon)

        # VWAP Cross Signal
        vwap_cross = np.zeros_like(close, dtype=int)
        close_s = pd.Series(close)
        vwap_s = vwap # Already a series

        # Ensure indices align and check for NaNs
        valid_indices = close_s.index.intersection(vwap_s.index)
        close_s = close_s[valid_indices]
        vwap_s = vwap_s[valid_indices]

        # Calculate crosses safely
        crossed_above = (close_s.shift(1) < vwap_s.shift(1)) & (close_s > vwap_s)
        crossed_below = (close_s.shift(1) > vwap_s.shift(1)) & (close_s < vwap_s)

        vwap_cross[crossed_above[crossed_above].index] = 1
        vwap_cross[crossed_below[crossed_below].index] = -1


        # Fill initial NaNs in VWAP
        vwap = vwap.fillna(method='ffill').fillna(method='bfill').fillna(close[0]).values

        return vwap, vwap_cross


    @staticmethod
    def calculate_roc(close, period=12):
        """Calculate Rate of Change (ROC) using pandas"""
        close_s = pd.Series(np.nan_to_num(np.array(close, dtype=float)))
        min_len = period + 1
        if len(close) < min_len:
            print(f"Warning: Insufficient data for ROC (need {min_len}, got {len(close)}). Returning zeros.")
            zeros = np.zeros_like(close_s.values)
            return zeros, zeros

        # Calculate ROC = ((Current Close - Close 'period' ago) / Close 'period' ago) * 100
        roc = ((close_s - close_s.shift(period)) / (close_s.shift(period) + 1e-10)) * 100 # Add epsilon

        # ROC Signal (crossing zero)
        roc_signal = np.zeros_like(close, dtype=int)
        crossed_above_zero = (roc.shift(1) < 0) & (roc > 0)
        crossed_below_zero = (roc.shift(1) > 0) & (roc < 0)

        roc_signal[crossed_above_zero[crossed_above_zero].index] = 1
        roc_signal[crossed_below_zero[crossed_below_zero].index] = -1


        # Fill initial NaNs
        roc = roc.fillna(0.0).values

        return roc, roc_signal

    # --- Methods below need careful review for suitability as predictive features ---
    # --- and handling lookahead bias or computational cost ---

    @staticmethod
    def calculate_fibonacci_retracement_levels(high, low, close, window=100):
        """Calculate dynamic Fibonacci retracement levels (potential lookahead bias)"""
        # WARNING: This function inherently looks within a window to find high/low,
        # which might introduce lookahead bias if not carefully implemented for training.
        # For prediction on latest data, it's less problematic if the window uses past data.
        high = np.nan_to_num(np.array(high, dtype=float))
        low = np.nan_to_num(np.array(low, dtype=float))
        close = np.nan_to_num(np.array(close, dtype=float))
        min_len = window + 2 # Need window + previous points for signal
        if len(close) < min_len:
            print(f"Warning: Insufficient data for Fibonacci (need {min_len}, got {len(close)}). Returning zero arrays.")
            return np.zeros((len(close), 4)), np.zeros_like(close) # Return 4 levels only


        fib_levels = np.zeros((len(close), 4)) # Store 23.6, 38.2, 50.0, 61.8 levels
        signals = np.zeros_like(close, dtype=int)
        fib_ratios = np.array([0.236, 0.382, 0.5, 0.618]) # Exclude 0 and 1 levels

        for i in range(window, len(close)):
            # Use data strictly *before* index i for calculations
            window_high = high[i-window:i]
            window_low = low[i-window:i]
            local_high = np.max(window_high)
            local_low = np.min(window_low)
            price_range = local_high - local_low

            if price_range < 1e-6: # Avoid division by zero if range is tiny
                 continue

            # Determine trend based on the window's start and end
            trend_up = close[i-1] > close[i-window] # Trend based on close price over the window

            for j, ratio in enumerate(fib_ratios):
                if trend_up: # Potential Uptrend -> Retracement levels are below high
                    fib_levels[i, j] = local_high - (price_range * ratio)
                else: # Potential Downtrend -> Retracement levels are above low
                    fib_levels[i, j] = local_low + (price_range * ratio)

            # --- Fibonacci Signal Logic (Example: Bounce off a level) ---
            # Check if current price is near a calculated level and reverses
            current_price = close[i]
            prev_price = close[i-1]
            prev_prev_price = close[i-2]
            tolerance = price_range * 0.02 # Tolerance zone around the level (e.g., 2%)

            for j in range(len(fib_ratios)):
                level = fib_levels[i, j]
                # Bounce Up Signal: Was falling, hit near level, now rising
                if prev_price < prev_prev_price and abs(prev_price - level) < tolerance and current_price > prev_price:
                     signals[i] = 1
                     break # Take first signal
                # Bounce Down Signal: Was rising, hit near level, now falling
                elif prev_price > prev_prev_price and abs(prev_price - level) < tolerance and current_price < prev_price:
                     signals[i] = -1
                     break # Take first signal

        # Fill initial NaNs/zeros appropriately
        # For levels, forward fill might make sense
        fib_levels_df = pd.DataFrame(fib_levels).fillna(method='ffill').fillna(method='bfill')
        # Ensure levels don't go below 0? (if necessary)

        return fib_levels_df.values, signals


    @staticmethod
    def volatility_breakout(close, high, low, period=20, multiplier=2.5):
        """Calculate volatility breakout using ATR (similar to Keltner Channels)"""
        high = np.nan_to_num(np.array(high, dtype=float))
        low = np.nan_to_num(np.array(low, dtype=float))
        close = np.nan_to_num(np.array(close, dtype=float))
        min_len = period
        if len(close) < min_len:
            print(f"Warning: Insufficient data for Volatility Breakout (need {min_len}, got {len(close)}). Returning zeros.")
            zeros = np.zeros_like(close)
            return zeros, zeros, zeros

        # Calculate ATR (using pandas as in Supertrend)
        tr1 = pd.Series(high - low)
        tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
        tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean().fillna(method='bfill').fillna(0)

        # Calculate Bands based on *previous* close + ATR (common breakout strategy)
        # Or use EMA/SMA of close as center line? Let's use previous close for simplicity.
        # Using current close might introduce lookahead issues if not lagged.
        center_line = pd.Series(close).shift(1) # Base breakout on previous close
        center_line.iloc[0] = close[0] # Handle first element NaN

        upper_band = center_line + (atr * multiplier)
        lower_band = center_line - (atr * multiplier)

        # Breakout Signals
        signals = np.zeros_like(close, dtype=int)
        close_s = pd.Series(close)

        # Use .iloc for safe access
        for i in range(1, len(close)):
             curr_close = close_s.iloc[i]
             prev_upper = upper_band.iloc[i-1] # Compare current close to *previous* band
             prev_lower = lower_band.iloc[i-1]

             if pd.notna(curr_close) and pd.notna(prev_upper) and pd.notna(prev_lower):
                 # Breakout above previous upper band
                 if curr_close > prev_upper:
                     signals[i] = 1
                 # Breakdown below previous lower band
                 elif curr_close < prev_lower:
                     signals[i] = -1


        # Fill NaNs in bands for feature completeness
        fill_val = close[0]
        upper_band = upper_band.fillna(method='ffill').fillna(method='bfill').fillna(fill_val).values
        lower_band = lower_band.fillna(method='ffill').fillna(method='bfill').fillna(fill_val).values

        return upper_band, lower_band, signals

    @staticmethod
    def calculate_price_patterns(open_price, high, low, close, volume, threshold=0.01):
        """Detect common candlestick patterns (more examples)"""
        open_price = np.nan_to_num(np.array(open_price, dtype=float))
        high = np.nan_to_num(np.array(high, dtype=float))
        low = np.nan_to_num(np.array(low, dtype=float))
        close = np.nan_to_num(np.array(close, dtype=float))
        # volume might be needed for some patterns, ensure it's passed and handled
        volume = np.nan_to_num(np.array(volume, dtype=float))

        if not all(len(arr) == len(close) for arr in [open_price, high, low, volume]):
             raise ValueError("OHLCV must have the same length for price patterns")
        if len(close) < 3: # Need at least 3 bars for some patterns
            return np.zeros_like(close, dtype=int)

        patterns = np.zeros_like(close, dtype=int)
        epsilon = 1e-6

        for i in range(2, len(close)): # Start from index 2 to check previous bars
            # Current bar data
            o, h, l, c = open_price[i], high[i], low[i], close[i]
            # Previous bar data
            o1, h1, l1, c1 = open_price[i-1], high[i-1], low[i-1], close[i-1]
            # Bar before previous
            o2, h2, l2, c2 = open_price[i-2], high[i-2], low[i-2], close[i-2]

            body = abs(c - o)
            body1 = abs(c1 - o1)
            body2 = abs(c2 - o2)
            range_ = h - l + epsilon
            range1 = h1 - l1 + epsilon
            range2 = h2 - l2 + epsilon

            # --- Bullish Engulfing ---
            # Prev bar red, current bar green
            # Current body engulfs previous body
            if c1 < o1 and c > o and c > o1 and o < c1:
                patterns[i] = 1 # Bullish

            # --- Bearish Engulfing ---
            # Prev bar green, current bar red
            # Current body engulfs previous body
            elif c1 > o1 and c < o and c < o1 and o > c1:
                patterns[i] = -1 # Bearish

            # --- Hammer (Bullish Reversal) ---
            # Small body near top, long lower shadow, minimal upper shadow
            # Occurs after a downtrend (check previous bar close < open)
            lower_shadow = min(o, c) - l
            upper_shadow = h - max(o, c)
            if c2 < o2 and lower_shadow > 2 * body and upper_shadow < body * 0.5 and body / range_ < 0.3:
                 patterns[i] = 1 # Bullish Hammer

            # --- Shooting Star (Bearish Reversal) ---
            # Small body near bottom, long upper shadow, minimal lower shadow
            # Occurs after an uptrend (check previous bar close > open)
            elif c2 > o2 and upper_shadow > 2 * body and lower_shadow < body * 0.5 and body / range_ < 0.3:
                 patterns[i] = -1 # Bearish Shooting Star

            # --- Morning Star (Bullish Reversal - 3 bars) ---
            # 1. Long red bar
            # 2. Small body (star) gapping down
            # 3. Long green bar closing well into the first red bar's body
            if c2 < o2 and body2 / range2 > 0.6 and \
               max(o1, c1) < l2 and body1 / range1 < 0.3 and \
               c > o and c > (o2 + c2) / 2 and o > max(o1,c1):
                 patterns[i] = 1 # Bullish Morning Star

            # --- Evening Star (Bearish Reversal - 3 bars) ---
            # 1. Long green bar
            # 2. Small body (star) gapping up
            # 3. Long red bar closing well into the first green bar's body
            elif c2 > o2 and body2 / range2 > 0.6 and \
                 min(o1, c1) > h2 and body1 / range1 < 0.3 and \
                 c < o and c < (o2 + c2) / 2 and o < min(o1,c1):
                 patterns[i] = -1 # Bearish Evening Star

            # Add more patterns here... (Piercing Line, Dark Cloud Cover, etc.)

        return patterns


    @staticmethod
    def calculate_gap_analysis(open_price, high, low, close):
        """Identify price gaps and whether they were filled"""
        open_price = np.nan_to_num(np.array(open_price, dtype=float))
        high = np.nan_to_num(np.array(high, dtype=float))
        low = np.nan_to_num(np.array(low, dtype=float))
        close = np.nan_to_num(np.array(close, dtype=float)) # Close not directly used but good for length check

        if not all(len(arr) == len(close) for arr in [open_price, high, low]):
             raise ValueError("OHLC must have the same length for gap analysis")
        if len(close) < 2: # Need previous bar
             return np.zeros_like(close, dtype=int), np.zeros_like(close, dtype=int)


        gaps = np.zeros_like(close, dtype=int) # 1 for gap up, -1 for gap down
        gap_filled = np.zeros_like(close, dtype=int) # 1 if the gap that *started* at this index was filled later

        for i in range(1, len(close)):
            # Gap Up: Current open is higher than previous high
            if open_price[i] > high[i-1]:
                gaps[i] = 1
            # Gap Down: Current open is lower than previous low
            elif open_price[i] < low[i-1]:
                gaps[i] = -1

        # Check for gap filling (within next few bars, e.g., 5)
        fill_window = 5
        for i in range(1, len(close)):
            if gaps[i] == 1: # Gap Up - needs price to fall back below previous high
                lookahead_end = min(i + fill_window + 1, len(close))
                if np.any(low[i+1:lookahead_end] <= high[i-1]):
                    gap_filled[i] = 1
            elif gaps[i] == -1: # Gap Down - needs price to rise back above previous low
                lookahead_end = min(i + fill_window + 1, len(close))
                if np.any(high[i+1:lookahead_end] >= low[i-1]):
                    gap_filled[i] = 1
            # Note: 'gap_filled' is a feature of the bar where the gap *occurred*,
            # based on *future* price action. Use with caution as a direct predictive feature.
            # It might be better to use 'gaps' itself, or features derived from it
            # like "gap size" or "time since last unfilled gap".

        return gaps, gap_filled # Return both flags

# --- Define the list of Nifty 50 Stocks (Keep as is) ---
NIFTY_50 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
    'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS', 'ITC.NS',
    'AXISBANK.NS', 'MARUTI.NS', 'LT.NS', 'ASIANPAINT.NS', 'HCLTECH.NS',
    'TITAN.NS', 'SUNPHARMA.NS', 'NESTLEIND.NS', 'ULTRACEMCO.NS', 'POWERGRID.NS',
    'TECHM.NS', 'WIPRO.NS', 'INDUSINDBK.NS', 'DIVISLAB.NS', 'JSWSTEEL.NS',
    'BHARTIARTL.NS', 'M&M.NS', 'BAJAJFINSV.NS', 'HDFCLIFE.NS', 'TATAMOTORS.NS',
    'GRASIM.NS', 'ADANIPORTS.NS', 'CIPLA.NS', 'DRREDDY.NS', 'BRITANNIA.NS',
    'EICHERMOT.NS', 'SBILIFE.NS', 'HEROMOTOCO.NS', 'ONGC.NS', 'COALINDIA.NS',
    'UPL.NS', 'BPCL.NS', 'SHREECEM.NS', 'NTPC.NS', 'TATASTEEL.NS',
    'HINDALCO.NS', 'IOC.NS', 'ADANIENT.NS', 'GAIL.NS', 'VEDL.NS'
]

# --- Fetch Data (Keep as is, maybe add interval parameter) ---
def fetch_data(symbols, start_date, end_date, interval='1d'): # Added interval
    data = {}
    print(f"Fetching data with interval: {interval}")
    for symbol in symbols:
        try:
            # Use interval parameter here
            df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)

            if df.empty:
                 print(f"Skipping {symbol}: No data returned for the period/interval.")
                 continue

            # --- Robust handling for yfinance returning slightly different formats ---
            # 1. Reset index BEFORE checking columns if DatetimeIndex is not standard
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()

             # 2. Handle potential MultiIndex columns (common with adjusted prices etc.)
            if isinstance(df.columns, pd.MultiIndex):
                 print(f"{symbol} has MultiIndex columns, flattening...")
                 # Simple flatten: take the first level name, ensure case-insensitivity
                 df.columns = [str(col[0]).title() if isinstance(col, tuple) else str(col).title() for col in df.columns]


            # 3. Check for required columns (case-insensitive)
            required_cols_case = ['Open', 'High', 'Low', 'Close', 'Volume']
            cols_map = {col.lower(): col for col in df.columns}
            found_cols = [cols_map.get(req.lower()) for req in required_cols_case]

            if not all(found_cols):
                 missing = [req for req, found in zip(required_cols_case, found_cols) if not found]
                 print(f"Skipping {symbol}: Missing required columns (case-insensitive): {missing}. Available: {list(df.columns)}")
                 continue

            # Rename columns to the standard required case
            df = df.rename(columns={found: req for req, found in zip(required_cols_case, found_cols) if found})


            # 4. Select only required columns + Handle Index
            # If 'Date' or 'Datetime' column exists after reset, use it as index
            date_col = None
            if 'Date' in df.columns: date_col = 'Date'
            elif 'Datetime' in df.columns: date_col = 'Datetime'

            if date_col:
                df['Date'] = pd.to_datetime(df[date_col])
                df = df.set_index('Date')
                # Keep only required cols if index is set
                df = df[required_cols_case]

            elif isinstance(df.index, pd.DatetimeIndex):
                 # If index is already DatetimeIndex, just select columns
                 df = df[required_cols_case]
            else:
                print(f"Skipping {symbol}: Could not identify a Datetime index or column.")
                continue


            # 5. Final checks and cleaning
            if df[required_cols_case].isnull().values.any(): # Check underlying numpy array for speed
                 print(f"Warning: {symbol} contains NaN values. Applying ffill/bfill.")
                 df = df.ffill().bfill() # Fill NaNs

            # Ensure Volume is numeric, replace non-numeric with 0
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)

            # Drop rows where essential prices might still be zero/NaN after filling (unlikely but safe)
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            df = df[(df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)] # Ensure prices are positive

            if df.empty or len(df) < 50: # Check length *after* cleaning
                 print(f"Skipping {symbol}: Insufficient valid data after cleaning ({len(df)} rows).")
                 continue


            print(f"Fetched and cleaned {symbol}: {len(df)} rows.")
            data[symbol] = df

        except Exception as e:
            print(f"!!!!! Error fetching/processing {symbol}: {e} !!!!!")
            import traceback
            traceback.print_exc()

    return data


# --- Generate Labels (Keep as is for training) ---
def generate_labels(df, threshold=0.005): # Added threshold flexibility
    """
    Generate labels for stock price movement based on next day's percentage change.
    Returns:
        numpy array of labels (-1: down, 0: neutral, 1: up) for df[0] to df[-2]
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column for label generation.")
    if len(df) < 2:
         return np.array([]) # Not enough data to calculate change

    # Calculate percentage change between current close and *next* close
    returns = df['Close'].pct_change().shift(-1)

    labels = np.zeros(len(df)) # Initialize labels for all rows
    labels[returns > threshold] = 1  # Up
    labels[returns < -threshold] = -1 # Down
    # Rows where change is within [-threshold, threshold] remain 0 (Neutral)

    # Return labels corresponding to df[0] up to df[-2]
    # The label for df[i] depends on the close price of df[i+1]
    # The last row's label cannot be determined this way and will be NaN in 'returns', resulting in 0 here.
    return labels[:-1]


# --- Computing ALL Indicators (Keep as is, ensure it returns computed_cols) ---
def compute_all_indicators(df):
    """Computes all defined technical indicators for the given DataFrame."""
    indicators_df = pd.DataFrame(index=df.index)
    computed_cols = [] # Keep track of added columns

    # Ensure input df has the required columns and is not empty
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input DataFrame missing required columns for indicators: {required_cols}")
        return pd.DataFrame(index=df.index), []
    if df.empty:
        print("Error: Input DataFrame is empty for indicators.")
        return pd.DataFrame(index=df.index), []


    try:
        # Ensure data types are correct and handle potential NaNs before passing to functions
        high = pd.to_numeric(df['High'], errors='coerce').values
        low = pd.to_numeric(df['Low'], errors='coerce').values
        close = pd.to_numeric(df['Close'], errors='coerce').values
        open_price = pd.to_numeric(df['Open'], errors='coerce').values
        volume = pd.to_numeric(df['Volume'], errors='coerce').values

        # Apply ffill/bfill *after* conversion if necessary (should be done in fetch_data ideally)
        high = pd.Series(high).ffill().bfill().values
        low = pd.Series(low).ffill().bfill().values
        close = pd.Series(close).ffill().bfill().values
        open_price = pd.Series(open_price).ffill().bfill().values
        volume = pd.Series(volume).ffill().bfill().fillna(0).values # Fill volume NaNs with 0


        # --- Calculate Indicators ---
        # Pass numpy arrays to calculation functions
        print("Calculating ADX...")
        indicators_df['adx'] = CustomTechnicalIndicators.calculate_adx(high, low, close)
        computed_cols.append('adx')

        print("Calculating RSI...")
        indicators_df['rsi'] = CustomTechnicalIndicators.calculate_rsi(close)
        computed_cols.append('rsi')

        print("Calculating OBV...")
        indicators_df['obv'] = CustomTechnicalIndicators.calculate_obv(close, volume)
        computed_cols.append('obv')

        print("Calculating MACD...")
        macd_line, signal_line = CustomTechnicalIndicators.calculate_macd(close)
        indicators_df['macd'] = macd_line
        indicators_df['macd_signal'] = signal_line
        computed_cols.extend(['macd', 'macd_signal'])

        print("Calculating Stochastic...")
        k_line, d_line = CustomTechnicalIndicators.calculate_stochastic(high, low, close)
        indicators_df['stochastic_k'] = k_line
        indicators_df['stochastic_d'] = d_line
        computed_cols.extend(['stochastic_k', 'stochastic_d'])

        print("Calculating Bollinger Bands...")
        upper_b, lower_b = CustomTechnicalIndicators.calculate_bollinger_bands(close)
        indicators_df['bollinger_upper'] = upper_b
        indicators_df['bollinger_lower'] = lower_b
        # Optional: Bollinger Band Width or %B
        sma20 = pd.Series(close).rolling(window=20).mean().replace(0, 1e-6) # Avoid division by zero
        indicators_df['bollinger_width'] = (upper_b - lower_b) / sma20
        computed_cols.extend(['bollinger_upper', 'bollinger_lower', 'bollinger_width'])

        print("Calculating Support/Resistance...")
        support, resistance = CustomTechnicalIndicators.detect_support_resistance(high, low)
        indicators_df['support'] = support
        indicators_df['resistance'] = resistance
        computed_cols.extend(['support', 'resistance'])

        print("Calculating Candle Patterns (Doji)...")
        indicators_df['candle_doji'] = CustomTechnicalIndicators.detect_candle_patterns(open_price, high, low, close)
        computed_cols.append('candle_doji')

        print("Calculating Supertrend...")
        supertrend_val, uptrend_flag = CustomTechnicalIndicators.calculate_supertrend(high, low, close)
        indicators_df['supertrend'] = supertrend_val
        indicators_df['supertrend_uptrend'] = uptrend_flag
        computed_cols.extend(['supertrend', 'supertrend_uptrend'])

        print("Calculating Ichimoku...")
        tenkan, kijun, span_a, span_b, cloud_str, cloud_break = CustomTechnicalIndicators.calculate_ichimoku(high, low, close)
        indicators_df['ichimoku_tenkan'] = tenkan
        indicators_df['ichimoku_kijun'] = kijun
        indicators_df['ichimoku_span_a'] = span_a # Note: Shifted forward
        indicators_df['ichimoku_span_b'] = span_b # Note: Shifted forward
        indicators_df['ichimoku_cloud_strength'] = cloud_str # Based on current projection
        indicators_df['ichimoku_breakout'] = cloud_break # Based on current projection
        computed_cols.extend(['ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_span_a',
                              'ichimoku_span_b', 'ichimoku_cloud_strength', 'ichimoku_breakout'])

        print("Calculating Elder Ray...")
        bull_p, bear_p, elder_sig = CustomTechnicalIndicators.calculate_elder_ray(high, low, close)
        indicators_df['elder_bull'] = bull_p
        indicators_df['elder_bear'] = bear_p
        indicators_df['elder_signal'] = elder_sig
        computed_cols.extend(['elder_bull', 'elder_bear', 'elder_signal'])

        print("Calculating VWAP...")
        vwap_val, vwap_cross_sig = CustomTechnicalIndicators.calculate_vwap(high, low, close, volume)
        indicators_df['vwap'] = vwap_val
        indicators_df['vwap_cross'] = vwap_cross_sig
        computed_cols.extend(['vwap', 'vwap_cross'])

        print("Calculating ROC...")
        roc_val, roc_sig = CustomTechnicalIndicators.calculate_roc(close)
        indicators_df['roc'] = roc_val
        indicators_df['roc_signal'] = roc_sig
        computed_cols.extend(['roc', 'roc_signal'])

        print("Calculating Fibonacci...")
        fib_levels_arr, fib_signals = CustomTechnicalIndicators.calculate_fibonacci_retracement_levels(high, low, close)
        # Assuming returns levels [23.6, 38.2, 50.0, 61.8] relative to index i
        fib_cols = ['fib_236', 'fib_382', 'fib_500', 'fib_618']
        if fib_levels_arr.shape[1] >= len(fib_cols): # Check if enough columns returned
             for idx, col_name in enumerate(fib_cols):
                 indicators_df[col_name] = fib_levels_arr[:, idx]
                 computed_cols.append(col_name)
        else:
             print(f"Warning: Fibonacci levels returned {fib_levels_arr.shape[1]} columns, expected {len(fib_cols)}. Skipping.")
        indicators_df['fib_signal'] = fib_signals
        computed_cols.append('fib_signal')

        print("Calculating Volatility Breakout...")
        vol_upper, vol_lower, vol_sig = CustomTechnicalIndicators.volatility_breakout(close, high, low)
        indicators_df['volatility_upper'] = vol_upper
        indicators_df['volatility_lower'] = vol_lower
        indicators_df['volatility_signal'] = vol_sig
        computed_cols.extend(['volatility_upper', 'volatility_lower', 'volatility_signal'])

        print("Calculating Price Patterns...")
        indicators_df['price_patterns'] = CustomTechnicalIndicators.calculate_price_patterns(open_price, high, low, close, volume)
        computed_cols.append('price_patterns')

        print("Calculating Gap Analysis...")
        gap_flag, gap_fill_flag = CustomTechnicalIndicators.calculate_gap_analysis(open_price, high, low, close)
        indicators_df['gap'] = gap_flag
        indicators_df['gap_filled'] = gap_fill_flag # Use with caution (lookahead)
        computed_cols.extend(['gap', 'gap_filled'])

        # --- Post Processing ---
        # Replace infinite values with NaN, then handle NaNs
        indicators_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Count NaNs before dropping/filling
        nan_counts_before = indicators_df.isnull().sum()
        initial_len = len(indicators_df)

        # Option 1: Drop rows with any NaNs (simplest, used in original training)
        # indicators_df = indicators_df.dropna()

        # Option 2: Fill NaNs (might be better for prediction where dropping isn't feasible)
        # Fill with 0 for signals/flags, ffill/bfill for continuous values
        for col in indicators_df.columns:
             if 'signal' in col or 'flag' in col or 'pattern' in col or 'gap' in col or 'support' in col or 'resistance' in col or 'uptrend' in col or 'doji' in col or 'breakout' in col or 'cross' in col:
                 indicators_df[col] = indicators_df[col].fillna(0)
             else:
                 # For continuous indicators, ffill then bfill is generally safe
                 indicators_df[col] = indicators_df[col].fillna(method='ffill').fillna(method='bfill')

        # Final check for any remaining NaNs (e.g., if all data was NaN initially)
        indicators_df = indicators_df.fillna(0) # Fill any truly persistent NaNs with 0

        final_len = len(indicators_df)
        # print(f"NaNs handled in indicators. Original length: {initial_len}, Final length: {final_len}.")
        # nan_counts_after = indicators_df.isnull().sum().sum()
        # print(f"Total remaining NaNs after filling: {nan_counts_after}")


        # Ensure all computed columns exist, even if calculation failed and returned zeros/NaNs
        final_cols = indicators_df.columns.tolist()
        for col in computed_cols:
            if col not in final_cols:
                 print(f"Warning: Column {col} was expected but is missing after indicator calculation. Adding as zeros.")
                 indicators_df[col] = 0.0 # Add as zero column

        # Reorder columns to match computed_cols order + any extras
        indicators_df = indicators_df[computed_cols]

        print(f"Indicator calculation complete. Shape: {indicators_df.shape}")
        return indicators_df, computed_cols

    except Exception as e:
        print(f"!!!!! Error computing indicators: {e} !!!!!")
        import traceback
        traceback.print_exc()
        # Return an empty DataFrame but preserve index if possible
        return pd.DataFrame(index=df.index), []


# --- Preparing Data for LSTM/RNN (Keep as is for training) ---
def prepare_data(df, all_feature_cols, lookback=60):
    """Prepares data for sequence modeling (training), including scaling and PCA."""
    X, y = [], []

    # 1. Generate Labels (aligned with df before dropping last row)
    print("Generating labels for training...")
    labels = generate_labels(df) # Gets labels for df[0] to df[-2]
    if len(labels) == 0:
        print("Warning: No labels generated, possibly insufficient data.")
        # Return empty arrays and None for scaler/pca
        return np.array([]), np.array([]), None, None


    # 2. Align features and labels: Use data up to the second to last row for features
    #    to predict the label corresponding to that row (which depends on the *next* day's close)
    df_features = df.iloc[:-1].copy() # Use data from index 0 to -2
    # Ensure labels align exactly with the selected features
    if len(labels) != len(df_features):
         print(f"Warning: Label length ({len(labels)}) mismatch with feature length ({len(df_features)}). Truncating features.")
         # This shouldn't happen with current generate_labels logic, but as a safeguard:
         min_len = min(len(labels), len(df_features))
         df_features = df_features.iloc[:min_len]
         aligned_labels = labels[:min_len]
    else:
         aligned_labels = labels # Labels are already aligned with df_features


    # 3. Check for missing columns
    missing_cols = [col for col in all_feature_cols if col not in df_features.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame for prepare_data: {missing_cols}. Available: {df_features.columns.tolist()}")

    # Select and scale features
    print(f"Selecting features: {all_feature_cols}")
    data = df_features[all_feature_cols].values

    # Handle potential infinities before scaling
    data[np.isinf(data)] = np.nan
    # Impute remaining NaNs if any slipped through (e.g., with column mean)
    if np.isnan(data).any():
        print("Warning: NaNs detected before scaling. Imputing with mean.")
        col_mean = np.nanmean(data, axis=0)
        # Find indices where nan is present
        inds = np.where(np.isnan(data))
        # Place column means in the indices. Align the arrays using take
        data[inds] = np.take(col_mean, inds[1])
        # Check if any NaNs remain (e.g., if a whole column was NaN)
        if np.isnan(data).any():
             print("Warning: NaNs still present after mean imputation. Filling with 0.")
             data = np.nan_to_num(data, nan=0.0)


    print("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=0.95) # Keep 95% variance
    # Handle potential issue if data_scaled has zero variance columns
    try:
        data_pca = pca.fit_transform(data_scaled)
    except ValueError as ve:
        print(f"PCA Error: {ve}. Trying without PCA.")
        # Fallback: Don't use PCA if it fails
        pca = None # Set pca object to None
        data_pca = data_scaled # Use scaled data directly
        # Alternatively, remove zero-variance columns before PCA:
        # var_thresh = VarianceThreshold(threshold=0.0)
        # data_scaled_var = var_thresh.fit_transform(data_scaled)
        # data_pca = pca.fit_transform(data_scaled_var) if data_scaled_var.shape[1] > 0 else data_scaled


    print(f"PCA applied: Reduced features from {data_scaled.shape[1]} to {data_pca.shape[1]}")

    # Create sequences
    print(f"Creating sequences with lookback={lookback}...")
    # We need 'lookback' historical points to predict the label at index 'i'
    # Labels (aligned_labels) correspond to data_pca indices 0 to len(data_pca)-1
    # Max index for 'i' should ensure data_pca[i-lookback:i] is valid
    # So, i ranges from lookback to len(data_pca)
    for i in range(lookback, len(data_pca)):
        X.append(data_pca[i-lookback:i])
        # The label for the sequence ending at index i-1 is aligned_labels[i-1]
        # Let's re-verify the alignment.
        # Sequence X[k] = data_pca[k: k+lookback] where k = i-lookback
        # This sequence uses data up to index k+lookback-1 = i-1.
        # The label corresponding to the state *at* index i-1 (predicting the move from i-1 to i)
        # is aligned_labels[i-1].
        y.append(aligned_labels[i-1]) # Label at i-1 corresponds to sequence ending at i-1

    if not X:
        print("Warning: No sequences created. Check lookback period and data length.")
        return np.array([]), np.array([]), scaler, pca


    X = np.array(X)
    y = np.array(y)

    # Convert labels: -1 -> 0, 0 -> 1, 1 -> 2 for categorical crossentropy
    y_mapped = y + 1
    # Convert to one-hot encoding
    y_cat = to_categorical(y_mapped, num_classes=3)

    print(f"Data preparation complete. X shape: {X.shape}, y shape: {y_cat.shape}")
    return X, y_cat, scaler, pca


def get_latest_predictions(ticker, model_path, sequence_length=20, forward_window=10, threshold=0.02):
    """
    Fetches latest **DAILY** data, loads a trained model, approximates data preparation,
    makes a prediction for the *very latest* time step (day), and returns a dictionary
    containing only the timestamp, OHLCV, and predicted signal for that step.

    Args:
        ticker (str): The stock symbol (e.g., 'RELIANCE.NS').
        model_path (str): The full path to the saved Keras model file (.keras or .h5).
        sequence_length (int): The lookback window size the model expects (in days).
        forward_window (int): Not used for prediction, kept for signature consistency.
        threshold (float): Not used for prediction, kept for signature consistency.

    Returns:
        dict: A dictionary containing the Date, OHLCV, and Signal for the latest
              prediction, or an error dictionary.

    **WARNING:**
    1. Data preparation (scaling/PCA) is approximated based on recent data only.
    2. This function now fetches DAILY data ('1d' interval). The loaded model
       **MUST** have been trained on DAILY data with the same features for the
       prediction to be meaningful.
    """
    print(f"\n===== Generating Latest DAILY Prediction for {ticker} (using model: {model_path}) =====")
    print(f"*** WARNING: Using DAILY ('1d') data interval. Model MUST be trained on daily data. ***")
    print(f"*** WARNING: Data preparation (scaling/PCA) is approximated based on recent data only. ***")

    # --- 1. Load Model and Infer Input Shape ---
    if not os.path.exists(model_path):
        return {"error": f"Model file not found at {model_path}"}
    try:
        model = load_model(model_path)
        try:
            model_input_shape = model.input_shape
            if not isinstance(model_input_shape, tuple) or len(model_input_shape) != 3:
                 model_input_shape = model.layers[0].input_shape
                 if isinstance(model_input_shape, list): model_input_shape = model_input_shape[0]
            if not isinstance(model_input_shape, tuple) or len(model_input_shape) != 3: raise ValueError(f"Invalid shape: {model_input_shape}")
            expected_seq_len = model_input_shape[1]
            expected_features = model_input_shape[2]
            if expected_seq_len is None or expected_features is None: raise ValueError(f"Shape dimensions None: ({expected_seq_len}, {expected_features})")
            if expected_seq_len != sequence_length:
                print(f"Info: Overriding sequence_length ({sequence_length}) with model's expected length ({expected_seq_len}).")
                sequence_length = expected_seq_len
        except Exception as e: return {"error": f"Could not determine input shape from model: {e}"}
    except Exception as e: return {"error": f"Failed to load model: {e}"}

    # --- 2. Fetch Recent Daily Data ---
    interval = '1d' # <--- Changed interval to daily
    # Need enough days for sequence + indicator calculation buffer
    indicator_buffer_days = 50 # Buffer for daily indicators (e.g., SMA50, RSI14 needs >14 days)
    required_days = sequence_length + indicator_buffer_days
    buffer_fetch_days = 15 # Extra buffer for non-trading days, holidays etc.
    days_to_fetch = required_days + buffer_fetch_days

    # Use current date as end date, go back calculated number of days
    # Add 1 day to end_date to ensure today's data might be included if available after market close
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=days_to_fetch)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"Fetching approx {days_to_fetch} days of '{interval}' data (ends {end_date_str})...")
    try:
        stock_data_map = fetch_data([ticker], start_date_str, end_date_str, interval=interval)
        if not stock_data_map or ticker not in stock_data_map:
            return {"error": f"No recent daily data found for {ticker} between {start_date_str} and {end_date_str}."}
        stock_data_raw = stock_data_map[ticker]
        print(f"Raw daily data loaded: {stock_data_raw.shape}")
        if len(stock_data_raw) < sequence_length:
            return {"error": f"Insufficient daily data ({len(stock_data_raw)}) for sequence ({sequence_length}). Need at least {sequence_length} days."}
        # Ensure data is sorted chronologically
        stock_data_raw.sort_index(inplace=True)
    except Exception as e:
        return {"error": f"Failed to fetch daily data: {e}"}

    # --- 3. Calculate Indicators ---
    print("Calculating indicators on daily data...")
    try:
         # Ensure input dataframe has the standard OHLCV columns
         required_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
         if not all(col in stock_data_raw.columns for col in required_ohlcv):
             return {"error": f"Fetched data missing required columns: {required_ohlcv}. Found: {stock_data_raw.columns.tolist()}"}

         indicators_df, indicator_cols = compute_all_indicators(stock_data_raw)
         if indicators_df.empty and indicator_cols: # Check if computation failed but returned columns
              return {"error": f"Indicator computation failed, returned empty DataFrame."}
         # Allow for cases where *no* indicators are computed (indicator_cols is empty)
         if not indicators_df.empty:
             df_full = stock_data_raw.join(indicators_df, how='left')
         else:
             df_full = stock_data_raw.copy() # No indicators to join

         # IMPORTANT: Fill NaNs *after* join. Fill forward first, then backward for initial rows.
         df_full.ffill(inplace=True)
         df_full.bfill(inplace=True)
         # Final fallback fill with 0, although bfill should handle most initial NaNs
         df_full.fillna(0, inplace=True)
         print(f"Indicators done. Full data shape: {df_full.shape}")

         # Ensure indicator_cols only contains columns actually present AFTER the join
         indicator_cols = [col for col in indicator_cols if col in df_full.columns]

    except Exception as e:
        import traceback
        print("--- ERROR TRACEBACK (Indicator Calculation) ---")
        traceback.print_exc()
        print("--- END TRACEBACK ---")
        return {"error": f"Error computing indicators: {e}"}


    # --- 4. Approximate Data Preparation ---
    print("Approximating data prep (scaling/PCA) on recent daily data...")
    try:
        base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Use only columns that actually exist in df_full
        all_available_features = base_features + [col for col in indicator_cols if col in df_full.columns]
        if not all(f in df_full.columns for f in all_available_features):
            missing_f = [f for f in all_available_features if f not in df_full.columns]
            return {"error": f"Internal error: Features missing from df_full after processing: {missing_f}"}

        feature_data_full = df_full[all_available_features].values

        # Handle potential inf/-inf values that might arise from calculations
        feature_data_full[np.isinf(feature_data_full)] = 0.0 # Replace inf with 0
        feature_data_full = np.nan_to_num(feature_data_full, nan=0.0) # Replace NaN with 0

        # Fit scaler on a recent subset of the data to approximate training conditions
        temp_scaler = StandardScaler()
        # Ensure we have enough data points to fit; use at least required_days or all available if less
        fit_data_len = min(len(feature_data_full), required_days)
        fit_data = feature_data_full[-fit_data_len:]
        if fit_data.shape[0] == 0:
            return {"error": "Not enough data points to fit scaler."}
        temp_scaler.fit(fit_data)
        scaled_data_full = temp_scaler.transform(feature_data_full)

        current_num_features = scaled_data_full.shape[1]
        pca_data_full = scaled_data_full # Default if no PCA needed or features match

        if current_num_features > expected_features:
            print(f"Applying temporary PCA ({current_num_features} -> {expected_features}) fitted on recent data...")
            temp_pca = PCA(n_components=expected_features)
            # Fit PCA on the same scaled data subset used for the scaler
            scaled_fit_data = temp_scaler.transform(fit_data)
            temp_pca.fit(scaled_fit_data)
            pca_data_full = temp_pca.transform(scaled_data_full)
            if pca_data_full.shape[1] != expected_features:
                 return {"error": f"PCA shape mismatch after transform. Expected {expected_features}, got {pca_data_full.shape[1]}."}
        elif current_num_features < expected_features:
             # This indicates a mismatch between data prep and model expectation
             return {"error": f"Feature count after prep ({current_num_features}) is LESS than model expects ({expected_features}). Check indicator list and data processing."}
        # Else (current_num_features == expected_features): No PCA needed, use scaled_data_full

        # Check if we have enough processed data points for the final sequence
        if len(pca_data_full) < sequence_length:
             return {"error": f"Not enough processed data points ({len(pca_data_full)}) to form a sequence of length {sequence_length}."}

        # Get the very last sequence
        latest_sequence_processed = pca_data_full[-sequence_length:]

        # Final shape check before prediction
        if latest_sequence_processed.shape != (sequence_length, expected_features):
             return {"error": f"Final sequence shape mismatch. Expected ({sequence_length}, {expected_features}), got {latest_sequence_processed.shape}."}

        X_latest = latest_sequence_processed.reshape((1, sequence_length, expected_features))
        # print(f"Prep done. Input shape for model: {X_latest.shape}")

    except Exception as e:
        import traceback
        print("--- ERROR TRACEBACK (Data Preparation) ---")
        traceback.print_exc()
        print("--- END TRACEBACK ---")
        return {"error": f"Error during data preparation: {type(e).__name__} - {e}"}

    # --- 5. Make Prediction ---
    print("Making prediction on the latest daily sequence...")
    try:
        # Ensure data type is float32 for TensorFlow/Keras
        X_latest = X_latest.astype(np.float32)

        prediction_probs = model.predict(X_latest)
        predicted_class = np.argmax(prediction_probs, axis=1)[0]

        # Standard signal mapping (adjust if your model uses a different scheme)
        signal_mapping = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
        predicted_signal = signal_mapping.get(predicted_class, f'Unknown Class ({predicted_class})')

        # --- 6. Get Data for the Corresponding Timestamp ---
        if not isinstance(df_full, pd.DataFrame) or df_full.empty:
             return {"error": "Internal error: df_full is not available for retrieving the last data point."}

        # Get the index (date) and data of the *last* row used in the prediction sequence
        last_timestamp_index = df_full.index[-1]
        last_data_point = df_full.iloc[-1]

        if not isinstance(last_data_point, pd.Series):
            return {"error": f"Internal error: Expected last data point to be a pandas Series, but got {type(last_data_point)}."}

        # Check required OHLCV columns exist in the last data point Series
        required_ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_ohlcv_cols if col not in last_data_point.index]
        if missing_cols:
             return {"error": f"Internal error: Last data point Series is missing required columns: {missing_cols}. Available: {last_data_point.index.tolist()}"}

        # Format timestamp as Date only for daily data
        try:
             # The index should already be datetime-like from yfinance/processing
             timestamp_str = last_timestamp_index.strftime('%Y-%m-%d')
        except Exception as fmt_err:
            print(f"Warning: Could not format timestamp {last_timestamp_index}: {fmt_err}")
            timestamp_str = str(last_timestamp_index) # Fallback to string representation


        # Construct the final output dictionary safely
        output_dict = {
            "Date": timestamp_str,
            "Open": float(last_data_point['Open']),
            "High": float(last_data_point['High']),
            "Low": float(last_data_point['Low']),
            "Close": float(last_data_point['Close']),
            "Volume": float(last_data_point.get('Volume', 0.0)), # Use .get() for robustness
            "Signal": predicted_signal
        }

        print(f"Prediction successful. Signal: {predicted_signal} for date: {timestamp_str}")
        # print(f"Output dict: {output_dict}") # Optional: print the full output
        return output_dict

    except Exception as e:
        import traceback
        print("--- ERROR TRACEBACK (Prediction/Output) ---")
        traceback.print_exc()
        print("--- END TRACEBACK ---")
        return {"error": f"Unexpected error during prediction or output formatting - {type(e).__name__}: {e}"}
