
import numpy as np
import pandas as pd
import os

def calculate_adx(high, low, close, window=14):
    """محاسبه شاخص میانگین جهت‌دار (ADX)"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = abs(100 * (minus_dm.rolling(window).mean() / atr))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=window).mean()
    return adx

def rolling_slope(series, window=30):
    """محاسبه شیب روند در یک پنجره متحرک"""
    def slope(x):
        x_clean = x.dropna()
        if len(x_clean) < 2:
            return np.nan
        idx_clean = np.arange(len(x_clean))
        return np.polyfit(idx_clean, x_clean, 1)[0]
    return series.rolling(window).apply(slope, raw=False)


def add_time_features(df):
    df["hour"] = df.index.hour
    df["weekday"] = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    return df

def add_return_and_lag_features(df):
    df["log_return_open"] = np.log(df["open"] / df["open"].shift(1))
    
    df["open_lag_1"] = df["open"].shift(1)
    df["open_lag_12"] = df["open"].shift(12)
    df["open_lag_24"] = df["open"].shift(24)
    return df

def add_moving_average_features(df):
    """اضافه کردن ویژگی‌های مبتنی بر میانگین متحرک بر اساس open"""
    df["sma_8_open"] = df["open"].rolling(window=8).mean()
    df["sma_24_open"] = df["open"].rolling(window=24).mean()
    df["sma_50_open"] = df["open"].rolling(window=50).mean()
    df["sma_100_open"] = df["open"].rolling(window=100).mean()
    
    df["ema_12_open"] = df["open"].ewm(span=12, adjust=False).mean()
    df["ema_26_open"] = df["open"].ewm(span=26, adjust=False).mean()
    
    df["dist_from_sma_24_open"] = (df["open"] / df["sma_24_open"]) - 1
    return df

def add_volatility_features(df):
    """اضافه کردن ویژگی‌های نوسان‌پذیری بر اساس بازدهی open"""
    df["volatility_12_open"] = df["log_return_open"].rolling(window=12).std()
    df["volatility_24_open"] = df["log_return_open"].rolling(window=24).std()
    
    df["realized_vol_daily_open"] = (df["log_return_open"].resample('D').transform(lambda x: np.sqrt(np.sum(x**2))))
    df["realized_vol_daily_open"] = df["realized_vol_daily_open"].bfill()

    df["high"].replace(0, 1e-8, inplace=True)
    df["low"].replace(0, 1e-8, inplace=True)
    df["open"].replace(0, 1e-8, inplace=True)
    df["garman_klass_vol"] = (0.5 * (np.log(df["high"] / df["low"])**2)) - (2*np.log(2)-1) * (np.log(df["close"] / df["open"])**2)
    return df

def add_technical_indicators(df):
    # RSI
    delta = df["open"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.inf)
    df["rsi_14_open"] = 100 - (100 / (1 + rs))

    # MACD
    macd = df["ema_12_open"] - df["ema_26_open"]
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_histogram_open"] = macd - signal

    # Bollinger Bands
    bb_mid = df["sma_24_open"]
    bb_std = df["open"].rolling(window=24).std()
    upper = bb_mid + 2 * bb_std
    lower = bb_mid - 2 * bb_std
    df["bb_percentb_open"] = (df["open"] - lower) / (upper - lower)
    df["bb_bandwidth_open"] = (upper - lower) / bb_mid
    
    # ATR
    high_low = df["high"] - df["low"]
    high_close_prev = (df["high"] - df["close"].shift(1)).abs()
    low_close_prev = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(window=14).mean()
    return df

def add_price_action_features(df):
    """ویژگی‌های اکشن قیمت (این‌ها به OHLC وابسته هستند و تغییر نمی‌کنند)"""
    df["body"] = df["close"] - df["open"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["range"] = df["high"] - df["low"]
    df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)
    df["upper_shadow"] = df["high"] - np.maximum(df["close"], df["open"])
    df["lower_shadow"] = np.minimum(df["close"], df["open"]) - df["low"]
    df["close_location"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
    df["close_location"] = df["close_location"].fillna(0.5)
    return df

def add_statistical_and_trend_features(df):
    """اضافه کردن ویژگی‌های آماری، روند و مومنتوم بر اساس open"""
    df["zscore_20_open"] = (df["open"] - df["sma_24_open"]) / df["open"].rolling(window=24).std()
    df["price_zscore_open"] = (df["open"] - df["sma_50_open"]) / df["open"].rolling(window=50).std()

    df["autocorr_1_open"] = df["log_return_open"].rolling(30).apply(lambda x: x.autocorr(lag=1), raw=False)
    df["slope_30_open"] = rolling_slope(df["open"], 30)

    df["momentum_12_open"] = df["open"] - df["open"].shift(12)
    return df

def add_advanced_features(df):
    """اضافه کردن ویژگی‌های پیشرفته بر اساس open"""
    print("Calculating advanced features (ADX, Fourier, Hurst) for 'open' price...")
    
    df["adx_14"] = calculate_adx(df["high"], df["low"], df["close"])
    df["is_trending"] = (df["adx_14"] > 25).astype(int)

    window_size = 168
    daily_freq_target = 1 / 24
    weekly_freq_target = 1 / 168
    df["fourier_power_daily_open"] = df["open"].rolling(window_size).apply(lambda x: calculate_fourier_power(x, daily_freq_target), raw=False)
    df["fourier_power_weekly_open"] = df["open"].rolling(window_size).apply(lambda x: calculate_fourier_power(x, weekly_freq_target), raw=False)

    window_size_hurst = 250
    df["hurst_exponent_open"] = df["open"].rolling(window=window_size_hurst).apply(calculate_hurst_exponent_robust, args=(window_size_hurst,))
    
    print("Advanced features calculation complete.")
    return df

def add_interaction_and_lag_features(df):
    """اضافه کردن ویژگی‌های ترکیبی و لگ اندیکاتورها"""
    df["high_low_ratio"] = df["high"] / df["low"]
    df["sma_high_low_ratio_24"] = df["high_low_ratio"].rolling(window=24).mean()
    df["high_low_ratio_lag_1"] = df["high_low_ratio"].shift(1)

    df["rsi_lag_1"] = df["rsi_14_open"].shift(1)
    df["atr_lag_1"] = df["atr_14"].shift(1)
    df["rsi_rolling_mean_24"] = df["rsi_14_open"].rolling(window=24).mean()
    df["rsi_rolling_std_24"] = df["rsi_14_open"].rolling(window=24).std()
    return df

# --- تابع اصلی برای اجرای فرآیند ---

def build_features(input_csv, output_csv, horizon=168):
    """تابع اصلی برای خواندن داده، محاسبه ویژگی‌ها و ذخیره نتیجه"""
    print("Reading and preparing data...")
    df = pd.read_csv(input_csv)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    print("Calculating features based on 'open' price for 7-day horizon...")
    df = add_time_features(df)
    df = add_return_and_lag_features(df)
    df = add_moving_average_features(df)
    df = add_volatility_features(df)
    df = add_technical_indicators(df)
    df = add_price_action_features(df)
    df = add_statistical_and_trend_features(df)
    df = add_interaction_and_lag_features(df)
    df = add_advanced_features(df)

    # --- محاسبه متغیر هدف و ویژگی‌های مرتبط با افق زمانی ---
    # محاسبه هدف اصلی (تفاوت قیمت خام)
    df['log_return_open_target'] = np.log(df['open'] / df['open'].shift(1))

    # محاسبه اهداف با تأخیر برای افق‌های زمانی مختلف
    for h in [1, 12, 24, 48, 168]: # 1h, 12h, 2d, 7d
        df[f'target_open_{h}h'] = df['open'].shift(-h)

    # محاسبه اهداف با تأخیر (Lagged Targets)
    for lag in [1, 24, 168]:
        df[f'target_open_168h_lag_{lag}'] = df[f'target_open_168h'].shift(lag)

    print("\nDropping unnecessary and problematic columns...")
    if 'hurst_exponent_open' in df.columns:
        df.drop(columns=['hurst_exponent_open'], inplace=True)
        print("Dropped 'hurst_exponent_open' column.")
    
    cols_to_drop = ['hour', 'weekday']
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Dropped auxiliary columns: {cols_to_drop}")

    print("\nCleaning up data...")
    df.bfill(inplace=True)
    df.dropna(inplace=True)

    print(f"Final shape after cleaning: {df.shape}")

    print(f"Saving features to {output_csv}...")
    df.to_csv(output_csv)
    
    print("\nFeature engineering process completed successfully!")
    print(df.head())

    print("\n--- Ready-to-copy feature list for model configuration ---")
    header_df = pd.read_csv(output_csv, nrows=0)
    all_columns = header_df.columns.tolist()
    non_feature_columns = ['datetime', 'timestamp', 'target_open_1h', 'target_open_12h', 'target_open_24h', 'target_open_48h', 'target_open_168h', 'target_open_168h_lag_1', 'target_open_168h_lag_24', 'target_open_168h_lag_168']
    feature_columns = [col for col in all_columns if col not in non_feature_columns]
    formatted_list = [f"'{col}'" for col in feature_columns]
    print(f"[{', '.join(formatted_list)}]")

if __name__ == '__main__':
    INPUT_CSV_FILE = 'BTCIRT_1647822600_1765769400.csv'
    OUTPUT_CSV_FILE = 'btcirt_features_open_7d_enhanced.csv'
    
    # ارسال متغیرها به عنوان پارامتر تابع
    build_features(INPUT_CSV_FILE, OUTPUT_CSV_FILE)
