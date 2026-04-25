import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import logging

from mamba_ssm import Mamba

# --- 1. پیکربندی برای پیش‌بینی قیمت low ---
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CONFIG = {
    "FILE_PATH": 'btcirt_features_low.csv',  # فایل ویژگی‌های جدید
    "TARGET_COL": 'low',  # متغیر هدف تغییر کرده است
    # --- لیست ویژگی‌های جدید بر اساس low ---
    # این لیست را از خروجی اسکریپت Preparation_low.py کپی کنید
    "FEATURE_COLS": [
        'open', 'high', 'low', 'close', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'log_returns_low', 'low_lag_1', 'low_lag_12', 'low_lag_24', 
        'sma_8_low', 'sma_24_low', 'sma_50_low', 'sma_100_low', 'ema_12_low', 'ema_26_low', 'dist_from_sma_24_low', 'volatility_12_low', 'volatility_24_low', 
        'realized_vol_daily_low', 'garman_klass_vol', 'rsi_14_low', 'macd_histogram_low', 'bb_percentb_low', 'bb_bandwidth_low', 'atr_14', 'body', 'upper_wick', 'lower_wick', 
        'range', 'body_ratio', 'upper_shadow', 'lower_shadow', 'close_location', 'zscore_20_low', 'price_zscore_low', 'autocorr_1_low', 'slope_30_low', 'momentum_12_low', 
        'high_low_ratio', 'sma_high_low_ratio_24', 'high_low_ratio_lag_1', 'rsi_lag_1', 'atr_lag_1', 'rsi_rolling_mean_24', 'rsi_rolling_std_24', 'adx_14', 'is_trending', 
        'fourier_power_daily_low', 'fourier_power_weekly_low'
    ],
    # --- پیکربندی مدل (بهترین مقادیر پیدا شده) ---
    "SEQUENCE_LENGTH": 360,
    "PREDICTION_LENGTH": 24,
    "BATCH_SIZE": 512,
    "NUM_EPOCHS": 100,
    "LEARNING_RATE": 5e-5,
    "D_MODEL": 512,
    "N_LAYERS": 2,
    "D_STATE": 32,
    "DROPOUT": 0.1, # از مقدار بهینه استفاده شد
    "PATIENCE": 15,
    "MODEL_SAVE_PATH": 'saved_models/mamba_btcirt_low_model.pth' # مسیر ذخیره مدل جدید
}

# --- 2. کلاس‌ها و توابع ---
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

# کلاس دیتاست برای پیش‌بینی low
class TimeSeriesDataset(Dataset):
    def __init__(self, X, log_target, context_length, horizon): # نام متغیر به log_target تغییر یافت
        self.X = X.astype(np.float32)
        self.log_target = log_target.astype(np.float32)
        self.context_length = context_length
        self.horizon = horizon
    def __len__(self):
        return len(self.X) - self.context_length - self.horizon + 1
    def __getitem__(self, idx):
        x = self.X[idx: idx + self.context_length]
        base_price = self.log_target[idx + self.context_length - 1] # استفاده از log_target
        future_prices = self.log_target[idx + self.context_length: idx + self.context_length + self.horizon] # استفاده از log_target
        y = future_prices - base_price
        return torch.tensor(x), torch.tensor(y, dtype=torch.float32)

class ScaledTargetDataset(Dataset):
    def __init__(self, base_dataset, scaler_y):
        self.base_dataset = base_dataset
        self.scaler_y = scaler_y
    def __len__(self): return len(self.base_dataset)
    def __getitem__(self, idx):
        x, y_raw = self.base_dataset[idx]
        y_scaled = self.scaler_y.transform(y_raw.reshape(-1, 1)).flatten()
        return x, torch.tensor(y_scaled, dtype=torch.float32)

class MambaForecaster(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, horizon, d_state=16, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.mamba_blocks = nn.ModuleList([Mamba(d_model, d_state=d_state, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.mamba_blocks:
            x = block(x)
        x = self.norm(x[:, -1, :])
        return self.head(x)

def train_epoch(model, loader, criterion_mse, criterion_mae, optimizer, device, alpha=0.1):
    model.train()
    total_loss = 0.0
    for X, y in tqdm(loader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        
        loss_mse = criterion_mse(preds, y)
        loss_mae = criterion_mae(preds, y)
        loss = loss_mse + alpha * loss_mae
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion_mse, criterion_mae, device, alpha=0.1):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Evaluating", leave=False):
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss_mse = criterion_mse(preds, y)
            loss_mae = criterion_mae(preds, y)
            loss = loss_mse + alpha * loss_mae
            total_loss += loss.item()
    return total_loss / len(loader)

# --- تابع اصلی ---
def main():
    setup_logging()
    logging.info("شروع فرآیند آموزش مدل Mamba برای پیش‌بینی قیمت low")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(CONFIG['MODEL_SAVE_PATH']), exist_ok=True)

    # آماده‌سازی داده‌ها
    logging.info("--- شروع آماده‌سازی داده‌ها ---")
    df = pd.read_csv(CONFIG['FILE_PATH']).dropna()
    X_raw = df[CONFIG['FEATURE_COLS']]
    n = len(df)
    train_end = int(n * 0.7)
    lower_bounds = X_raw[:train_end].quantile(0.01)
    upper_bounds = X_raw[:train_end].quantile(0.99)
    X_clipped = X_raw.clip(lower_bounds, upper_bounds, axis=1)
    X_clipped = X_clipped.values
    
    # خواندن قیمت‌های low به عنوان متغیر هدف
    low_prices = df[CONFIG['TARGET_COL']].values
    log_low = np.log(low_prices + 1e-8) # تغییر نام متغیر برای خوانایی

    X_train, X_val, X_test = X_clipped[:train_end], X_clipped[train_end:int(n*0.85)], X_clipped[int(n*0.85):]
    log_train, log_val, log_test = log_low[:train_end], log_low[train_end:int(n*0.85)], log_low[int(n*0.85):] # استفاده از log_low
    
    scaler_X = RobustScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)
    
    train_ds_raw = TimeSeriesDataset(X_train, log_train, CONFIG['SEQUENCE_LENGTH'], CONFIG['PREDICTION_LENGTH'])
    val_ds_raw = TimeSeriesDataset(X_val, log_val, CONFIG['SEQUENCE_LENGTH'], CONFIG['PREDICTION_LENGTH'])
    test_ds_raw = TimeSeriesDataset(X_test, log_test, CONFIG['SEQUENCE_LENGTH'], CONFIG['PREDICTION_LENGTH'])
    
    y_samples = [y.numpy() for _, y in DataLoader(train_ds_raw, batch_size=512, shuffle=False)]
    y_all = np.concatenate(y_samples).reshape(-1, 1)
    scaler_y = StandardScaler()
    scaler_y.fit(y_all)
    
    train_ds = ScaledTargetDataset(train_ds_raw, scaler_y)
    val_ds = ScaledTargetDataset(val_ds_raw, scaler_y)
    test_ds = ScaledTargetDataset(test_ds_raw, scaler_y)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)
    logging.info("--- آماده‌سازی داده‌ها به پایان رسید ---\n")

    # ساخت و آموزش مدل
    logging.info("--- شروع فرآیند آموزش مدل ---")
    model = MambaForecaster(
        input_dim=len(CONFIG['FEATURE_COLS']),
        d_model=CONFIG['D_MODEL'],
        n_layers=CONFIG['N_LAYERS'],
        horizon=CONFIG['PREDICTION_LENGTH'],
        d_state=CONFIG['D_STATE'],
        dropout=CONFIG['DROPOUT']
    ).to(device)

    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(CONFIG['NUM_EPOCHS']):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion_mse, criterion_mae, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion_mse, criterion_mae, device)
        scheduler.step(val_loss)
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {epoch_time:.2f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'scaler_X': scaler_X, 'scaler_y': scaler_y, 'config': CONFIG, 'clip_lower_bounds': lower_bounds, 'clip_upper_bounds': upper_bounds}, CONFIG['MODEL_SAVE_PATH'])
            logging.info("  → بهترین مدل ذخیره شد!")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['PATIENCE']:
                logging.info("Early stopping triggered")
                break
    
    logging.info("--- آموزش با موفقیت به پایان رسید ---")

if __name__ == '__main__':
    main()
