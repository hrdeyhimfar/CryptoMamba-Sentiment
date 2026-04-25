# train_mamba.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau # <<<--- تغییر: ایمپورت جدید
from tqdm import tqdm
import logging

from mamba_ssm import Mamba

# --- پیکربندی ---
torch.manual_seed(42)
np.random.seed(42)

CONFIG = {
    "TRAIN_DATA_PATH": 'processed_data/ethirt_walkforward_train_1h.npz',
    "VAL_DATA_PATH": 'processed_data/ethirt_walkforward_val_1h.npz',
    "SCALER_X_PATH": 'processed_data/scaler_X_1h.pkl',
    "SCALER_Y_PATH": 'processed_data/scaler_y_1h.pkl',
    "PREDICTION_LENGTH": 1,
    "BATCH_SIZE": 128,
    "NUM_EPOCHS": 350, 
    "LEARNING_RATE": 5e-4,    
    "D_MODEL": 512,
    "N_LAYERS": 3,
    "D_STATE": 16,
    "DROPOUT": 0.2,
    "WEIGHT_DECAY": 5e-4,
    "MODEL_SAVE_PATH": 'saved_models/mamba_ethirt_1h_improved.pth', # نام مدل برای تشخیص
    "PATIENCE": 25
}

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PreprocessedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class MambaForecaster(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, horizon, d_state=16, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.mamba_blocks = nn.ModuleList([Mamba(d_model, d_state=d_state, d_conv=4, expand=2) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, horizon)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02) 
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.mamba_blocks: x = block(x)
        x = self.norm(x[:, -1, :])
        return self.head(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in tqdm(loader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        
        # محدود کردن خروجی مدل به بازه [0, 1] که همان بازه MinMaxScaler است
        preds = torch.clamp(preds, 0.0, 1.0)
        
        y = y.squeeze(-1)
        
        # <<<--- تغییر: افزودن Label Smoothing --- >>>
        # برای جلوگیری از overfitting و بهبود تعمیم، مقدار کمی نویز به برچسب‌های واقعی اضافه می‌کنیم.
        # این کار باعث می‌شود مدل کمتر "مطمئن" باشد و عملکرد بهتری روی داده‌های جدید داشته باشد.
        noise = torch.randn_like(y) * 0.005 # 0.01 یک هایپرپارامتر قابل تنظیم است
        smooth_y = y + noise
        
        loss = criterion(preds, smooth_y)
        
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Evaluating", leave=False):
            X, y = X.to(device), y.to(device)
            preds = model(X)
            
            # محدود کردن خروجی مدل به بازه [0, 1] برای ارزیابی صحیح
            preds = torch.clamp(preds, 0.0, 1.0)
            
            y = y.squeeze(-1)
            loss = criterion(preds, y)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    setup_logging()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(CONFIG['MODEL_SAVE_PATH']), exist_ok=True)

    logging.info("--- شروع بارگذاری داده‌های جدید (هدف سطح قیمت) ---")
    train_data = np.load(CONFIG['TRAIN_DATA_PATH'])
    val_data = np.load(CONFIG['VAL_DATA_PATH'])
    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    input_dim = X_train.shape[2]
    logging.info(f"داده‌ها بارگذاری شدند. Train: {X_train.shape}, Val: {X_val.shape}")

    train_dataset = PreprocessedDataset(X_train, y_train)
    val_dataset = PreprocessedDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)

    logging.info("--- شروع فرآیند آموزش مدل (با ReduceLROnPlateau و Label Smoothing) ---")
    model = MambaForecaster(
        input_dim=input_dim, d_model=CONFIG['D_MODEL'], n_layers=CONFIG['N_LAYERS'],
        horizon=CONFIG['PREDICTION_LENGTH'], d_state=CONFIG['D_STATE'], dropout=CONFIG['DROPOUT']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
    
    # <<<--- تغییر: استفاده از ReduceLROnPlateau --- >>>
    # این اسکدولر نرخ یادگیری را زمانی که عملکرد روی مجموعه اعتبارسنجی بهبود پیدا نکند، کاهش می‌دهد.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(CONFIG['NUM_EPOCHS']):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr'] # گرفتن نرخ یادگیری فعلی
        logging.info(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e} | Time: {epoch_time:.2f}s")
        
        # <<<--- تغییر: فراخوانی اسکدولر --- >>>
        scheduler.step(val_loss) # نرخ یادگیری بر اساس خطای اعتبارسنجی تنظیم می‌شود
        
        # --- منطق Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(), 'config': CONFIG}, CONFIG['MODEL_SAVE_PATH'])
            logging.info("  → بهترین مدل ذخیره شد!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= CONFIG['PATIENCE']:
            logging.info(f"--- Early Stopping فعال شد. آموزش پس از {epoch+1} اپاک متوقف شد. ---")
            break
    
    logging.info("--- آموزش با موفقیت به پایان رسید ---")

if __name__ == '__main__':
    main()
