import pandas as pd
import numpy as np
import torch
import joblib
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

class NoPlotTFT(TemporalFusionTransformer):
    def log_prediction(self, *args, **kwargs):
        return None


df = pd.read_csv('ethirt_features.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

df["time_idx"] = np.arange(len(df))
df["symbol"] = "CRYPTO_IRT"
df["symbol"] = df["symbol"].astype("category")

target = "close"
static_categoricals = ["symbol"]
time_idx = "time_idx"

scaler = joblib.load('full_scaler_with_target.pkl')

exclude_cols_from_scaling = [time_idx] + static_categoricals + ['datetime', 'timestamp', 'is_trending']
features_to_scale = [col for col in df.columns if col not in exclude_cols_from_scaling]

df_scaled = df.copy()
df_scaled[features_to_scale] = scaler.transform(df_scaled[features_to_scale])

checkpoint = torch.load('CRYPTO_IRT_TFT_ULTIMATE_WORKING_BEST.pth', map_location='cpu')
dataset_params = checkpoint['dataset_params']
max_encoder_length = checkpoint['max_encoder_length']
max_prediction_length = checkpoint['max_prediction_length']

training = TimeSeriesDataSet.from_parameters(dataset_params, df_scaled, predict=True)

validation = TimeSeriesDataSet.from_dataset(training, df_scaled, predict=True, stop_randomization=True)
predict_loader = validation.to_dataloader(train=False, batch_size=1, num_workers=0)

model = NoPlotTFT.from_dataset(training, loss=QuantileLoss(), log_interval=-1)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("مدل با موفقیت بارگذاری شد.")

with torch.no_grad():
    forecast_scaled = model.predict(predict_loader, mode="quantiles", return_x=False)

forecast_scaled = forecast_scaled[:, :, 3].cpu().numpy().flatten()


median_close = scaler.center_[features_to_scale.index('close')]
iqr_close = scaler.scale_[features_to_scale.index('close')]

forecast_original = forecast_scaled * iqr_close + median_close

# 9. ساخت دیتافریم نهایی
last_dt = df['datetime'].iloc[-1]
dates = [last_dt + timedelta(hours=i+1) for i in range(max_prediction_length)]

result = pd.DataFrame({
    'datetime': dates,
    'predicted_price_toman': np.round(forecast_original).astype('int64')
})

result.to_csv('CRYPTO_IRT_30Days_Prediction_FIXED_AND_CORRECT.csv', index=False)

