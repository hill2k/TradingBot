import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import joblib

from data_fetcher import BybitDataFetcher
from preprocess import FeatureEngine, DataPreprocessor
from lstm_model import EnhancedAttentionLSTM
from config import SYMBOL, PRIMARY_TIMEFRAME, DEVICE, SECONDARY_SYMBOL, PREDICTION_HORIZON
from utils.logger import logger
from indicators import add_indicators

# --- Параметры обучения ---
# --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
TRAIN_DATA_LIMIT = 12000 # Увеличиваем количество данных для анализа
# --------------------------
SEQUENCE_LENGTH = 24
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15
MODEL_SAVE_PATH = "trading_bot_model.pth"
SCALER_SAVE_PATH = "scaler.pkl"
FEATURES_SAVE_PATH = "feature_names.pkl"

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} из {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Сохранение модели...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def run_training():
    try:
        logger.info("--- Этап 1: Загрузка и подготовка данных ---")
        fetcher = BybitDataFetcher()
        
        logger.info(f"Загрузка данных для основного символа ({SYMBOL})...")
        raw_data = fetcher.fetch_historical_data(SYMBOL, PRIMARY_TIMEFRAME, TRAIN_DATA_LIMIT)
        if raw_data is None or raw_data.empty:
            raise ConnectionError("Не удалось загрузить основные данные для обучения.")

        logger.info(f"Загрузка данных для вторичного символа ({SECONDARY_SYMBOL})...")
        btc_data = fetcher.fetch_historical_data(SECONDARY_SYMBOL, "60", TRAIN_DATA_LIMIT)
        if btc_data is None or btc_data.empty:
            raise ConnectionError("Не удалось загрузить данные по BTC для обучения.")

        logger.info("--- Этап 2: Создание признаков ---")
        feature_engine = FeatureEngine()
        
        btc_data_with_ta = add_indicators(btc_data)
        
        df_with_features = feature_engine.create_features(raw_data, df_secondary=btc_data_with_ta, is_training=True)
        
        df_with_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_with_features.dropna(inplace=True)

        logger.info("--- Этап 3: Предобработка данных для LSTM ---")
        
        preprocessor = DataPreprocessor(sequence_length=SEQUENCE_LENGTH)
        
        X, y_price, y_conf, feature_names, scaler = preprocessor.prepare_for_lstm(df_with_features, fit_scaler=True)
        
        if X is None:
            raise ValueError("Не удалось создать обучающие последовательности.")

        joblib.dump(scaler, SCALER_SAVE_PATH)
        joblib.dump(feature_names, FEATURES_SAVE_PATH)
        logger.info(f"Scaler сохранен в {SCALER_SAVE_PATH}")
        logger.info(f"Список признаков сохранен в {FEATURES_SAVE_PATH}")

        split_index = int(len(X) * 0.8)
        X_train, X_val = X[:split_index], X[split_index:]
        y_price_train, y_price_val = y_price[:split_index], y_price[split_index:]
        y_conf_train, y_conf_val = y_conf[:split_index], y_conf[split_index:]
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_price_train_t = torch.tensor(y_price_train, dtype=torch.float32).to(DEVICE)
        y_conf_train_t = torch.tensor(y_conf_train, dtype=torch.float32).to(DEVICE)
        
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
        y_price_val_t = torch.tensor(y_price_val, dtype=torch.float32).to(DEVICE)
        y_conf_val_t = torch.tensor(y_conf_val, dtype=torch.float32).to(DEVICE)

        train_dataset = TensorDataset(X_train_t, y_price_train_t, y_conf_train_t)
        val_dataset = TensorDataset(X_val_t, y_price_val_t, y_conf_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        logger.info(f"Данные готовы: Train - {len(X_train_t)} сэмплов, Val - {len(X_val_t)} сэмплов.")
        logger.info(f"Количество признаков: {X_train.shape[2]}")

        model = EnhancedAttentionLSTM(input_size=X_train.shape[2]).to(DEVICE)
        
        criterion_price = nn.SmoothL1Loss()
        criterion_conf = nn.BCELoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7)
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True, path=MODEL_SAVE_PATH)

        logger.info("--- Начало обучения модели с двумя выходами ---")
        for epoch in range(EPOCHS):
            model.train()
            train_losses = []
            for features, labels_price, labels_conf in train_loader:
                optimizer.zero_grad()
                pred_price, pred_conf = model(features)
                
                loss_price = criterion_price(pred_price, labels_price)
                loss_conf = criterion_conf(pred_conf, labels_conf)
                
                total_loss = loss_price + loss_conf 
                
                total_loss.backward()
                optimizer.step()
                train_losses.append(total_loss.item())
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for features, labels_price, labels_conf in val_loader:
                    pred_price, pred_conf = model(features)
                    loss_price = criterion_price(pred_price, labels_price)
                    loss_conf = criterion_conf(pred_conf, labels_conf)
                    total_loss = loss_price + loss_conf
                    val_losses.append(total_loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            logger.info(f"Эпоха {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                logger.info("Ранняя остановка.")
                break
        
        logger.info("Обучение завершено. Лучшая модель сохранена в " + MODEL_SAVE_PATH)

    except Exception as e:
        logger.error(f"Произошла ошибка во время обучения: {e}", exc_info=True)

if __name__ == '__main__':
    run_training()