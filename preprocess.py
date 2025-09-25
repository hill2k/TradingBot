import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

from config import logger, PREDICTION_HORIZON
from indicators import add_indicators
from smart_money import SmartMoneyAnalysis

# Константы больше не нужны для фильтрации в реальном времени
# IMPORTANCE_CSV_PATH = "feature_importance_ensemble.csv"
# TOP_N_FEATURES = 30

class FeatureEngine:
    """
    Класс для создания полного набора признаков.
    Логика фильтрации удалена, так как модель сама выберет нужные ей признаки.
    """
    def __init__(self):
        self.smart_money_analyzer = SmartMoneyAnalysis()
        # Загрузка топ-признаков больше не нужна в этом классе
        # self.top_features = self._load_top_features()

    def create_features(self, df: pd.DataFrame, df_secondary: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Создает и добавляет в DataFrame все необходимые признаки.
        Больше не фильтрует их.
        """
        logger.info("Начало создания признаков...")
        df_copy = df.copy()

        df_with_ta = add_indicators(df_copy)
        df_with_sm = self.smart_money_analyzer.add_features(df_with_ta)

        if df_secondary is not None:
            logger.info("Добавление признаков из вторичного символа (BTC)...")
            df_btc = df_secondary[['close', 'volume', 'RSI_14', 'MACD_12_26_9']].add_prefix('btc_')
            df_with_sm = pd.merge(df_with_sm, df_btc, left_index=True, right_index=True, how='left').ffill()
            logger.info("Признаки BTC успешно интегрированы.")

        df_with_sm.columns = [col.replace(' ', '_').replace('(', '').replace(')', '') for col in df_with_sm.columns]
        logger.info(f"Создание признаков завершено. Итоговое количество признаков: {len(df_with_sm.columns)}")
            
        return df_with_sm

class DataPreprocessor:
    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length
        self.prediction_horizon = PREDICTION_HORIZON
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_names = []

    def prepare_for_lstm(self, df: pd.DataFrame, fit_scaler: bool = False):
        df_copy = df.copy()
        df_copy['future_price'] = df_copy['close'].shift(-self.prediction_horizon)
        df_copy.dropna(subset=['future_price'], inplace=True)
        
        y_price = ((df_copy['future_price'] - df_copy['close']) / df_copy['close'])
        y_conf = (y_price.abs() > 0.001).astype(int)

        features_df = df_copy.select_dtypes(include=np.number)
        cols_to_drop = [col for col in ['open', 'high', 'low', 'close', 'volume', 'turnover', 'future_price'] if col in features_df.columns]
        features_df = features_df.drop(columns=cols_to_drop, errors='ignore')

        self.feature_names = features_df.columns.tolist()
        
        if features_df.empty:
            logger.error("Не осталось признаков для обучения.")
            return None, None, None, None, None

        if fit_scaler:
            self.scaler.fit(features_df)
        
        if not hasattr(self.scaler, 'scale_') or self.scaler.scale_ is None:
             logger.error("Scaler не обучен. Запустите с fit_scaler=True.")
             return None, None, None, None, None

        features_scaled = self.scaler.transform(features_df)

        X, y_p, y_c = [], [], []
        for i in range(len(features_scaled) - self.sequence_length + 1):
            X.append(features_scaled[i:i + self.sequence_length])
            y_p.append(y_price.iloc[i + self.sequence_length - 1])
            y_c.append(y_conf.iloc[i + self.sequence_length - 1])
        
        if not X:
            return None, None, None, None, None

        return np.array(X), np.array(y_p), np.array(y_c), self.feature_names, self.scaler

def prepare_for_garch(df: pd.DataFrame) -> pd.Series | None:
    logger.info("Подготовка данных для GARCH модели...")
    if 'close' not in df.columns or df['close'].isnull().all():
        return None
    returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    return returns