import torch
import joblib
import pandas as pd
import numpy as np

from utils.logger import logger
from config import DEVICE, MIN_DATA_LENGTH, SMART_MONEY_CONFIG
from lstm_model import EnhancedAttentionLSTM
from garch_model import GARCHVolatilityModel
from preprocess import FeatureEngine, prepare_for_garch, add_indicators

class HybridModel:
    def __init__(self, sequence_length: int, lstm_model_path: str, scaler_path: str, features_path: str):
        self.sequence_length = sequence_length
        self.feature_engine = FeatureEngine()
        
        try:
            logger.info(f"Загрузка артефактов LSTM: {lstm_model_path}, {scaler_path}, {features_path}")
            state_dict = torch.load(lstm_model_path, map_location=DEVICE, weights_only=True)
            self.feature_names = joblib.load(features_path)
            self.scaler = joblib.load(scaler_path)

            input_size = len(self.feature_names)
            self.lstm_model = EnhancedAttentionLSTM(input_size=input_size).to(DEVICE)
            self.lstm_model.load_state_dict(state_dict)
            self.lstm_model.eval()
            logger.info("Модель LSTM (с 2 выходами) успешно загружена и переведена в режим оценки.")

        except FileNotFoundError as e:
            logger.error(f"Ошибка: Не найден один из файлов модели: {e}. Убедитесь, что вы запустили train.py.")
            raise e

        self.garch_model = GARCHVolatilityModel()

    def _predict_with_lstm(self, df_with_features: pd.DataFrame) -> dict:
        available_features = [f for f in self.feature_names if f in df_with_features.columns]
        if len(available_features) != len(self.feature_names):
            missing = set(self.feature_names) - set(available_features)
            logger.warning(f"Пропущены признаки, необходимые для LSTM: {missing}")
            return {}

        features_df = df_with_features[self.feature_names]
        features_df = features_df.dropna()
        if len(features_df) < self.sequence_length:
            logger.warning(f"После удаления NaN осталось недостаточно данных для LSTM ({len(features_df)}).")
            return {}

        features_scaled = self.scaler.transform(features_df)
        last_sequence = features_scaled[-self.sequence_length:]
        
        if last_sequence.shape[0] < self.sequence_length: return {}

        sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prediction_fraction_tensor, confidence_tensor = self.lstm_model(sequence_tensor)
            prediction_fraction = prediction_fraction_tensor.item()
            confidence = confidence_tensor.item()

        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: ПРЕОБРАЗУЕМ ДОЛЮ В ПРОЦЕНТ ---
        prediction_pct = prediction_fraction * 100.0
        
        logger.info(f"LSTM Output: Pct={prediction_pct:.4f}%, Confidence={confidence:.4f}")

        if confidence > 0.6:
            return {"change_pct": prediction_pct, "confidence": confidence, "source": "LSTM"}
        
        logger.info("Прогноз LSTM отклонен: недостаточная уверенность.")
        return {}

    def _predict_with_smi(self, df: pd.DataFrame) -> dict:
        last_row = df.iloc[-1]
        smi_smooth = last_row.get('smart_money_index_smooth', 0)
        smi_regime = last_row.get('smart_money_regime', 0)
        bull_threshold = SMART_MONEY_CONFIG['smi_bullish_threshold']
        bear_threshold = SMART_MONEY_CONFIG['smi_bearish_threshold']
        if smi_regime == 1:
            logger.info(f"SMI Fallback: Обнаружен бычий режим (SMI: {smi_smooth:.3f} > {bull_threshold})")
            return {"change_pct": 0.5, "confidence": 0.75, "source": "SMI"}
        elif smi_regime == -1:
            logger.info(f"SMI Fallback: Обнаружен медвежий режим (SMI: {smi_smooth:.3f} < {bear_threshold})")
            return {"change_pct": -0.5, "confidence": 0.75, "source": "SMI"}
        return {}

    def _predict_volatility(self, df: pd.DataFrame) -> float | None:
        try:
            returns = prepare_for_garch(df)
            if returns is None or returns.empty or len(returns) < 50:
                 logger.warning(f"Недостаточно данных для GARCH (получено {len(returns) if returns is not None else 0}).")
                 return None
            self.garch_model.fit(returns)
            forecast = self.garch_model.predict_volatility(horizon=1)
            return forecast if forecast is not None else None
        except Exception as e:
            logger.error(f"Ошибка при прогнозировании волатильности GARCH: {e}", exc_info=True)
            return None

    def predict(self, df: pd.DataFrame, df_secondary: pd.DataFrame | None = None) -> dict:
        if len(df) < MIN_DATA_LENGTH:
            logger.warning(f"Недостаточно данных для прогноза (требуется {MIN_DATA_LENGTH}, получено {len(df)}).")
            return {"success": False}

        df_secondary_with_ta = None
        if df_secondary is not None:
            df_secondary_with_ta = add_indicators(df_secondary)

        df_with_features = self.feature_engine.create_features(df, df_secondary=df_secondary_with_ta)
        
        volatility_forecast = self._predict_volatility(df)
        if volatility_forecast is None:
            logger.error("Не удалось получить прогноз волатильности от GARCH. Прогноз невозможен.")
            return {"success": False}

        directional_prediction = {}
        try:
            lstm_prediction = self._predict_with_lstm(df_with_features)
            if lstm_prediction:
                directional_prediction = lstm_prediction
        except Exception as e:
            logger.error(f"Критическая ошибка при прогнозировании с LSTM: {e}", exc_info=True)

        if not directional_prediction:
            logger.info("LSTM не дал прогноза. Переход к fallback-логике (SMI)...")
            smi_prediction = self._predict_with_smi(df_with_features)
            if smi_prediction:
                directional_prediction = smi_prediction
        
        if directional_prediction:
            final_prediction = {
                "success": True,
                "change_pct": directional_prediction['change_pct'],
                "confidence": directional_prediction['confidence'],
                "source": directional_prediction['source'],
                "volatility": volatility_forecast
            }
            logger.info(f"✅ Итоговый прогноз сгенерирован: {final_prediction}")
            return final_prediction
        else:
            logger.info("Ни одна из моделей (LSTM, SMI) не сгенерировала уверенный прогноз.")
            return {"success": False}