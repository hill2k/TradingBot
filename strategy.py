import numpy as np
from datetime import datetime, timedelta

from utils.logger import logger

class TradingStrategy:
    """
    Класс, реализующий торговую логику: генерация сигналов,
    расчет Take Profit (TP) и Stop Loss (SL).
    """
    def __init__(self, min_change_pct: float = 0.05, max_change_pct: float = 15.0, 
                 confidence_threshold: float = 0.5, signal_cooldown_minutes: int = 6):
        self.min_change_pct = min_change_pct
        self.max_change_pct = max_change_pct
        self.confidence_threshold = confidence_threshold
        self.signal_cooldown = timedelta(minutes=signal_cooldown_minutes)
        self.last_signal_time = None

    def _is_cooldown_active(self) -> bool:
        if self.last_signal_time is None: return False
        if datetime.now() < self.last_signal_time + self.signal_cooldown:
            logger.info(f"Кулдаун активен. Следующий сигнал возможен после {self.last_signal_time + self.signal_cooldown}.")
            return True
        return False

    def _validate_prediction(self, prediction: dict) -> bool:
        """Валидирует прогноз модели по заданным порогам."""
        if not prediction or not prediction.get('success'): return False
        change_pct = prediction['change_pct']
        confidence = prediction['confidence']
        
        # --- ИЗМЕНЕНИЕ ФОРМАТИРОВАНИЯ ---
        if abs(change_pct) < self.min_change_pct:
            logger.info(f"Прогноз отклонен: изменение {change_pct:.4f}% < мин. порога {self.min_change_pct}%.")
            return False
        if abs(change_pct) > self.max_change_pct:
            logger.warning(f"Прогноз отклонен: изменение {change_pct:.4f}% > макс. порога {self.max_change_pct}%.")
            return False
        if confidence < self.confidence_threshold:
            logger.info(f"Прогноз отклонен: уверенность {confidence:.4f} < порога {self.confidence_threshold}.")
            return False
        return True

    def calculate_dynamic_sl_tp(self, current_price: float, change_pct: float, 
                                side: str, volatility: float) -> dict:
        """
        Рассчитывает динамические уровни TP и SL с учетом проскальзывания.
        """
        # --- ИЗМЕНЕНИЕ ФОРМАТИРОВАНИЯ ---
        logger.info(f"Расчет TP/SL для {side} от цены {current_price:.6f} с прогнозом {change_pct:.4f}% и vol={volatility:.4f}")

        MIN_TP_DISTANCE_PCT = 0.08
        MIN_SL_DISTANCE_PCT = 0.08

        base_tp_distance_pct = abs(change_pct) * 0.8
        base_sl_distance_pct = abs(change_pct) * 0.6
        
        tp_distance_pct = max(base_tp_distance_pct, MIN_TP_DISTANCE_PCT)
        sl_distance_pct = max(base_sl_distance_pct, MIN_SL_DISTANCE_PCT)

        vol_factor = np.clip(volatility / 0.025, 0.7, 1.5)
        
        final_tp_distance_pct = tp_distance_pct * vol_factor
        final_sl_distance_pct = sl_distance_pct / vol_factor

        logger.debug(f"vol_factor={vol_factor:.2f}, tp_dist={final_tp_distance_pct:.4f}%, sl_dist={final_sl_distance_pct:.4f}%")

        if side == "Buy":
            tp_price = current_price * (1 + final_tp_distance_pct / 100)
            sl_price = current_price * (1 - final_sl_distance_pct / 100)
        else: # Sell
            tp_price = current_price * (1 - final_tp_distance_pct / 100)
            sl_price = current_price * (1 + final_sl_distance_pct / 100)
            
        try:
            assert (side == "Buy" and tp_price > current_price > sl_price) or \
                   (side == "Sell" and sl_price > current_price > tp_price), "Логика TP/SL нарушена!"
        except AssertionError as e:
            logger.error(f"КРИТИЧЕСКАЯ ОШИБКА В РАСЧЕТЕ TP/SL: {e}")
            return {}

        return {"tp_price": tp_price, "sl_price": sl_price}

    def generate_signal(self, prediction: dict, current_price: float) -> dict | None:
        if self._is_cooldown_active(): return None
        if not self._validate_prediction(prediction): return None
        change_pct = prediction['change_pct']
        volatility = prediction['volatility']
        side = "Buy" if change_pct > 0 else "Sell"
        sl_tp_prices = self.calculate_dynamic_sl_tp(current_price, change_pct, side, volatility)
        if not sl_tp_prices: return None
        signal = {
            "timestamp": datetime.now(),
            "symbol": "XRPUSDT",
            "side": side,
            "entry_price": current_price,
            "tp_price": sl_tp_prices['tp_price'],
            "sl_price": sl_tp_prices['sl_price'],
            "prediction_source": prediction['source'],
            "predicted_change_pct": change_pct,
            "confidence": prediction['confidence'],
            "volatility": volatility
        }
        self.last_signal_time = datetime.now()
        logger.info(f"🎯 СГЕНЕРИРОВАН СИГНАЛ: {signal}")
        return signal