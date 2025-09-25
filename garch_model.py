import pandas as pd
import numpy as np
from arch import arch_model

from utils.logger import logger

class GARCHVolatilityModel:
    """
    Класс для моделирования и прогнозирования волатильности с использованием GARCH(1,1).
    Версия 4.0: Финальная рабочая версия, исправляющая DataScaleWarning.
    """
    def __init__(self, p: int = 1, q: int = 1, dist: str = 'Normal'):
        self.p = p
        self.q = q
        self.dist = dist
        self.model = None
        self.results = None

    def fit(self, returns: pd.Series):
        """
        Обучает модель GARCH на предоставленных данных.
        :param returns: Временной ряд ЛОГАРИФМИЧЕСКИХ доходностей (в долях, НЕ в процентах).
        """
        if returns.empty or returns.isnull().all():
            logger.warning("Пустой или полностью NaN ряд доходностей передан в GARCH.fit().")
            self.results = None
            return

        try:
            # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Мы передаем данные, умноженные на 100,
            # как ожидает библиотека для численной стабильности.
            self.model = arch_model(returns * 100, vol='Garch', p=self.p, q=self.q, dist=self.dist)
            self.results = self.model.fit(disp='off')
            logger.info("Модель GARCH успешно обучена.")
        except Exception as e:
            logger.error(f"Ошибка при обучении GARCH модели: {e}", exc_info=True)
            self.results = None

    def predict_volatility(self, horizon: int = 1) -> float | None:
        """
        Прогнозирует волатильность на 'horizon' шагов вперед.
        Возвращает прогноз в виде ОДНОГО ЧИСЛА (float) в ДОЛЯХ (например, 0.015 для 1.5%).
        """
        if self.results is None:
            logger.error("Модель GARCH не обучена. Вызовите .fit() перед прогнозированием.")
            return None
        
        forecast = self.results.forecast(horizon=horizon, reindex=False)
        
        # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ:
        # 1. Прогноз variance в единицах (%^2), т.к. на вход подавали log_returns * 100
        # 2. Берем корень, чтобы получить ст. отклонение в %
        predicted_volatility_percent = np.sqrt(forecast.variance.values[0, 0])
        
        # 3. Делим на 100, чтобы получить долю, которую ожидает strategy.py
        predicted_volatility_fraction = predicted_volatility_percent / 100.0
        
        logger.info(f"Прогноз волатильности на {horizon} шаг(а): {predicted_volatility_fraction:.4f} (или {predicted_volatility_percent:.2f}%)")
        return predicted_volatility_fraction