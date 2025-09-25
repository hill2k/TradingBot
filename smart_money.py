import pandas as pd
import numpy as np
# pandas_ta больше не нужен!

from utils.logger import logger

class SmartMoneyAnalysis:
    """
    Класс для расчета продвинутых признаков на основе концепций Smart Money.
    ФИНАЛЬНАЯ ВЕРСИЯ: с самостоятельной реализацией VWAP и A/D для устранения конфликтов зависимостей.
    """
    def __init__(self):
        pass

    def _detect_volume_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Индикатор 1: Кластерные объемы"""
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_std_20'] = df['volume'].rolling(20).std()
        
        threshold_stat = df['volume_ma_20'] + (df['volume_std_20'] * 2)
        df['volume_cluster_stat'] = df['volume'] > threshold_stat
        
        threshold_quant = df['volume'].rolling(50).quantile(0.85)
        df['volume_cluster_quant'] = df['volume'] > threshold_quant
        
        df['volume_cluster'] = df['volume_cluster_stat'] | df['volume_cluster_quant']
        
        df['bullish_volume_cluster'] = df['volume_cluster'] & (df['close'] > df['open'])
        df['bearish_volume_cluster'] = df['volume_cluster'] & (df['close'] < df['open'])
        
        df['volume_cluster_strength'] = np.where(
            df['volume_cluster'],
            (df['volume'] - df['volume_ma_20']) / df['volume_std_20'],
            0
        )
        return df

    def _detect_whale_activity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Индикатор 2: Активность китов"""
        df['candle_body_size'] = abs(df['close'] - df['open'])
        df['candle_body_pct'] = df['candle_body_size'] / df['open'] * 100
        
        volume_threshold = df['volume'].rolling(50).quantile(0.9)
        body_threshold = df['candle_body_pct'].rolling(50).quantile(0.85)
        
        df['whale_activity'] = (df['volume'] > volume_threshold) & \
                               (df['candle_body_pct'] > body_threshold)
        
        df['bullish_whale'] = df['whale_activity'] & (df['close'] > df['open'])
        df['bearish_whale'] = df['whale_activity'] & (df['close'] < df['open'])
        return df

    def _calculate_vwap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Индикатор 3: VWAP Analysis (ручная реализация с дневным сбросом)."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['price_volume'] = typical_price * df['volume']
        
        # Группируем по дате и считаем кумулятивные суммы внутри каждого дня
        # Это ключ к правильному расчету VWAP, который сбрасывается каждый день
        daily_groups = df.groupby(df.index.date)
        cumulative_price_volume = daily_groups['price_volume'].cumsum()
        cumulative_volume = daily_groups['volume'].cumsum()

        df['VWAP_D'] = cumulative_price_volume / cumulative_volume.replace(0, np.nan)
        
        df['vwap_deviation'] = (df['close'] - df['VWAP_D']) / df['VWAP_D'] * 100
        df['price_above_vwap'] = df['close'] > df['VWAP_D']

        df.drop(columns=['price_volume'], inplace=True)
        return df

    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Индикатор 4: Accumulation/Distribution (ручная реализация)."""
        # Money Flow Multiplier
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.fillna(0) # Заполняем NaN, если high == low, чтобы избежать ошибок
        
        # Money Flow Volume
        mfv = mfm * df['volume']
        
        df['ad'] = mfv.cumsum()
        df['ad_momentum'] = df['ad'].diff()
        df['ad_momentum_smooth'] = df['ad_momentum'].rolling(5).mean()
        return df

    def _detect_liquidity_sweeps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Индикатор 5: Ликвидные свипы"""
        df['recent_high'] = df['high'].rolling(20).max()
        df['recent_low'] = df['low'].rolling(20).min()
        
        df['high_sweep_candidate'] = (
            (df['high'] > df['recent_high'].shift(1)) &
            (df['close'] < df['open']) &
            (df['volume'] > df['volume'].rolling(20).mean() * 1.5)
        )
        
        df['low_sweep_candidate'] = (
            (df['low'] < df['recent_low'].shift(1)) &
            (df['close'] > df['open']) &
            (df['volume'] > df['volume'].rolling(20).mean() * 1.5)
        )
        return df

    def _detect_institutional_footprint(self, df: pd.DataFrame) -> pd.DataFrame:
        """Индикатор 6: Institutional Footprint"""
        if 'candle_body_pct' not in df.columns:
             df['candle_body_pct'] = abs(df['close'] - df['open']) / df['open'] * 100
        if 'volume_ma_20' not in df.columns:
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
        if 'VWAP_D' not in df.columns:
            df = self._calculate_vwap_features(df)

        df['stealth_trading'] = (
            (df['volume'] > df['volume_ma_20'] * 2) &
            (df['candle_body_pct'] < 0.5)
        )
        
        df['consistent_buying'] = (
            (df['close'] > df['VWAP_D']) &
            (df['volume'] > df['volume_ma_20']) &
            (df['close'] > df['open'])
        ).rolling(3).sum() >= 2
        
        df['consistent_selling'] = (
            (df['close'] < df['VWAP_D']) &
            (df['volume'] > df['volume_ma_20']) &
            (df['close'] < df['open'])
        ).rolling(3).sum() >= 2
        
        df['buying_absorption'] = df['stealth_trading'] & (df['close'] >= df['open'])
        df['selling_absorption'] = df['stealth_trading'] & (df['close'] <= df['open'])
        return df

    def _calculate_smart_money_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Комбинированный Smart Money Index, согласно спецификации."""
        components = []
        required_cols = [
            'bullish_volume_cluster', 'bearish_volume_cluster', 'vwap_deviation',
            'bullish_whale', 'bearish_whale', 'ad_momentum_smooth', 'consistent_buying',
            'consistent_selling', 'buying_absorption', 'selling_absorption'
        ]
        if not all(col in df.columns for col in required_cols):
            logger.warning("Не все колонки для SMI были созданы. Расчет SMI пропущен.")
            df['smart_money_index'] = 0
            df['smart_money_index_smooth'] = 0
            df['smart_money_regime'] = 0
            return df

        components.append(df['bullish_volume_cluster'].astype(int) * 0.2)
        components.append(df['bearish_volume_cluster'].astype(int) * -0.2)
        
        vwap_component = np.tanh(df['vwap_deviation'].fillna(0) / 3) * 0.25
        components.append(vwap_component)
        
        components.append(df['bullish_whale'].astype(int) * 0.2)
        components.append(df['bearish_whale'].astype(int) * -0.2)
        
        ad_norm = df['ad_momentum_smooth'].fillna(0)
        ad_norm = ad_norm / (df['ad_momentum_smooth'].abs().rolling(100).mean().replace(0, 1))
        ad_component = np.tanh(ad_norm) * 0.2
        components.append(ad_component)
        
        components.append(df['consistent_buying'].astype(int) * 0.075)
        components.append(df['consistent_selling'].astype(int) * -0.075)
        components.append(df['buying_absorption'].astype(int) * 0.075)
        components.append(df['selling_absorption'].astype(int) * -0.075)
        
        df['smart_money_index'] = sum(components)
        df['smart_money_index_smooth'] = df['smart_money_index'].rolling(3).mean()
        
        df['smart_money_regime'] = np.where(
            df['smart_money_index_smooth'] > 0.3, 1,
            np.where(df['smart_money_index_smooth'] < -0.3, -1, 0)
        )
        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Последовательно вызывает все методы для создания полного набора признаков Smart Money.
        """
        logger.info("Создание признаков Smart Money (самодостаточная версия)...")
        df_copy = df.copy()
        
        if not isinstance(df_copy.index, pd.DatetimeIndex):
             raise TypeError("Индекс DataFrame должен быть типа pd.DatetimeIndex для расчета VWAP.")

        df_copy = self._detect_volume_clusters(df_copy)
        df_copy = self._detect_whale_activity(df_copy)
        df_copy = self._calculate_vwap_features(df_copy)
        df_copy = self._calculate_accumulation_distribution(df_copy)
        df_copy = self._detect_liquidity_sweeps(df_copy)
        df_copy = self._detect_institutional_footprint(df_copy)
        df_copy = self._calculate_smart_money_index(df_copy)
        
        logger.info("Все признаки Smart Money успешно созданы.")
        return df_copy