import pandas as pd

"""
Модуль для расчета технических индикаторов с использованием только pandas.
Это делает код независимым от внешних библиотек вроде pandas-ta.
"""

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет в DataFrame набор стандартных технических индикаторов.
    
    :param df: DataFrame с колонками 'high', 'low', 'close', 'volume'.
    :return: DataFrame с добавленными индикаторами.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # --- Momentum ---
    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1.0 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema_fast - ema_slow
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']

    # Stochastic Oscillator
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    df['STOCHk_14_3_3'] = 100 * ((close - low_14) / (high_14 - low_14))
    df['STOCHd_14_3_3'] = df['STOCHk_14_3_3'].rolling(window=3).mean()

    # --- Volume ---
    # OBV (On-Balance Volume)
    df['OBV'] = (pd.Series(df['volume']) * (~df['close'].diff().le(0) * 2 - 1)).cumsum()

    # MFI (Money Flow Index)
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    mf_positive = money_flow.where(typical_price.diff() > 0, 0).rolling(window=14).sum()
    mf_negative = money_flow.where(typical_price.diff() < 0, 0).rolling(window=14).sum()
    mfi_ratio = mf_positive / mf_negative
    df['MFI_14'] = 100 - (100 / (1 + mfi_ratio))
    
    # --- Volatility ---
    # Bollinger Bands
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    df['BBL_20_2.0'] = sma_20 - (std_20 * 2)
    df['BBM_20_2.0'] = sma_20
    df['BBU_20_2.0'] = sma_20 + (std_20 * 2)

    # ATR (Average True Range)
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATRr_14'] = true_range.ewm(alpha=1/14, adjust=False).mean()

    # --- Trend ---
    df['EMA_50'] = close.ewm(span=50, adjust=False).mean()
    df['EMA_100'] = close.ewm(span=100, adjust=False).mean()

    return df