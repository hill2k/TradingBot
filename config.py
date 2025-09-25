import os
import torch
from dotenv import load_dotenv
from utils.logger import logger

# --- Загрузка переменных окружения ---
load_dotenv(dotenv_path='API.env')

if os.getenv("BYBIT_API_KEY"):
    logger.info("Файл API.env найден и ключи загружены.")
else:
    logger.warning("Файл API.env не найден или пуст. Используются переменные окружения системы.")

# --- Ключевая логика режимов ---
DEMO_MODE = True 
USE_TESTNET = False

if DEMO_MODE:
    logger.info("🤖 Бот работает в режиме ДЕМО-СЧЕТА (Paper Trading). Ордера будут отправляться на демо-счет Bybit.")
else:
    logger.warning("🚨 БОТ РАБОТАЕТ В РЕЖИМЕ РЕАЛЬНОЙ ТОРГОВЛИ.")

# --- Основные параметры ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Вычисления будут производиться на устройстве: {DEVICE}")

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    logger.warning("API ключи не найдены. Функционал, требующий аутентификации, будет недоступен.")

# --- Параметры торговли ---
SYMBOL = "XRPUSDT"
PRIMARY_TIMEFRAME = "60"
# --- НОВЫЕ ПАРАМЕТРЫ ---
SECONDARY_SYMBOL = "BTCUSDT" # Добавляем BTC для анализа
PREDICTION_HORIZON = 4       # Прогнозируем на 4 часа вперед
# --------------------------

LEVERAGE = 75
MIN_ORDER_VALUE_USDT = 5.0

# --- НОВЫЕ ЛИМИТЫ ЗАГРУЗКИ ДАННЫХ ---
PRIMARY_DATA_LIMIT = 1000   # Основные данные (1000 свечей по 1 часу)
SECONDARY_DATA_LIMIT = 4000 # Дополнительные данные (4000 свечей по 15 мин)
MIN_DATA_LENGTH = 300       # Минимально необходимое количество свечей для старта
# ------------------------------------

# --- Параметры управления рисками ---
RISK_PER_TRADE_PERCENT = 1.5
MAX_POSITIONS = 1
MAX_DAILY_TRADES = 8
MAX_DAILY_LOSS_PERCENT = -5.0
EMERGENCY_STOP_PERCENT = -10.0
MIN_BALANCE_USDT = 10.0

# --- ГЛАВНЫЙ ЦИКЛ ---
TRADING_INTERVAL_MINUTES = 10

# ... (остальная часть файла без изменений) ...
SMART_MONEY_CONFIG = {
    'volume_cluster_quantile': 0.85,
    'volume_cluster_std_multiplier': 2.0,
    'whale_volume_quantile': 0.9,
    'whale_body_quantile': 0.85,
    'strong_vwap_deviation': 2.0,
    'extreme_vwap_deviation': 5.0,
    'smi_bullish_threshold': 0.3,
    'smi_bearish_threshold': -0.3,
    'smi_strong_signal_threshold': 0.5,
    'max_risk_per_trade_pct': 1.5,
    'cooldown_period_minutes': 30,
}