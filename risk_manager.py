from datetime import datetime
from utils.logger import logger
from config import (
    MAX_DAILY_TRADES, MAX_DAILY_LOSS_PERCENT, EMERGENCY_STOP_PERCENT,
    MAX_POSITIONS, MIN_BALANCE_USDT, SYMBOL
)
from trading import BybitTrader

class RiskManager:
    """
    Класс для применения глобальных правил управления рисками перед исполнением сделки.
    """
    def __init__(self):
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.start_of_day_balance = None
        self.initial_balance = None # Баланс при первом запуске
        self.last_check_date = datetime.now().date()

    def _reset_daily_stats_if_needed(self, current_balance: float):
        """Сбрасывает дневную статистику, если наступил новый день."""
        today = datetime.now().date()
        if today != self.last_check_date:
            logger.info(f"Наступил новый день ({today}). Сброс дневной статистики.")
            self.last_check_date = today
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.start_of_day_balance = current_balance

    def validate_trade(self, trader: BybitTrader, symbol: str) -> bool:
        """
        Основной метод валидации. Проверяет все правила риска перед сделкой.
        Возвращает True, если сделка разрешена, иначе False.
        """
        current_balance = trader.get_usdt_balance()
        if current_balance == 0:
            logger.error("Не удалось получить баланс. Торговля запрещена.")
            return False

        if self.initial_balance is None: self.initial_balance = current_balance
        if self.start_of_day_balance is None: self.start_of_day_balance = current_balance

        self._reset_daily_stats_if_needed(current_balance)

        if current_balance < MIN_BALANCE_USDT:
            logger.error(f"Риск-менеджер: Сделка отклонена. Баланс {current_balance:.2f} USDT < {MIN_BALANCE_USDT} USDT.")
            return False

        pnl_from_start = ((current_balance / self.initial_balance) - 1) * 100
        if pnl_from_start < EMERGENCY_STOP_PERCENT:
            logger.critical(f"!!! АВАРИЙНАЯ ОСТАНОВКА !!! Общий убыток {pnl_from_start:.2f}% > {EMERGENCY_STOP_PERCENT}%.")
            return False

        daily_pnl_percent = ((current_balance / self.start_of_day_balance) - 1) * 100
        if daily_pnl_percent < MAX_DAILY_LOSS_PERCENT:
            logger.warning(f"Риск-менеджер: Сделка отклонена. Дневной убыток {daily_pnl_percent:.2f}% > {MAX_DAILY_LOSS_PERCENT}%.")
            return False

        if self.trades_today >= MAX_DAILY_TRADES:
            logger.warning(f"Риск-менеджер: Сделка отклонена. Лимит сделок в день ({self.trades_today}/{MAX_DAILY_TRADES}).")
            return False

        open_position = trader.get_open_position(symbol)
        if open_position and float(open_position['size']) > 0:
            logger.warning(f"Риск-менеджер: Сделка отклонена. Уже есть открытая позиция по {symbol}.")
            return False

        logger.info("✅ Риск-менеджер: Все проверки пройдены. Сделка разрешена.")
        return True
    
    def record_trade(self):
        """Увеличивает счетчик сделок после успешного размещения ордера."""
        self.trades_today += 1
        logger.info(f"Сделка записана. Сделок сегодня: {self.trades_today}/{MAX_DAILY_TRADES}.")