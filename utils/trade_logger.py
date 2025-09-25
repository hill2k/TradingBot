import os
import pandas as pd
from datetime import datetime, timedelta

from .logger import logger

class TradeLogger:
    """
    Класс для логирования результатов сделок в CSV файлы и управления ими.
    """
    LOG_DIR = os.path.join("logs", "trading_logs")
    RETENTION_DAYS = 90

    def __init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        self._cleanup_old_logs()

    def _get_log_filepath(self) -> str:
        """Возвращает путь к лог-файлу для текущего дня."""
        filename = f"{datetime.now().strftime('%y%m%d')}_trades.csv"
        return os.path.join(self.LOG_DIR, filename)

    def log_trade(self, trade_result: dict):
        """
        Записывает результат закрытой сделки в CSV файл.

        :param trade_result: Словарь с данными о сделке.
                             Ожидаемые ключи: 'timestamp', 'symbol', 'side', 
                             'qty', 'avgEntryPrice', 'avgExitPrice', 
                             'closedPnl', 'leverage'.
        """
        filepath = self._get_log_filepath()
        
        # Определяем статус сделки
        pnl = float(trade_result['closedPnl'])
        status = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"

        log_entry = {
            "timestamp": trade_result['updatedTime'],
            "symbol": trade_result['symbol'],
            "side": trade_result['side'],
            "qty": float(trade_result['qty']),
            "entry_price": float(trade_result['avgEntryPrice']),
            "exit_price": float(trade_result['avgExitPrice']),
            "pnl_usd": pnl,
            "status": status,
            "leverage": float(trade_result.get('leverage', 1))
        }

        try:
            # Создаем DataFrame из одной записи
            df_new = pd.DataFrame([log_entry])
            
            # Если файл не существует, создаем его с заголовками
            if not os.path.exists(filepath):
                df_new.to_csv(filepath, index=False)
                logger.info(f"Создан новый лог сделок: {filepath}")
            else:
                # Иначе дописываем в конец без заголовков
                df_new.to_csv(filepath, mode='a', header=False, index=False)
            
            logger.info(f"Сделка записана в лог: PnL={pnl:.2f} USD, Статус={status}")

        except Exception as e:
            logger.error(f"Не удалось записать сделку в лог: {e}", exc_info=True)

    def _cleanup_old_logs(self):
        """Удаляет лог-файлы сделок старше 90 дней."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.RETENTION_DAYS)
            for filename in os.listdir(self.LOG_DIR):
                if filename.endswith(".csv"):
                    try:
                        file_date = datetime.strptime(filename.split('_')[0], "%y%m%d")
                        if file_date < cutoff_date:
                            os.remove(os.path.join(self.LOG_DIR, filename))
                            logger.info(f"Удален старый лог сделок: {filename}")
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            logger.error(f"Ошибка при очистке старых логов сделок: {e}")