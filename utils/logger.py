import logging
import sys
import os
from datetime import datetime, timedelta

LOGS_DIR = "logs"
LOG_FILENAME_FORMAT = "%y%m%d_trading_bot.log"
LOG_RETENTION_DAYS = 30

def cleanup_old_logs():
    """Сканирует папку logs и удаляет файлы старше LOG_RETENTION_DAYS."""
    if not os.path.isdir(LOGS_DIR):
        return

    try:
        now = datetime.now()
        retention_limit = timedelta(days=LOG_RETENTION_DAYS)
        
        for filename in os.listdir(LOGS_DIR):
            if filename.endswith(".log"):
                try:
                    # Извлекаем дату из имени файла
                    file_date_str = filename.split('_')[0]
                    file_date = datetime.strptime(file_date_str, "%y%m%d")
                    
                    if now - file_date > retention_limit:
                        file_path = os.path.join(LOGS_DIR, filename)
                        os.remove(file_path)
                        logging.getLogger("TradingBot").info(f"Удален старый лог-файл: {filename}")
                except (ValueError, IndexError):
                    # Пропускаем файлы с некорректным форматом имени
                    continue
    except Exception as e:
        logging.getLogger("TradingBot").error(f"Ошибка при очистке старых логов: {e}")


def setup_logger():
    """
    Настраивает логгер для вывода в консоль и в суточный файл
    с автоматической ротацией и очисткой.
    """
    logger = logging.getLogger("TradingBot")
    if logger.hasHandlers():
        # Если логгер уже настроен, просто возвращаем его
        return logger

    logger.setLevel(logging.INFO)

    # 1. Создаем папку для логов, если ее нет
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # 2. Очищаем старые логи
    cleanup_old_logs()

    # 3. Настраиваем форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 4. Настраиваем консольный хендлер
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 5. Настраиваем файловый хендлер с динамическим именем файла
    log_file_path = os.path.join(LOGS_DIR, datetime.now().strftime(LOG_FILENAME_FORMAT))
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Логирование настроено. Запись ведется в файл: {log_file_path}")
    return logger

# Инициализируем логгер при импорте модуля
logger = setup_logger()