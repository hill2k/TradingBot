import requests
import pandas as pd
from datetime import datetime
import time

from utils.logger import logger
import config

class BybitDataFetcher:
    """
    Класс для загрузки исторических данных с биржи Bybit.
    Реализует корректную постраничную загрузку (пагинацию).
    """
    BASE_URL = "https://api.bybit.com"

    def __init__(self):
        # Проверка соединения при инициализации
        try:
            response = requests.get(f"{self.BASE_URL}/v5/market/time")
            response.raise_for_status()
            logger.info("✅ Успешное соединение с API Bybit.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Не удалось подключиться к API Bybit: {e}")
            raise ConnectionError("Не удалось подключиться к Bybit API.")

    def fetch_historical_data(self, symbol: str, interval: str, total_limit: int = 2000) -> pd.DataFrame | None:
        """
        Загружает исторические данные для указанного символа, обрабатывая пагинацию.
        
        :param symbol: Торговый символ (например, 'XRPUSDT').
        :param interval: Таймфрейм (например, '60' для 1 часа).
        :param total_limit: Общее количество свечей, которое нужно загрузить.
        :return: DataFrame с данными или None в случае ошибки.
        """
        endpoint = "/v5/market/kline"
        url = self.BASE_URL + endpoint
        
        all_data = []
        end_timestamp = None
        limit_per_request = 1000 # Максимальный лимит Bybit на один запрос

        logger.info(f"Начало загрузки {total_limit} свечей для {symbol} ({interval}m) в несколько этапов...")

        while len(all_data) < total_limit:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit_per_request, total_limit - len(all_data))
            }
            if end_timestamp:
                params["end"] = end_timestamp

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if data.get('retCode') != 0:
                    logger.error(f"Ошибка API Bybit: {data.get('retMsg')}")
                    return None

                result_list = data.get('result', {}).get('list', [])
                if not result_list:
                    logger.warning("API не вернул новых данных. Загрузка завершена.")
                    break

                all_data.extend(result_list)
                
                # Обновляем временную метку для следующего запроса (чтобы получить более старые данные)
                oldest_timestamp_in_batch = int(result_list[-1][0])
                end_timestamp = oldest_timestamp_in_batch - 1 # -1 мс, чтобы не дублировать свечу

                logger.info(f"Загружено {len(all_data)} / {total_limit} свечей...")
                time.sleep(0.2) # Небольшая задержка, чтобы не превышать лимиты API

            except requests.exceptions.RequestException as e:
                logger.error(f"Ошибка при запросе к API Bybit: {e}")
                return None
            except Exception as e:
                logger.error(f"Непредвиденная ошибка при обработке данных: {e}", exc_info=True)
                return None

        if not all_data:
            logger.error("Не удалось загрузить данные.")
            return None

        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])

        df = df.iloc[::-1].reset_index(drop=True)

        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ ---
        # 1. Сначала преобразуем в числовой тип, который может вместить timestamp (int64)
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        # 2. Затем преобразуем в datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
            df[col] = pd.to_numeric(df[col])

        df.set_index('timestamp', inplace=True)
        
        df = df[~df.index.duplicated(keep='first')]
        
        logger.info(f"✅ Загрузка завершена. Итоговый размер DataFrame: {df.shape[0]} строк.")
        
        return df

if __name__ == '__main__':
    # Пример использования
    fetcher = BybitDataFetcher()
    df_test = fetcher.fetch_historical_data(
        symbol=config.SYMBOL, 
        interval=config.PRIMARY_TIMEFRAME, 
        total_limit=8000
    )
    if df_test is not None:
        print("Пример загруженных данных (первые 5 строк):")
        print(df_test.head())
        print("\nПример загруженных данных (последние 5 строк):")
        print(df_test.tail())
        print(f"\nВсего загружено: {len(df_test)} свечей.")
