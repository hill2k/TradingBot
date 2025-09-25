from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError

from utils.logger import logger
from config import (
    BYBIT_API_KEY, BYBIT_API_SECRET, USE_TESTNET, SYMBOL, 
    DEMO_MODE, LEVERAGE, MIN_ORDER_VALUE_USDT
)

class BybitTrader:
    def __init__(self):
        try:
            # Увеличиваем окно получения до 10 секунд для стабильности
            self.session = HTTP(
                testnet=USE_TESTNET,
                demo=DEMO_MODE,
                api_key=BYBIT_API_KEY,
                api_secret=BYBIT_API_SECRET,
                recv_window=10000 
            )
            logger.info("✅ Успешное соединение с API Bybit (v5) для торговли.")
            self._set_leverage()
        except Exception as e:
            logger.error(f"Не удалось инициализировать торговую сессию Bybit: {e}")
            raise ConnectionError("Не удалось подключиться к Bybit API для торговли.")

    def _set_leverage(self):
        try:
            logger.info(f"Установка плеча x{LEVERAGE} для {SYMBOL}...")
            self.session.set_leverage(
                category="linear",
                symbol=SYMBOL,
                buyLeverage=str(LEVERAGE),
                sellLeverage=str(LEVERAGE)
            )
            logger.info(f"✅ Плечо x{LEVERAGE} успешно установлено для {SYMBOL}.")
        except InvalidRequestError as e:
            if "110043" in str(e):
                logger.info(f"ℹ️ Плечо x{LEVERAGE} уже было установлено для {SYMBOL}. Изменения не требуются.")
            else:
                logger.error(f"❌ Ошибка установки плеча: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Исключение при установке плеча: {e}", exc_info=True)
            
    def get_usdt_balance(self) -> float:
        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if response['retCode'] == 0 and response['result']['list']:
                balance_info = response['result']['list'][0]
                wallet_balance_str = balance_info.get('totalEquity') or balance_info.get('walletBalance')
                if wallet_balance_str is None: return 0.0
                wallet_balance = float(wallet_balance_str)
                logger.info(f"Баланс кошелька (Equity): {wallet_balance:.2f} USDT")
                return wallet_balance
            return 0.0
        except Exception:
            return 0.0

    def get_open_position(self, symbol: str) -> dict | None:
        try:
            response = self.session.get_positions(category="linear", symbol=symbol)
            if response['retCode'] == 0 and float(response['result']['list'][0]['size']) > 0:
                return response['result']['list'][0]
        except Exception:
            pass
        return None

    # --- ВОЗВРАЩАЕМ ПРАВИЛЬНУЮ ЛОГИКУ РАСЧЕТА ПОЗИЦИИ ---
    def calculate_position_size(self, balance: float, entry_price: float, sl_price: float, risk_pct: float) -> int:
        """
        Рассчитывает размер позиции так, чтобы убыток при срабатывании SL не превышал risk_pct от баланса.
        """
        if entry_price <= 0 or balance <= 0 or sl_price <= 0:
            return 0

        # 1. Сумма, которой мы готовы рискнуть в USDT (например, 15 USDT)
        risk_amount_usd = balance * (risk_pct / 100)
        logger.info(f"Сумма риска ({risk_pct}% от баланса): {risk_amount_usd:.2f} USDT")

        # 2. Расстояние до стопа в долях (например, 0.005 для 0.5%)
        stop_loss_fraction = abs(entry_price - sl_price) / entry_price
        if stop_loss_fraction == 0:
            logger.error("Расстояние до Stop Loss равно нулю. Расчет невозможен.")
            return 0
        
        # 3. Общая стоимость позиции, которую нужно открыть
        # position_value * stop_loss_fraction = risk_amount_usd
        position_value_usd = risk_amount_usd / stop_loss_fraction
        
        # 4. Рассчитываем количество монет
        position_size_float = position_value_usd / entry_price
        position_size_int = int(position_size_float)
        
        # 5. Проверка на минимальный размер ордера
        final_position_value = position_size_int * entry_price
        if final_position_value < MIN_ORDER_VALUE_USDT:
            logger.warning(
                f"Итоговая стоимость позиции ({final_position_value:.2f} USDT) < минимальной ({MIN_ORDER_VALUE_USDT} USDT). "
                f"Сделка отменена."
            )
            return 0
            
        logger.info(
            f"Расчет размера позиции: Стоимость={final_position_value:.2f} USDT, "
            f"Кол-во={position_size_int} XRP, Маржа={final_position_value / LEVERAGE:.2f} USDT"
        )
        
        if position_size_int == 0:
            logger.warning("Размер позиции после округления равен нулю. Сделка отменена.")
            return 0
            
        return position_size_int

    def place_trade(self, signal: dict, position_size: int) -> bool:
        if not signal or position_size <= 0:
            return False
        try:
            mode_log = "ДЕМО-ОРДЕР" if DEMO_MODE else "РЕАЛЬНЫЙ ОРДЕР"
            logger.info(f"--- Попытка размещения ордера ({mode_log}) ---")
            logger.info(f"Параметры: {signal['side']} {position_size} {signal['symbol']} (плечо x{LEVERAGE} будет применено биржей)")
            
            response = self.session.place_order(
                category="linear",
                symbol=signal['symbol'],
                side=signal['side'],
                orderType="Market",
                qty=str(position_size),
                takeProfit=f"{signal['tp_price']:.5f}",
                stopLoss=f"{signal['sl_price']:.5f}",
                tpTriggerBy="LastPrice",
                slTriggerBy="LastPrice"
            )
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                logger.info(f"✅ {mode_log} успешно размещен! ID: {order_id}")
                return True
            else:
                logger.error(f"❌ Ошибка размещения ордера: {response['retMsg']} (Код: {response['retCode']})")
                return False
        except Exception as e:
            logger.error(f"Исключение при размещении ордера: {e}", exc_info=True)
            return False

    def get_closed_pnl(self, symbol: str, limit: int = 1) -> list:
        try:
            response = self.session.get_closed_pnl(category="linear", symbol=symbol, limit=limit)
            if response['retCode'] == 0:
                return response['result']['list']
        except Exception:
            pass
        return []