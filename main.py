import time
from datetime import datetime, date, timedelta

from utils.logger import logger
from utils.trade_logger import TradeLogger
from utils.reporting import ReportGenerator
import config

from data_fetcher import BybitDataFetcher
from model import HybridModel
from strategy import TradingStrategy
from trading import BybitTrader
from risk_manager import RiskManager

def run_bot():
    """Главная функция, запускающая торгового бота."""
    logger.info("🚀 Запуск торгового бота GARCH+LSTM с логикой Smart Money v3 (с отчетностью)...")

    try:
        data_fetcher = BybitDataFetcher()
        hybrid_model = HybridModel(
            sequence_length=24,
            lstm_model_path='trading_bot_model.pth',
            scaler_path='scaler.pkl',
            features_path='feature_names.pkl'
        )
        strategy = TradingStrategy()
        trader = BybitTrader()
        risk_manager = RiskManager()
        trade_logger = TradeLogger()
        report_generator = ReportGenerator()
    except Exception as e:
        logger.critical(f"Критическая ошибка при инициализации: {e}", exc_info=True)
        return

    logger.info("✅ Все компоненты успешно инициализированы.")

    is_position_open = False
    last_day_reported = date.today() - timedelta(days=1)

    while True:
        try:
            now = datetime.now()
            
            if now.date() > last_day_reported:
                logger.info(f"Наступил новый день. Генерация отчета за {last_day_reported}...")
                report_generator.generate_daily_report(date=last_day_reported)
                last_day_reported = now.date()

            current_position = trader.get_open_position(config.SYMBOL)
            
            if is_position_open and current_position is None:
                logger.info("Позиция была закрыта. Запрос результата сделки...")
                closed_trades = trader.get_closed_pnl(config.SYMBOL, limit=1)
                if closed_trades:
                    trade_logger.log_trade(closed_trades[0])
                is_position_open = False
            
            elif not is_position_open and current_position is not None:
                is_position_open = True
            
            if current_position:
                logger.info("Открыта позиция, пропускаем цикл генерации сигналов.")
                time.sleep(60)
                continue

            logger.info(f"--- 🏁 Начало нового торгового цикла: {now.strftime('%Y-%m-%d %H:%M:%S')} ---")

            market_data_primary = data_fetcher.fetch_historical_data(
                symbol=config.SYMBOL,
                interval=config.PRIMARY_TIMEFRAME,
                total_limit=config.PRIMARY_DATA_LIMIT
            )
            
            market_data_secondary = data_fetcher.fetch_historical_data(
                symbol=config.SECONDARY_SYMBOL,
                interval="60",
                total_limit=config.PRIMARY_DATA_LIMIT
            )

            if market_data_primary is None or len(market_data_primary) < config.MIN_DATA_LENGTH:
                logger.warning("Недостаточно основных данных, пропускаем цикл.")
                time.sleep(60)
                continue
            
            current_price = market_data_primary['close'].iloc[-1]
            prediction = hybrid_model.predict(market_data_primary, df_secondary=market_data_secondary)
            
            if prediction.get("success"):
                signal = strategy.generate_signal(prediction, current_price)
                if signal:
                    logger.info(f"🔥 Получен торговый сигнал: {signal['side']} @ {current_price:.5f}")
                    if risk_manager.validate_trade(trader, config.SYMBOL):
                        balance = trader.get_usdt_balance()
                        
                        # --- ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ V5.0 ---
                        position_size = trader.calculate_position_size(
                            balance=balance,
                            entry_price=signal['entry_price'],
                            sl_price=signal['sl_price'],
                            risk_pct=config.RISK_PER_TRADE_PERCENT
                        )
                        # ----------------------------------

                        if position_size > 0:
                            trade_successful = trader.place_trade(signal, position_size)
                            if trade_successful:
                                risk_manager.record_trade()
                                is_position_open = True
                                logger.info("🎉 СДЕЛКА УСПЕШНО ИСПОЛНЕНА!")

            interval = config.TRADING_INTERVAL_MINUTES
            logger.info(f"--- 🔚 Торговый цикл завершен. Пауза на {interval} минут. ---")
            time.sleep(interval * 60)

        except KeyboardInterrupt:
            logger.info("Получен сигнал на остановку (Ctrl+C). Завершение работы...")
            break
        except Exception as e:
            logger.error(f"Произошла непредвиденная ошибка в главном цикле: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    run_bot()