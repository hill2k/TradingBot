import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from .logger import logger

class ReportGenerator:
    """
    Класс для генерации ежедневных и ежемесячных отчетов и графиков по торгам.
    """
    LOG_DIR = os.path.join("logs", "trading_logs")
    REPORT_DIR = os.path.join("logs", "reports")

    def __init__(self):
        os.makedirs(self.REPORT_DIR, exist_ok=True)
        sns.set_theme(style="darkgrid")

    def _generate_pnl_graph(self, df: pd.DataFrame, period: str, filename: str):
        """Создает и сохраняет график кумулятивного PnL."""
        if df.empty:
            return
        
        df['cumulative_pnl'] = df['pnl_usd'].cumsum()
        
        plt.figure(figsize=(12, 6))
        plot = sns.lineplot(x=df.index, y='cumulative_pnl', data=df, marker='o')
        plt.title(f'Кумулятивный PnL за {period}', fontsize=16)
        plt.xlabel('Сделка')
        plt.ylabel('Кумулятивный PnL (USD)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filepath = os.path.join(self.REPORT_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"График PnL сохранен: {filepath}")
        return filename

    def generate_daily_report(self, date: datetime = datetime.now()):
        """Генерирует полный отчет за указанный день."""
        date_str = date.strftime('%y%m%d')
        log_filename = f"{date_str}_trades.csv"
        log_filepath = os.path.join(self.LOG_DIR, log_filename)

        if not os.path.exists(log_filepath):
            logger.warning(f"Лог сделок за {date.strftime('%Y-%m-%d')} не найден. Отчет не создан.")
            return

        df = pd.read_csv(log_filepath)
        
        # Расчет метрик
        total_trades = len(df)
        wins = df[df['status'] == 'WIN'].shape[0]
        losses = df[df['status'] == 'LOSS'].shape[0]
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
        total_pnl_usd = df['pnl_usd'].sum()
        
        avg_win = df[df['status'] == 'WIN']['pnl_usd'].mean() if wins > 0 else 0
        avg_loss = df[df['status'] == 'LOSS']['pnl_usd'].mean() if losses > 0 else 0
        
        # Генерация графика
        graph_filename = f"{date_str}_daily_pnl.png"
        self._generate_pnl_graph(df, f"день ({date.strftime('%Y-%m-%d')})", graph_filename)
        
        # Формирование отчета
        report_md = f"""
# 📈 Ежедневный отчет по торгам за {date.strftime('%Y-%m-%d')}

## Основные метрики
- **Всего сделок**: {total_trades}
- **Прибыльные сделки**: {wins}
- **Убыточные сделки**: {losses}
- **Win Rate**: **{win_rate:.2f}%**
- **Итоговый PnL**: **{total_pnl_usd:+.2f} USD**
- **Средняя прибыльная сделка**: {avg_win:+.2f} USD
- **Средняя убыточная сделка**: {avg_loss:+.2f} USD

## График кумулятивного PnL за день
![Daily PnL](./{graph_filename})

## Детализация сделок
{df.to_markdown(index=False)}
"""
        report_filename = f"{date_str}_report.md"
        report_filepath = os.path.join(self.REPORT_DIR, report_filename)
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write(report_md)
            
        logger.info(f"Ежедневный отчет сгенерирован: {report_filepath}")

    # TODO: Добавить метод для генерации месячного отчета,
    # который будет собирать данные из всех дневных логов за месяц.