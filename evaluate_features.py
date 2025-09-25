import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from utils.logger import logger
from data_fetcher import BybitDataFetcher
from preprocess import FeatureEngine
from config import SYMBOL, PRIMARY_TIMEFRAME, SECONDARY_SYMBOL
from indicators import add_indicators

# --- Константы ---
EVAL_DATA_LIMIT = 8000 # Увеличим данные для более качественной оценки
IMPORTANCE_PLOT_PATH = "feature_importance_ensemble.png"
IMPORTANCE_CSV_PATH = "feature_importance_ensemble.csv"
TOP_N_FEATURES = 40

def prepare_evaluation_data():
    """Подготовка данных с учетом BTC для оценки признаков."""
    logger.info("--- Загрузка данных для оценки признаков ---")
    fetcher = BybitDataFetcher()
    
    raw_data = fetcher.fetch_historical_data(SYMBOL, PRIMARY_TIMEFRAME, EVAL_DATA_LIMIT)
    if raw_data is None:
        raise ConnectionError("Не удалось загрузить основные данные для оценки.")

    btc_data = fetcher.fetch_historical_data(SECONDARY_SYMBOL, "60", EVAL_DATA_LIMIT)
    if btc_data is None:
        raise ConnectionError("Не удалось загрузить данные по BTC для оценки.")

    logger.info("--- Создание признаков (включая BTC) ---")
    feature_engine = FeatureEngine()
    
    btc_data_with_ta = add_indicators(btc_data)
    # Передаем is_training=True, чтобы FeatureEngine не фильтровал признаки, а вернул все
    df_with_features = feature_engine.create_features(raw_data, df_secondary=btc_data_with_ta, is_training=True)
    
    df_with_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Горизонт прогноза ---
    # Целевая переменная должна соответствовать тому, что мы используем в обучении (4 часа)
    prediction_horizon = 4 
    target = df_with_features['close'].pct_change(periods=prediction_horizon).shift(-prediction_horizon) * 100
    
    features = df_with_features.drop(columns=[col for col in ['open', 'high', 'low', 'close', 'volume', 'turnover'] if col in df_with_features.columns], errors='ignore')

    combined = pd.concat([features, target.rename('target')], axis=1)
    combined.dropna(inplace=True)
    
    final_features = combined.drop(columns=['target'])
    final_target = combined['target']
    
    return final_features, final_target

def run_feature_evaluation():
    """
    Обучает ансамбль моделей для оценки важности признаков и сохраняет результат.
    """
    logger.info("--- 🚀 Начало ансамблевой оценки важности признаков ---")

    try:
        features_df, target = prepare_evaluation_data()
        feature_names = features_df.columns.tolist()
        logger.info(f"Данные успешно подготовлены. Форма признаков: {features_df.shape}")

        X_train, _, y_train, _ = train_test_split(features_df, target, test_size=0.1, shuffle=False)

        logger.info("Обучение LightGBM...")
        lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        lgbm.fit(X_train, y_train)
        lgbm_importance = lgbm.feature_importances_
        logger.info("LightGBM обучен.")

        logger.info("Обучение XGBoost...")
        xgbr = xgb.XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')
        xgbr.fit(X_train, y_train)
        xgb_importance = xgbr.feature_importances_
        logger.info("XGBoost обучен.")

        logger.info("Обучение CatBoost...")
        cbr = cb.CatBoostRegressor(random_state=42, verbose=0, thread_count=-1)
        cbr.fit(X_train, y_train)
        cb_importance = cbr.feature_importances_
        logger.info("CatBoost обучен.")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'lgbm': lgbm_importance,
            'xgb': xgb_importance,
            'catboost': cb_importance
        })

        scaler = MinMaxScaler()
        importance_df[['lgbm_norm', 'xgb_norm', 'catboost_norm']] = scaler.fit_transform(
            importance_df[['lgbm', 'xgb', 'catboost']]
        )

        importance_df['ensemble_importance'] = importance_df[
            ['lgbm_norm', 'xgb_norm', 'catboost_norm']
        ].mean(axis=1)

        importance_df = importance_df.sort_values('ensemble_importance', ascending=False).reset_index(drop=True)
        
        importance_df.to_csv(IMPORTANCE_CSV_PATH, index=False)
        logger.info(f"Полный отчет по важности сохранен в {IMPORTANCE_CSV_PATH}")

        print("\n--- Топ-20 самых важных признаков (ансамблевая оценка) ---")
        print(importance_df[['feature', 'ensemble_importance']].head(20).to_string(index=False))
        print("----------------------------------------------------------\n")

        logger.info(f"Создание графика для топ-{TOP_N_FEATURES} признаков...")
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, max(10, TOP_N_FEATURES / 2)))
        
        top_features = importance_df.head(TOP_N_FEATURES)
        
        sns.barplot(x='ensemble_importance', y='feature', data=top_features, palette='viridis', ax=ax)
        
        ax.set_title(f'Топ-{TOP_N_FEATURES} признаков (ансамбль LGBM, XGB, CatBoost)', fontsize=16)
        ax.set_xlabel('Нормализованная важность (Ensemble Importance)', fontsize=12)
        ax.set_ylabel('Признак', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(IMPORTANCE_PLOT_PATH)
        logger.info(f"График важности признаков сохранен в {IMPORTANCE_PLOT_PATH}")

    except Exception as e:
        logger.error(f"Произошла непредвиденная ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    run_feature_evaluation()