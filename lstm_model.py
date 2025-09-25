import torch
import torch.nn as nn
from config import DEVICE
from utils.logger import logger

class EnhancedAttentionLSTM(nn.Module):
    """
    Гибридная модель LSTM с механизмом Multi-head Attention и двумя выходами:
    1. Прогноз процентного изменения цены.
    2. Уверенность модели в прогнозе (от 0 до 1).
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, num_heads: int = 8):
        super(EnhancedAttentionLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.input_norm = nn.LayerNorm(input_size)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.lstm_dropout = nn.Dropout(dropout)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.residual_norm = nn.LayerNorm(hidden_size)
        
        # Общая "голова" для извлечения признаков
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Выход №1: Прогноз изменения цены
        self.prediction_head = nn.Linear(hidden_size // 2, 1)

        # Выход №2: Уверенность (с активацией Sigmoid для получения значения от 0 до 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        logger.info(f"Модель EnhancedAttentionLSTM инициализирована с {input_size} признаками и ДВУМЯ выходами.")
        logger.info(f"hidden_size={hidden_size}, num_layers={num_layers}, num_heads={num_heads}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход. Возвращает кортеж: (прогноз, уверенность).
        """
        residual = x if self.input_size == self.hidden_size else None
        x_norm = self.input_norm(x)
        
        lstm_out, _ = self.lstm(x_norm)
        lstm_out = self.lstm_dropout(self.lstm_norm(lstm_out))

        if residual is not None:
            lstm_out = lstm_out + residual

        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = self.residual_norm(attn_output + lstm_out)
        
        last_step_output = attn_output[:, -1, :]
        
        # Пропускаем через общую "голову"
        shared_features = self.shared_head(last_step_output)
        
        # Получаем два разных выхода
        prediction = self.prediction_head(shared_features)
        confidence = self.confidence_head(shared_features)
        
        # Убираем лишние размерности
        return prediction.squeeze(-1), confidence.squeeze(-1)
