"""
Módulo responsável pelo pré-processamento da série temporal.

Inclui:
- Validação da série
- Tratamento de valores nulos
- Garantia de frequência mensal
- Normalização (MinMaxScaler)
- Criação de sequências para LSTM
- Split temporal sem shuffle
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# =====================================================
# VALIDAÇÃO DA SÉRIE
# =====================================================

def validar_serie(serie: pd.Series) -> pd.Series:
    """
    Garante que a série:
    - Seja pandas Series
    - Tenha índice datetime
    - Tenha frequência mensal
    """

    if not isinstance(serie, pd.Series):
        raise TypeError("A entrada deve ser um pandas Series.")

    if not isinstance(serie.index, pd.DatetimeIndex):
        raise TypeError("O índice da série deve ser DatetimeIndex.")

    # Garantir frequência mensal
    if serie.index.freq is None:
        serie = serie.asfreq("MS")

    return serie


# =====================================================
# TRATAMENTO DE VALORES NULOS
# =====================================================

def tratar_nulos(serie: pd.Series) -> pd.Series:
    """
    Preenche valores ausentes usando interpolação linear.
    """

    if serie.isnull().sum() > 0:
        serie = serie.interpolate(method="linear")

    return serie


# =====================================================
# NORMALIZAÇÃO
# =====================================================

def normalizar_serie(serie: pd.Series):
    """
    Aplica MinMaxScaler (0,1).
    Retorna:
    - scaler treinado
    - série escalada
    """

    scaler = MinMaxScaler(feature_range=(0, 1))

    serie_scaled = scaler.fit_transform(
        serie.values.reshape(-1, 1)
    )

    return scaler, serie_scaled


# =====================================================
# CRIAÇÃO DE SEQUÊNCIAS (LSTM)
# =====================================================

def criar_sequencias(data: np.ndarray, seq_length: int):
    """
    Converte série escalada em janelas deslizantes.
    """

    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


# =====================================================
# SPLIT TEMPORAL (SEM VAZAMENTO)
# =====================================================

def split_temporal(X, y, proporcao_treino=0.8):
    """
    Divide dados respeitando ordem temporal.
    """

    split_index = int(len(X) * proporcao_treino)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


# =====================================================
# PIPELINE COMPLETO PARA LSTM
# =====================================================

def preparar_dados_lstm(serie: pd.Series, seq_length=12):
    """
    Executa pipeline completo:
    - Validação
    - Tratamento de nulos
    - Normalização
    - Criação de sequências
    - Split temporal

    Retorna:
    scaler, X_train, X_test, y_train, y_test
    """

    serie = validar_serie(serie)
    serie = tratar_nulos(serie)

    scaler, serie_scaled = normalizar_serie(serie)

    X, y = criar_sequencias(serie_scaled, seq_length)

    X_train, X_test, y_train, y_test = split_temporal(X, y)

    return scaler, X_train, X_test, y_train, y_test