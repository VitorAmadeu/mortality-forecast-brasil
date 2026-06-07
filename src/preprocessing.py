"""
Módulo responsável pelo pré-processamento da série temporal.

Inclui:
- Validação da série
- Tratamento de valores nulos
- Garantia de frequência mensal
- Normalização (MinMaxScaler)
- Criação de sequências para LSTM
- Split temporal sem shuffle

Autor: Vitor Hugo Amadeu da Silva
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
    - Tenha índice DatetimeIndex
    - Tenha frequência mensal (MS)
    - Não esteja vazia
    """

    if not isinstance(serie, pd.Series):
        raise TypeError("A entrada deve ser um pandas Series.")

    if not isinstance(serie.index, pd.DatetimeIndex):
        raise TypeError("O índice da série deve ser DatetimeIndex.")

    if serie.empty:
        raise ValueError("A série está vazia.")

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
    Valores no início/fim (não alcançados pela interpolação)
    são preenchidos com forward/backward fill como fallback.
    """

    if serie.isnull().sum() > 0:
        serie = serie.interpolate(method="linear")
        # fallback para NaNs nas bordas (interpolação não cobre extremos)
        serie = serie.ffill().bfill()

    return serie


# =====================================================
# NORMALIZAÇÃO
# =====================================================

def normalizar_serie(serie: pd.Series):
    """
    Aplica MinMaxScaler no intervalo (0, 1).

    Retorna
    -------
    scaler       : MinMaxScaler treinado (necessário para inverse_transform)
    serie_scaled : np.ndarray com shape (n, 1)
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
    Converte array escalado em janelas deslizantes para LSTM.

    Parâmetros
    ----------
    data       : np.ndarray com shape (n, 1)
    seq_length : tamanho da janela de entrada

    Retorna
    -------
    X : np.ndarray com shape (n - seq_length, seq_length, 1)
    y : np.ndarray com shape (n - seq_length,)
    """

    if len(data) <= seq_length:
        raise ValueError(
            f"A série tem {len(data)} observações, mas seq_length={seq_length}. "
            "São necessárias pelo menos seq_length + 1 observações."
        )

    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])       # shape (seq_length, 1)
        y.append(data[i + seq_length])          # escalar

    # y.flatten() garante shape (n,) em vez de (n, 1)
    return np.array(X), np.array(y).flatten()


# =====================================================
# SPLIT TEMPORAL (SEM VAZAMENTO)
# =====================================================

def split_temporal(X, y, proporcao_treino=0.8):
    """
    Divide dados respeitando a ordem temporal (sem shuffle).

    Retorna
    -------
    X_train, X_test, y_train, y_test
    """

    if not (0 < proporcao_treino < 1):
        raise ValueError("proporcao_treino deve estar entre 0 e 1 (exclusive).")

    split_index = int(len(X) * proporcao_treino)

    if split_index == 0 or split_index == len(X):
        raise ValueError(
            f"Split resultou em conjunto vazio. "
            f"Verifique o tamanho da série e proporcao_treino={proporcao_treino}."
        )

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test


# =====================================================
# PIPELINE COMPLETO PARA LSTM
# =====================================================

def preparar_dados_lstm(serie: pd.Series, seq_length: int = 12,
                        proporcao_treino: float = 0.8):
    """
    Executa o pipeline completo de pré-processamento para LSTM:
      1. Validação da série
      2. Tratamento de nulos
      3. Normalização MinMax
      4. Criação de sequências deslizantes
      5. Split temporal

    Parâmetros
    ----------
    serie            : pd.Series com índice DatetimeIndex mensal
    seq_length       : tamanho da janela de entrada (padrão 12)
    proporcao_treino : fração usada para treino (padrão 0.8)

    Retorna
    -------
    scaler, X_train, X_test, y_train, y_test
    """

    serie = validar_serie(serie)
    serie = tratar_nulos(serie)

    scaler, serie_scaled = normalizar_serie(serie)

    X, y = criar_sequencias(serie_scaled, seq_length)

    X_train, X_test, y_train, y_test = split_temporal(
        X, y, proporcao_treino=proporcao_treino
    )

    return scaler, X_train, X_test, y_train, y_test