# -*- coding: utf-8 -*-
"""
Módulo responsável pelos modelos de previsão:

- SARIMA
- Holt-Winters
- LSTM (Adam)
- LSTM (AdamW)

Modelos organizados para uso modular e produção.
"""

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW


# =====================================================
# SARIMA
# =====================================================

def modelo_sarima(train, test,
                  order=(1, 1, 1),
                  seasonal_order=(1, 1, 1, 12)):

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit = model.fit(disp=False)

    forecast = fit.forecast(steps=len(test))
    forecast = pd.Series(forecast.values, index=test.index)

    return forecast


# =====================================================
# HOLT-WINTERS
# =====================================================

def modelo_holt_winters(train, test,
                        trend='add',
                        seasonal='mul',
                        seasonal_periods=12):

    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )

    fit = model.fit()

    forecast = fit.forecast(steps=len(test))
    forecast = pd.Series(forecast.values, index=test.index)

    return forecast


# =====================================================
# CONSTRUTOR DE LSTM
# =====================================================

def construir_lstm(input_shape,
                   unidades_lstm=[64, 32],
                   unidades_dense=[16],
                   optimizer='adam',
                   learning_rate=0.001,
                   weight_decay=0.0):

    model = Sequential()
    model.add(Input(shape=input_shape))

    # Camadas LSTM
    for i, units in enumerate(unidades_lstm):
        return_seq = (i < len(unidades_lstm) - 1)
        model.add(LSTM(units, return_sequences=return_seq))

    # Camadas Dense
    for units in unidades_dense:
        model.add(Dense(units))

    model.add(Dense(1))

    # Otimizador
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'adamw':
        opt = AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError("Optimizer deve ser 'adam' ou 'adamw'")

    model.compile(
        optimizer=opt,
        loss='mean_squared_error'
    )

    return model


# =====================================================
# TREINAMENTO LSTM
# =====================================================

def treinar_lstm(model,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 epochs=60,
                 batch_size=32,
                 verbose=0):

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=verbose
    )

    return history


# =====================================================
# PREVISÃO LSTM
# =====================================================

def prever_lstm(model,
                X_test,
                scaler,
                y_test,
                serie_index,
                seq_length,
                proporcao_treino=0.8):
    """
    Gera previsões do modelo LSTM e retorna DataFrame com DATA, REAL, PREVISAO
    
    ✅ CORREÇÃO: Agora retorna DataFrame sem índice, com coluna DATA explícita
    """

    # Fazer previsões
    pred = model.predict(X_test, verbose=0)

    # Desnormalizar
    pred_real = scaler.inverse_transform(pred)
    y_test_real = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    # ✅ Recriar índice correto
    total_obs = len(serie_index)
    total_seq = total_obs - seq_length
    split_index = int(total_seq * proporcao_treino)

    idx_test = serie_index[seq_length + split_index:]
    idx_test = idx_test[:len(pred_real)]

    # ✅ CRIAR DATAFRAME SEM ÍNDICE
    df_resultado = pd.DataFrame({
        "DATA": idx_test,
        "REAL": y_test_real.flatten(),
        "PREVISAO": pred_real.flatten()
    })

    # ✅ NÃO USAR set_index - deixar DATA como coluna
    return df_resultado