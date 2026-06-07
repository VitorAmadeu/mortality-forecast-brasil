"""
Módulo responsável pelos modelos de previsão:

- SARIMA
- Holt-Winters
- LSTM (Adam)
- LSTM (AdamW)

Modelos organizados para uso modular e produção.

Nota: modelo_sarima e modelo_holt_winters retornam uma tupla
(forecast, fit_result) para permitir extração de AIC/BIC.

Autor: Vitor Hugo Amadeu da Silva
"""

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam, AdamW


# =====================================================
# SARIMA
# =====================================================

def modelo_sarima(train, test,
                  order=(1, 1, 1),
                  seasonal_order=(1, 1, 1, 12)):
    """
    Ajusta SARIMA e retorna previsão + objeto fit.

    Retorna
    -------
    forecast : pd.Series com as previsões no índice de test
    fit      : objeto SARIMAXResults (contém .aic e .bic)
    """
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

    return forecast, fit


# =====================================================
# HOLT-WINTERS
# =====================================================

def modelo_holt_winters(train, test,
                        trend='add',
                        seasonal='mul',
                        seasonal_periods=12):
    """
    Ajusta Holt-Winters e retorna previsão + objeto fit.

    Retorna
    -------
    forecast : pd.Series com as previsões no índice de test
    fit      : objeto HoltWintersResultsWrapper (contém .aic e .bic)
    """
    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )

    fit = model.fit()

    forecast = fit.forecast(steps=len(test))
    forecast = pd.Series(forecast.values, index=test.index)

    return forecast, fit


# =====================================================
# CONSTRUTOR DE LSTM
# =====================================================

def construir_lstm(input_shape,
                   unidades_lstm=None,
                   unidades_dense=None,
                   optimizer='adam',
                   learning_rate=0.001,
                   weight_decay=0.0):
    """
    Constrói e compila um modelo LSTM sequencial.

    Parâmetros
    ----------
    input_shape    : tuple (seq_length, n_features)
    unidades_lstm  : lista com nº de unidades por camada LSTM
    unidades_dense : lista com nº de unidades por camada Dense oculta
    optimizer      : 'adam' ou 'adamw'
    learning_rate  : taxa de aprendizado
    weight_decay   : regularização L2 (somente AdamW)
    """
    # Defaults mutáveis como None para evitar bug de argumento padrão
    if unidades_lstm is None:
        unidades_lstm = [64, 32]
    if unidades_dense is None:
        unidades_dense = [16]

    model = Sequential()
    model.add(Input(shape=input_shape))

    # Camadas LSTM
    for i, units in enumerate(unidades_lstm):
        return_seq = (i < len(unidades_lstm) - 1)
        model.add(LSTM(units, return_sequences=return_seq))

    # Camadas Dense ocultas
    for units in unidades_dense:
        model.add(Dense(units, activation='relu'))

    # Saída
    model.add(Dense(1))

    # Otimizador
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'adamw':
        opt = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer deve ser 'adam' ou 'adamw'")

    model.compile(optimizer=opt, loss='mean_squared_error')

    return model


# =====================================================
# TREINAMENTO LSTM
# =====================================================

def treinar_lstm(model,
                 X_train, y_train,
                 X_test,  y_test,
                 epochs=60,
                 batch_size=32,
                 verbose=0):
    """
    Treina o modelo LSTM e retorna o histórico de treino.
    """
    history = model.fit(
        X_train, y_train,
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
    Gera previsões do LSTM e reconstrói o DataFrame com o índice correto.

    Retorna
    -------
    df_resultado : DataFrame com colunas DATA (índice), REAL, PREVISAO
    """
    pred = model.predict(X_test)

    pred_real   = scaler.inverse_transform(pred)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Recriar índice temporal correto
    total_obs   = len(serie_index)
    total_seq   = total_obs - seq_length
    split_index = int(total_seq * proporcao_treino)

    idx_test = serie_index[seq_length + split_index:]
    idx_test = idx_test[:len(pred_real)]

    df_resultado = pd.DataFrame({
        "DATA":     idx_test,
        "REAL":     y_test_real.flatten(),
        "PREVISAO": pred_real.flatten()
    })

    df_resultado.set_index("DATA", inplace=True)

    return df_resultado