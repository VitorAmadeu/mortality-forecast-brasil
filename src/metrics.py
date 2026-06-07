"""
Módulo responsável pelas métricas de avaliação dos modelos.

Inclui:
- MSE, RMSE, MAE, MAPE (seguro), R²
- AIC / BIC (para modelos statsmodels — SARIMA, Holt-Winters)
- Função consolidada de avaliação
- Geração de DataFrame comparativo
- Salvamento automático em CSV

Autor: Vitor Hugo Amadeu da Silva
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# =====================================================
# MÉTRICAS INDIVIDUAIS
# =====================================================

def calcular_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def calcular_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calcular_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def calcular_mape(y_true, y_pred):
    """
    MAPE seguro contra divisão por zero.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0

    if np.sum(mask) == 0:
        return np.nan

    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100


def calcular_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def calcular_aic_bic(fit_result):
    """
    Extrai AIC e BIC de um objeto de fit do statsmodels.
    Compatível com SARIMAX e ExponentialSmoothing.

    Retorna (aic, bic) ou (np.nan, np.nan) se não disponível.
    """
    aic = getattr(fit_result, "aic", np.nan)
    bic = getattr(fit_result, "bic", np.nan)
    return aic, bic


# =====================================================
# AVALIAÇÃO COMPLETA — SEM AIC/BIC (LSTM)
# =====================================================

def avaliar_modelo(y_true, y_pred):
    """
    Retorna métricas de erro em dicionário.
    Usado para modelos que não possuem AIC/BIC (ex: LSTM).
    """
    return {
        "MSE":      calcular_mse(y_true, y_pred),
        "RMSE":     calcular_rmse(y_true, y_pred),
        "MAE":      calcular_mae(y_true, y_pred),
        "MAPE (%)": calcular_mape(y_true, y_pred),
        "R2":       calcular_r2(y_true, y_pred),
        "AIC":      np.nan,
        "BIC":      np.nan,
    }


def avaliar_modelo_classico(y_true, y_pred, fit_result):
    """
    Retorna métricas de erro + AIC/BIC para modelos statsmodels.
    Usado para SARIMA e Holt-Winters.

    Parâmetros
    ----------
    fit_result : objeto retornado por model.fit() do statsmodels
    """
    metricas = avaliar_modelo(y_true, y_pred)
    aic, bic = calcular_aic_bic(fit_result)
    metricas["AIC"] = aic
    metricas["BIC"] = bic
    return metricas


# =====================================================
# GERAR DATAFRAME DE MÉTRICAS
# =====================================================

def gerar_df_metricas(nome_modelo, y_true, y_pred, fit_result=None):
    """
    Gera DataFrame de uma linha para um modelo.

    Parâmetros
    ----------
    fit_result : opcional — objeto fit do statsmodels para AIC/BIC.
                 Se None, AIC e BIC ficam como NaN.
    """
    if fit_result is not None:
        metricas = avaliar_modelo_classico(y_true, y_pred, fit_result)
    else:
        metricas = avaliar_modelo(y_true, y_pred)

    metricas["Modelo"] = nome_modelo

    return pd.DataFrame([metricas])


# =====================================================
# CONSOLIDAR MÚLTIPLOS MODELOS
# =====================================================

def consolidar_metricas(lista_metricas):
    """
    Recebe lista de DataFrames e consolida.
    Ordena pelo menor MSE.
    Colunas AIC/BIC ficam NaN para modelos sem suporte.
    """
    df_final = pd.concat(lista_metricas, ignore_index=True)

    colunas_ordem = ["Modelo", "MSE", "RMSE", "MAE", "MAPE (%)", "R2", "AIC", "BIC"]

    # Garante que todas as colunas existam (tolerância a versões antigas)
    for col in colunas_ordem:
        if col not in df_final.columns:
            df_final[col] = np.nan

    df_final = df_final[colunas_ordem]

    return df_final.sort_values(by="MSE").reset_index(drop=True)


# =====================================================
# SALVAR MÉTRICAS
# =====================================================

def salvar_metricas(df_metricas, path="../Data/processed/metricas_modelos.csv"):
    """
    Salva métricas consolidadas em CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_metricas.to_csv(path, index=False)
    print(f"✅ Métricas salvas em: {path}")