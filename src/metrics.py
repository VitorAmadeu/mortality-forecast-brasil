"""
Módulo responsável pelas métricas de avaliação dos modelos.

Inclui:
- MSE
- RMSE
- MAE
- MAPE (seguro)
- R²
- Função consolidada de avaliação
- Geração de DataFrame comparativo
- Salvamento automático em CSV
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

    # Evita divisão por zero
    mask = y_true != 0

    if np.sum(mask) == 0:
        return np.nan

    return np.mean(
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    ) * 100


def calcular_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


# =====================================================
# AVALIAÇÃO COMPLETA
# =====================================================

def avaliar_modelo(y_true, y_pred):
    """
    Retorna todas as métricas em dicionário.
    """

    return {
        "MSE": calcular_mse(y_true, y_pred),
        "RMSE": calcular_rmse(y_true, y_pred),
        "MAE": calcular_mae(y_true, y_pred),
        "MAPE (%)": calcular_mape(y_true, y_pred),
        "R2": calcular_r2(y_true, y_pred)
    }


# =====================================================
# GERAR DATAFRAME DE MÉTRICAS
# =====================================================

def gerar_df_metricas(nome_modelo, y_true, y_pred):
    """
    Gera DataFrame de uma linha para um modelo.
    """

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
    """

    df_final = pd.concat(lista_metricas, ignore_index=True)

    df_final = df_final[
        ["Modelo", "MSE", "RMSE", "MAE", "MAPE (%)", "R2"]
    ]

    return df_final.sort_values(by="MSE")


# =====================================================
# SALVAR MÉTRICAS
# =====================================================

def salvar_metricas(df_metricas, path="../Data/processed/metricas_modelos.csv"):
    """
    Salva métricas consolidadas.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_metricas.to_csv(path, index=False)