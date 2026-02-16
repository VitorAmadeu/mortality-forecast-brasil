"""
Módulo responsável pelo carregamento de dados do projeto de séries temporais.
Inclui:
- Série temporal principal
- Arquivos de previsão
- Arquivo consolidado de métricas
"""

import os
import pandas as pd


# =====================================================
# FUNÇÃO INTERNA DE VALIDAÇÃO
# =====================================================

def _verificar_arquivo(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")


# =====================================================
# CARREGAR SÉRIE TEMPORAL PRINCIPAL
# =====================================================

def carregar_serie(path: str) -> pd.Series:
    """
    Carrega a série temporal principal.
    Espera colunas: DATA + valor numérico.
    """

    _verificar_arquivo(path)

    df = pd.read_csv(path)

    if "DATA" not in df.columns:
        raise ValueError("O arquivo deve conter a coluna 'DATA'.")

    df["DATA"] = pd.to_datetime(df["DATA"])
    df.set_index("DATA", inplace=True)

    # Garante frequência mensal
    df = df.asfreq("MS")

    # Retorna apenas a primeira coluna numérica
    return df.iloc[:, 0]


# =====================================================
# CARREGAR PREVISÕES
# =====================================================

def carregar_previsao(path: str) -> pd.DataFrame:
    """
    Carrega arquivos de previsão.
    Espera colunas: DATA, REAL, PREVISAO
    """

    _verificar_arquivo(path)

    df = pd.read_csv(path)

    colunas_necessarias = {"DATA", "REAL", "PREVISAO"}

    if not colunas_necessarias.issubset(df.columns):
        raise ValueError(
            f"O arquivo deve conter as colunas: {colunas_necessarias}"
        )

    df["DATA"] = pd.to_datetime(df["DATA"])
    df.set_index("DATA", inplace=True)

    return df.sort_index()


# =====================================================
# CARREGAR MÉTRICAS
# =====================================================

def carregar_metricas(path: str) -> pd.DataFrame:
    """
    Carrega o arquivo consolidado de métricas.
    Espera colunas como:
    Modelo, MSE, MAPE (%)
    """

    _verificar_arquivo(path)

    df = pd.read_csv(path)

    if "Modelo" not in df.columns:
        raise ValueError("O arquivo de métricas deve conter a coluna 'Modelo'.")

    return df.sort_values(by="MSE")


# =====================================================
# FUNÇÃO UTILITÁRIA OPCIONAL
# =====================================================

def listar_arquivos_processados(diretorio: str) -> list:
    """
    Lista todos os CSV dentro da pasta processed.
    """

    if not os.path.isdir(diretorio):
        raise NotADirectoryError(f"Diretório inválido: {diretorio}")

    return [
        arquivo
        for arquivo in os.listdir(diretorio)
        if arquivo.endswith(".csv")
    ]