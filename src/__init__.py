"""
Pacote de Modelagem de Séries Temporais

Este módulo centraliza:
- Carregamento de dados
- Pré-processamento
- Modelos de previsão
- Métricas de avaliação

Projeto estruturado para uso acadêmico e produção.

Autor: Vitor Hugo Amadeu da Silva
"""

__version__ = "1.0.0"
__author__ = "Vitor Hugo Amadeu da Silva"


# ==========================
# DATA LOADER
# ==========================

from .data_loader import (
    carregar_serie,
    carregar_previsao,
    carregar_metricas,
    listar_arquivos_processados,
)


# ==========================
# PREPROCESSING
# ==========================

from .preprocessing import (
    normalizar_serie,
    criar_sequencias,
    preparar_dados_lstm,   # função que main.py importa
)


# ==========================
# FORECASTING
# ==========================

from .forecasting import (
    modelo_sarima,         # retorna (forecast, fit)
    modelo_holt_winters,   # retorna (forecast, fit)
    construir_lstm,
    treinar_lstm,
    prever_lstm,
)


# ==========================
# METRICS
# ==========================

from .metrics import (
    avaliar_modelo,
    avaliar_modelo_classico,
    calcular_mape,
    calcular_aic_bic,
    gerar_df_metricas,
    consolidar_metricas,
    salvar_metricas,
)


# ==========================
# EXPORTS PÚBLICOS
# ==========================

__all__ = [
    # data_loader
    "carregar_serie",
    "carregar_previsao",
    "carregar_metricas",
    "listar_arquivos_processados",
    # preprocessing
    "normalizar_serie",
    "criar_sequencias",
    "preparar_dados_lstm",
    # forecasting
    "modelo_sarima",
    "modelo_holt_winters",
    "construir_lstm",
    "treinar_lstm",
    "prever_lstm",
    # metrics
    "avaliar_modelo",
    "avaliar_modelo_classico",
    "calcular_mape",
    "calcular_aic_bic",
    "gerar_df_metricas",
    "consolidar_metricas",
    "salvar_metricas",
]