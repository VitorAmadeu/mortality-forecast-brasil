"""
Pacote de Modelagem de Séries Temporais

Este módulo centraliza:
- Carregamento de dados
- Pré-processamento
- Modelos de previsão
- Métricas de avaliação

Projeto estruturado para uso acadêmico e produção.
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
    listar_arquivos_processados
)


# ==========================
# PREPROCESSING
# ==========================

from .preprocessing import (
    validar_serie,
    tratar_nulos,
    normalizar_serie,
    criar_sequencias,
    split_temporal,
    preparar_dados_lstm  # ✅ ADICIONADO
)


# ==========================
# FORECASTING
# ==========================

from .forecasting import (
    modelo_sarima,        # ✅ ADICIONADO
    modelo_holt_winters,
    construir_lstm,       # ✅ ADICIONADO
    treinar_lstm,         # ✅ ADICIONADO
    prever_lstm          # ✅ ADICIONADO
)


# ==========================
# METRICS
# ==========================

from .metrics import (
    calcular_mse,
    calcular_rmse,
    calcular_mae,
    calcular_mape,
    calcular_r2,
    avaliar_modelo,
    gerar_df_metricas,    # ✅ ADICIONADO
    consolidar_metricas,  # ✅ ADICIONADO
    salvar_metricas       # ✅ ADICIONADO
)


# ==========================
# EXPORTS PÚBLICOS
# ==========================

__all__ = [
    # Data Loader
    "carregar_serie",
    "carregar_previsao",
    "carregar_metricas",
    "listar_arquivos_processados",
    
    # Preprocessing
    "validar_serie",
    "tratar_nulos",
    "normalizar_serie",
    "criar_sequencias",
    "split_temporal",
    "preparar_dados_lstm",
    
    # Forecasting
    "modelo_sarima",
    "modelo_holt_winters",
    "construir_lstm",
    "treinar_lstm",
    "prever_lstm",
    
    # Metrics
    "calcular_mse",
    "calcular_rmse",
    "calcular_mae",
    "calcular_mape",
    "calcular_r2",
    "avaliar_modelo",
    "gerar_df_metricas",
    "consolidar_metricas",
    "salvar_metricas"
]
