# -*- coding: utf-8 -*-
"""
Main pipeline do projeto de previsão de séries temporais.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random

# ✅ FIXAR SEEDS PARA REPRODUTIBILIDADE
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuração adicional para garantir determinismo
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

from src import (
    carregar_serie,
    preparar_dados_lstm,
    modelo_sarima,
    modelo_holt_winters,
    construir_lstm,
    treinar_lstm,
    prever_lstm,
    gerar_df_metricas,
    consolidar_metricas,
    salvar_metricas
)


# =====================================================
# CONFIGURAÇÕES
# =====================================================

DATA_PATH = "Data/processed/serie_temporal_mensal.csv"
OUTPUT_DIR = "Data/processed"
SEQ_LENGTH = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# FUNÇÃO PRINCIPAL
# =====================================================

def main():
    
    print("📊 Iniciando pipeline de previsão...\n")
    
    # ✅ VALIDAÇÃO DE ARQUIVO
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"❌ Arquivo não encontrado: {DATA_PATH}\n"
            f"Execute primeiro os notebooks de tratamento!"
        )
    
    # 1️⃣ CARREGAR SÉRIE
    serie = carregar_serie(DATA_PATH)
    
    train_size = int(len(serie) * 0.8)
    train = serie[:train_size]
    test = serie[train_size:]
    
    lista_metricas = []
    
    # =====================================================
    # 2️⃣ MODELOS CLÁSSICOS
    # =====================================================
    
    print("🔹 Executando SARIMA...")
    forecast_sarima = modelo_sarima(train, test)
    
    df_sarima = pd.DataFrame({
        "DATA": test.index,
        "REAL": test.values,
        "PREVISAO": forecast_sarima.values
    })
    
    df_sarima.to_csv(f"{OUTPUT_DIR}/previsao_sarima.csv", index=False)
    
    lista_metricas.append(
        gerar_df_metricas("SARIMA", test.values, forecast_sarima.values)
    )
    
    
    print("🔹 Executando Holt-Winters...")
    forecast_hw = modelo_holt_winters(train, test)
    
    df_hw = pd.DataFrame({
        "DATA": test.index,
        "REAL": test.values,
        "PREVISAO": forecast_hw.values
    })
    
    df_hw.to_csv(f"{OUTPUT_DIR}/previsao_holt_winters.csv", index=False)
    
    lista_metricas.append(
        gerar_df_metricas("Holt-Winters", test.values, forecast_hw.values)
    )
    
    
    # =====================================================
    # 3️⃣ PREPARAR DADOS LSTM
    # =====================================================
    
    print("🔹 Preparando dados para LSTM...")
    scaler, X_train, X_test, y_train, y_test = preparar_dados_lstm(
        serie,
        seq_length=SEQ_LENGTH
    )
    
    
    # =====================================================
    # 4️⃣ MODELOS LSTM
    # =====================================================
    
    configuracoes_lstm = {
        "LSTM_E1_Adam": {
            "unidades_lstm": [128, 128, 64],
            "unidades_dense": [32, 16],
            "optimizer": "adam",
            "epochs": 120,
            "batch_size": 16
        },
        "LSTM_E2_AdamW": {
            "unidades_lstm": [128, 128, 64],
            "unidades_dense": [32, 16],
            "optimizer": "adamw",
            "weight_decay": 0.004,
            "epochs": 120,
            "batch_size": 16
        },
        "LSTM_E3_Adam": {
            "unidades_lstm": [64, 32],
            "unidades_dense": [16],
            "optimizer": "adam",
            "epochs": 60,
            "batch_size": 32
        },
        "LSTM_E4_AdamW": {
            "unidades_lstm": [64, 32],
            "unidades_dense": [16],
            "optimizer": "adamw",
            "weight_decay": 0.005,
            "epochs": 60,
            "batch_size": 32
        }
    }
    
    
    for nome_modelo, config in configuracoes_lstm.items():
        
        print(f"🔹 Treinando {nome_modelo}...")
        
        # ✅ RESETAR SEED ANTES DE CADA MODELO
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        
        model = construir_lstm(
            input_shape=(SEQ_LENGTH, 1),
            unidades_lstm=config["unidades_lstm"],
            unidades_dense=config["unidades_dense"],
            optimizer=config["optimizer"],
            weight_decay=config.get("weight_decay", 0.0)
        )
        
        treinar_lstm(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            verbose=0
        )
        
        df_lstm = prever_lstm(
            model,
            X_test,
            scaler,
            y_test,
            serie.index,
            seq_length=SEQ_LENGTH
        )
        
        df_lstm.to_csv(
            f"{OUTPUT_DIR}/previsao_{nome_modelo.lower()}.csv",
            index=False  # ✅ IMPORTANTE: não salvar índice
        )
        
        lista_metricas.append(
            gerar_df_metricas(
                nome_modelo,
                df_lstm["REAL"].values,
                df_lstm["PREVISAO"].values
            )
        )
    
    
    # =====================================================
    # 5️⃣ CONSOLIDAR MÉTRICAS
    # =====================================================
    
    print("📈 Consolidando métricas...")
    
    df_metricas = consolidar_metricas(lista_metricas)
    
    salvar_metricas(
        df_metricas,
        path=f"{OUTPUT_DIR}/metricas_modelos.csv"
    )
    
    print("\n✅ Pipeline finalizado com sucesso!")
    print("📂 Arquivos salvos em Data/processed/")
    print("\n🏆 Ranking dos Modelos:")
    print(df_metricas)


# =====================================================
# EXECUÇÃO
# =====================================================

if __name__ == "__main__":
    main()