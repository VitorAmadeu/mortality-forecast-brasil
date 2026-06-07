"""
Main pipeline do projeto de previsão de séries temporais.

Executa:
- Carregamento da série
- Modelos clássicos (SARIMA e Holt-Winters)
- Modelos LSTM (E1, E2, E3, E4)
- Avaliação automática (incluindo AIC/BIC para clássicos)
- Salvamento das previsões
- Consolidação das métricas

Autor: Vitor Hugo Amadeu da Silva
"""

import os
import sys
import pandas as pd

# Garante que a raiz do projeto esteja no path,
# independente de onde o arquivo é executado
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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
    salvar_metricas,
)


# =====================================================
# CONFIGURAÇÕES
# =====================================================

DATA_PATH  = os.path.join(ROOT_DIR, "Data", "processed", "serie_temporal_mensal.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Data", "processed")
SEQ_LENGTH = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# FUNÇÃO PRINCIPAL
# =====================================================

def main():

    print("📊 Iniciando pipeline de previsão...\n")

    # ==============================
    # 1️⃣ CARREGAR SÉRIE
    # ==============================
    serie = carregar_serie(DATA_PATH)

    train_size = int(len(serie) * 0.8)
    train = serie[:train_size]
    test  = serie[train_size:]

    lista_metricas = []


    # =====================================================
    # 2️⃣ MODELOS CLÁSSICOS
    # =====================================================

    # --- SARIMA ---
    print("🔹 Executando SARIMA...")
    forecast_sarima, fit_sarima = modelo_sarima(train, test)

    df_sarima = pd.DataFrame({
        "DATA":     test.index,
        "REAL":     test.values,
        "PREVISAO": forecast_sarima.values,
    })
    df_sarima.to_csv(f"{OUTPUT_DIR}/previsao_sarima.csv", index=False)

    # Passa fit_result → gerar_df_metricas extrai AIC/BIC automaticamente
    lista_metricas.append(
        gerar_df_metricas("SARIMA", test.values, forecast_sarima.values,
                          fit_result=fit_sarima)
    )


    # --- Holt-Winters ---
    print("🔹 Executando Holt-Winters...")
    forecast_hw, fit_hw = modelo_holt_winters(train, test)

    df_hw = pd.DataFrame({
        "DATA":     test.index,
        "REAL":     test.values,
        "PREVISAO": forecast_hw.values,
    })
    df_hw.to_csv(f"{OUTPUT_DIR}/previsao_holt_winters.csv", index=False)

    lista_metricas.append(
        gerar_df_metricas("Holt-Winters", test.values, forecast_hw.values,
                          fit_result=fit_hw)
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
        "lstm_adam": {
            "nome_csv":      "LSTM Adam (E1)",
            "unidades_lstm": [128, 128, 64],
            "unidades_dense":[32, 16],
            "optimizer":     "adam",
            "epochs":        120,
            "batch_size":    16,
        },
        "lstm_adamw": {
            "nome_csv":      "LSTM AdamW (E2)",
            "unidades_lstm": [128, 128, 64],
            "unidades_dense":[32, 16],
            "optimizer":     "adamw",
            "weight_decay":  0.004,
            "epochs":        120,
            "batch_size":    16,
        },
        "lstm_adam_e3": {
            "nome_csv":      "LSTM Adam (E3)",
            "unidades_lstm": [64, 32],
            "unidades_dense":[16],
            "optimizer":     "adam",
            "epochs":        60,
            "batch_size":    32,
        },
        "lstm_adamw_e4": {
            "nome_csv":      "LSTM AdamW (E4)",
            "unidades_lstm": [64, 32],
            "unidades_dense":[16],
            "optimizer":     "adamw",
            "weight_decay":  0.005,
            "epochs":        60,
            "batch_size":    32,
        },
    }

    for chave, config in configuracoes_lstm.items():

        nome_csv = config["nome_csv"]
        print(f"🔹 Treinando {nome_csv}...")

        model = construir_lstm(
            input_shape=(SEQ_LENGTH, 1),
            unidades_lstm=config["unidades_lstm"],
            unidades_dense=config["unidades_dense"],
            optimizer=config["optimizer"],
            weight_decay=config.get("weight_decay", 0.0),
        )

        treinar_lstm(
            model,
            X_train, y_train,
            X_test,  y_test,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            verbose=0,
        )

        df_lstm = prever_lstm(
            model, X_test, scaler, y_test,
            serie.index, seq_length=SEQ_LENGTH
        )

        # Nome de arquivo alinhado com MAPA_NOMES do app.py
        df_lstm.to_csv(f"{OUTPUT_DIR}/previsao_{chave}.csv")

        # LSTM não tem AIC/BIC → fit_result=None (padrão)
        lista_metricas.append(
            gerar_df_metricas(nome_csv,
                              df_lstm["REAL"].values,
                              df_lstm["PREVISAO"].values)
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
    print(df_metricas.to_string(index=False))


# =====================================================
# EXECUÇÃO
# =====================================================

if __name__ == "__main__":
    main()