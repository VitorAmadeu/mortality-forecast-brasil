"""
Dashboard Interativo - Previsão de Séries Temporais

Visualiza:
- Previsões dos modelos
- Comparação Real vs Previsto
- Ranking de métricas (MSE, RMSE, MAE, MAPE, R², AIC, BIC)
- Melhor modelo automaticamente destacado

Autor: Vitor Hugo Amadeu da Silva
"""

import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================

st.set_page_config(
    page_title="Dashboard de Previsão Temporal",
    layout="wide"
)

st.title("📊 Dashboard de Previsão de Séries Temporais")
st.markdown("Comparação entre Modelos Clássicos e LSTM")


# =====================================================
# CAMINHOS E MAPA DE NOMES
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Data", "processed")

# Mapeia chave do arquivo → nome usado no CSV de métricas
MAPA_NOMES = {
    "sarima":        "SARIMA",
    "holt_winters":  "Holt-Winters",   # ← estava faltando
    "lstm_adam":     "LSTM Adam (E1)",
    "lstm_adamw":    "LSTM AdamW (E2)",
    "lstm_adam_e3":  "LSTM Adam (E3)",
    "lstm_adamw_e4": "LSTM AdamW (E4)",
}

# Todas as métricas disponíveis no CSV (AIC/BIC são NaN para LSTM)
COLUNAS_METRICAS = ["MSE", "RMSE", "MAE", "MAPE (%)", "R2", "AIC", "BIC"]


# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

@st.cache_data
def carregar_metricas():
    path = os.path.join(DATA_DIR, "metricas_modelos.csv")
    return pd.read_csv(path)


@st.cache_data
def carregar_previsao(nome_arquivo):
    path = os.path.join(DATA_DIR, nome_arquivo)
    df = pd.read_csv(path)
    df["DATA"] = pd.to_datetime(df["DATA"])
    return df


def listar_modelos():
    arquivos = os.listdir(DATA_DIR)
    return sorted([
        arq for arq in arquivos
        if arq.startswith("previsao_") and arq.endswith(".csv")
    ])


def plotar_previsao(df, nome_modelo):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["DATA"], y=df["REAL"],
        mode="lines", name="Real",
        line=dict(color="#1f77b4")
    ))

    fig.add_trace(go.Scatter(
        x=df["DATA"], y=df["PREVISAO"],
        mode="lines", name="Previsto",
        line=dict(color="#ff7f0e", dash="dash")
    ))

    fig.update_layout(
        title=f"Real vs Previsto — {nome_modelo}",
        xaxis_title="Data",
        yaxis_title="Valor",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", y=-0.15)
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙ Configurações")

modelos_disponiveis = listar_modelos()

modelo_escolhido = st.sidebar.selectbox(
    "Escolha o modelo:",
    modelos_disponiveis,
    format_func=lambda f: MAPA_NOMES.get(
        f.replace("previsao_", "").replace(".csv", ""),
        f.replace("previsao_", "").replace(".csv", "").upper()
    )
)

mostrar_ranking = st.sidebar.checkbox(
    "Mostrar ranking completo de métricas",
    value=True
)


# =====================================================
# CARREGAR DADOS
# =====================================================

df_metricas = carregar_metricas()

chave_modelo      = modelo_escolhido.replace("previsao_", "").replace(".csv", "")
nome_modelo_csv   = MAPA_NOMES.get(chave_modelo, None)
nome_exibicao     = nome_modelo_csv if nome_modelo_csv else chave_modelo.upper()

df_previsao = carregar_previsao(modelo_escolhido)


# =====================================================
# EXIBIÇÃO PRINCIPAL
# =====================================================

col1, col2 = st.columns([2, 1])

with col1:
    plotar_previsao(df_previsao, nome_exibicao)

with col2:
    st.subheader("📈 Métricas do Modelo")

    if nome_modelo_csv:
        metricas_modelo = df_metricas[df_metricas["Modelo"] == nome_modelo_csv]
    else:
        metricas_modelo = pd.DataFrame()

    if not metricas_modelo.empty:
        row = metricas_modelo.iloc[0]

        # Exibe todas as métricas disponíveis
        for col_name in COLUNAS_METRICAS:
            if col_name in row.index:
                valor = row[col_name]
                if pd.isna(valor):
                    st.metric(label=col_name, value="N/A")
                else:
                    fmt = f"{valor:.2f}" if col_name in ("AIC", "BIC") else f"{valor:.4f}"
                    st.metric(label=col_name, value=fmt)
    else:
        st.warning("Métricas não encontradas para este modelo.")


# =====================================================
# RANKING COMPLETO
# =====================================================

if mostrar_ranking:
    st.markdown("---")
    st.subheader("🏆 Ranking Geral dos Modelos")

    df_rank     = df_metricas.sort_values(by="MSE").reset_index(drop=True)
    melhor      = df_rank.iloc[0]["Modelo"]
    melhor_aic  = df_rank.dropna(subset=["AIC"]).sort_values("AIC")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.success(f"🥇 Melhor por MSE: **{melhor}**")
    with col_r2:
        if not melhor_aic.empty:
            st.info(f"📉 Melhor AIC: **{melhor_aic.iloc[0]['Modelo']}**")

    st.dataframe(df_rank, use_container_width=True)


# =====================================================
# COMPARAÇÃO MULTIMODELO
# =====================================================

st.markdown("---")
st.subheader("📊 Comparação Visual entre Modelos")

comparar = st.multiselect(
    "Selecione modelos para comparar:",
    modelos_disponiveis,
    default=modelos_disponiveis[:2],
    format_func=lambda f: MAPA_NOMES.get(
        f.replace("previsao_", "").replace(".csv", ""),
        f.replace("previsao_", "").replace(".csv", "").upper()
    )
)

if comparar:
    fig = go.Figure()

    for arquivo in comparar:
        df_temp  = carregar_previsao(arquivo)
        chave    = arquivo.replace("previsao_", "").replace(".csv", "")
        nome_tmp = MAPA_NOMES.get(chave, chave.upper())

        fig.add_trace(go.Scatter(
            x=df_temp["DATA"], y=df_temp["PREVISAO"],
            mode="lines", name=nome_tmp
        ))

    fig.update_layout(
        title="Comparação entre Modelos (Previsões)",
        xaxis_title="Data",
        yaxis_title="Valor Previsto",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", y=-0.15)
    )

    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# RODAPÉ
# =====================================================

st.markdown("---")
st.markdown(
    "Projeto desenvolvido para análise comparativa entre modelos clássicos "
    "e Deep Learning aplicados a séries temporais."
)