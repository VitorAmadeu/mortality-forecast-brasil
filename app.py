# -*- coding: utf-8 -*-
"""
Dashboard Interativo - Previsão de Séries Temporais
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

# ✅ CAMINHO ABSOLUTO
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")


# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

@st.cache_data
def carregar_metricas():
    path = os.path.join(DATA_DIR, "metricas_modelos.csv")
    
    if not os.path.exists(path):
        st.error("❌ Arquivo de métricas não encontrado! Execute o pipeline primeiro (main.py)")
        st.stop()
    
    return pd.read_csv(path)


@st.cache_data
def carregar_previsao(nome_arquivo):
    path = os.path.join(DATA_DIR, nome_arquivo)
    
    if not os.path.exists(path):
        st.error(f"❌ Arquivo não encontrado: {nome_arquivo}")
        st.stop()
    
    df = pd.read_csv(path)
    
    # ✅ VALIDAÇÃO DE COLUNAS
    if "DATA" not in df.columns:
        st.error(f"❌ Arquivo {nome_arquivo} não contém a coluna 'DATA'")
        st.info(f"Colunas encontradas: {df.columns.tolist()}")
        st.stop()
    
    df["DATA"] = pd.to_datetime(df["DATA"])
    return df


@st.cache_data
def carregar_serie_completa():
    """✅ NOVA FUNÇÃO: Carrega série temporal completa"""
    path = os.path.join(DATA_DIR, "serie_temporal_mensal.csv")
    
    if not os.path.exists(path):
        return None
    
    df = pd.read_csv(path)
    df["DATA"] = pd.to_datetime(df["DATA"])
    return df


def listar_modelos():
    if not os.path.exists(DATA_DIR):
        st.error("❌ Pasta Data/processed não encontrada! Execute o pipeline primeiro.")
        st.stop()
    
    arquivos = os.listdir(DATA_DIR)
    modelos = [
        arq for arq in arquivos
        if arq.startswith("previsao_") and arq.endswith(".csv")
    ]
    
    if not modelos:
        st.warning("⚠️ Nenhum arquivo de previsão encontrado! Execute o main.py.")
        st.stop()
    
    return modelos


def extrair_nome_modelo(nome_arquivo):
    """✅ NOVA FUNÇÃO: Extrai nome limpo do modelo"""
    nome = nome_arquivo.replace("previsao_", "").replace(".csv", "")
    
    # Padroniza nomes conhecidos
    mapeamento = {
        "holt_winters": "Holt-Winters",
        "sarima": "SARIMA",
        "lstm_e1_adam": "LSTM_E1_Adam",
        "lstm_e2_adamw": "LSTM_E2_AdamW",
        "lstm_e3_adam": "LSTM_E3_Adam",
        "lstm_e4_adamw": "LSTM_E4_AdamW"
    }
    
    return mapeamento.get(nome.lower(), nome)


def plotar_previsao(df_previsao, nome_modelo, df_serie_completa=None):
    """✅ GRÁFICO COM SÉRIE COMPLETA + PREVISÃO"""
    
    fig = go.Figure()
    
    # ✅ Adiciona série histórica completa (se disponível)
    if df_serie_completa is not None:
        # Pega apenas dados anteriores ao período de teste
        data_inicio_teste = df_previsao["DATA"].min()
        df_historico = df_serie_completa[
            df_serie_completa["DATA"] < data_inicio_teste
        ]
        
        fig.add_trace(
            go.Scatter(
                x=df_historico["DATA"],
                y=df_historico.iloc[:, 1],  # Primeira coluna numérica
                mode="lines",
                name="Histórico",
                line=dict(color="gray", width=1.5),
                opacity=0.7
            )
        )
    
    # Real (período de teste)
    fig.add_trace(
        go.Scatter(
            x=df_previsao["DATA"],
            y=df_previsao["REAL"],
            mode="lines+markers",
            name="Real (Teste)",
            line=dict(color="blue", width=2),
            marker=dict(size=6)
        )
    )
    
    # Previsão
    fig.add_trace(
        go.Scatter(
            x=df_previsao["DATA"],
            y=df_previsao["PREVISAO"],
            mode="lines+markers",
            name="Previsto",
            line=dict(color="red", width=2, dash="dash"),
            marker=dict(size=6)
        )
    )
    
    fig.update_layout(
        title=f"Série Completa + Previsão — {nome_modelo}",
        xaxis_title="Data",
        yaxis_title="Número de Óbitos",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
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
    format_func=extrair_nome_modelo  # ✅ Exibe nome formatado
)

mostrar_ranking = st.sidebar.checkbox(
    "Mostrar ranking completo de métricas",
    value=True
)


# =====================================================
# CARREGAR DADOS
# =====================================================

df_metricas = carregar_metricas()
df_previsao = carregar_previsao(modelo_escolhido)
df_serie_completa = carregar_serie_completa()  # ✅ Série completa

nome_modelo_exibicao = extrair_nome_modelo(modelo_escolhido)


# =====================================================
# EXIBIÇÃO PRINCIPAL
# =====================================================

col1, col2 = st.columns([2, 1])

with col1:
    plotar_previsao(df_previsao, nome_modelo_exibicao, df_serie_completa)

with col2:
    st.subheader("📈 Métricas do Modelo")
    
    # ✅ BUSCA MELHORADA - ignora case e espaços
    nome_busca = nome_modelo_exibicao.lower().replace("-", "").replace("_", "").replace(" ", "")
    
    metricas_modelo = df_metricas[
        df_metricas["Modelo"].str.lower().str.replace("-", "").str.replace("_", "").str.replace(" ", "") == nome_busca
    ]
    
    if not metricas_modelo.empty:
        metricas = metricas_modelo.iloc[0]
        
        st.metric(label="MSE", value=f"{metricas['MSE']:.2f}")
        st.metric(label="RMSE", value=f"{metricas['RMSE']:.2f}")
        st.metric(label="MAE", value=f"{metricas['MAE']:.2f}")
        st.metric(label="MAPE (%)", value=f"{metricas['MAPE (%)']:.2f}%")
        st.metric(label="R²", value=f"{metricas['R2']:.4f}")
    else:
        st.warning(f"⚠️ Métricas não encontradas para: {nome_modelo_exibicao}")
        st.info("Modelos disponíveis no CSV:")
        st.write(df_metricas["Modelo"].tolist())


# =====================================================
# RANKING COMPLETO
# =====================================================

if mostrar_ranking:
    st.markdown("---")
    st.subheader("🏆 Ranking Geral dos Modelos")
    
    df_rank = df_metricas.sort_values(by="MSE")
    melhor_modelo = df_rank.iloc[0]["Modelo"]
    
    st.success(f"🥇 Melhor Modelo: {melhor_modelo}")
    
    st.dataframe(
        df_rank.style.format({
            "MSE": "{:.2f}",
            "RMSE": "{:.2f}",
            "MAE": "{:.2f}",
            "MAPE (%)": "{:.2f}",
            "R2": "{:.4f}"
        }),
        use_container_width=True
    )


# =====================================================
# COMPARAÇÃO MULTIMODELO
# =====================================================

st.markdown("---")
st.subheader("📊 Comparação Visual entre Modelos")

comparar = st.multiselect(
    "Selecione modelos para comparar:",
    modelos_disponiveis,
    default=modelos_disponiveis[:2] if len(modelos_disponiveis) >= 2 else modelos_disponiveis,
    format_func=extrair_nome_modelo
)

if comparar:
    fig = go.Figure()
    
    for arquivo in comparar:
        df_temp = carregar_previsao(arquivo)
        nome_temp = extrair_nome_modelo(arquivo)
        
        fig.add_trace(
            go.Scatter(
                x=df_temp["DATA"],
                y=df_temp["PREVISAO"],
                mode="lines+markers",
                name=nome_temp
            )
        )
    
    fig.update_layout(
        title="Comparação entre Modelos (Previsões)",
        xaxis_title="Data",
        yaxis_title="Valor Previsto",
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


# =====================================================
# RODAPÉ
# =====================================================

st.markdown("---")
st.markdown(
    "**Projeto desenvolvido para análise comparativa entre modelos clássicos "
    "e Deep Learning aplicados a séries temporais.**"
)
st.markdown("*Autor: Vitor Hugo Amadeu da Silva*")