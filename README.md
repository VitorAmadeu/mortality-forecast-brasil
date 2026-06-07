# 📊 Previsão de Mortalidade por IAM no Brasil

Comparação de algoritmos de séries temporais para previsão 
de mortalidade por Infarto Agudo do Miocárdio (IAM) no Brasil, 
utilizando dados do Sistema de Informações sobre Mortalidade (SIM/DATASUS).
Trabalho de Conclusão de Curso — Engenharia de Computação, UTFPR Cornélio Procópio.

---

## 🎯 Objetivo

Comparar o desempenho preditivo de três abordagens de modelagem 
em séries temporais — Holt-Winters, SARIMA e Redes Neurais LSTM — 
na estimativa da mortalidade por IAM no Brasil, identificando o modelo 
com maior precisão e aplicabilidade prática para a vigilância em saúde pública.

---

## 📋 Contexto

- **Dados:** Sistema de Informações sobre Mortalidade (SIM), código CID-10: I219
- **Período:** Janeiro de 2010 a Dezembro de 2022
- **Metodologia:** CRISP-DM (Cross Industry Standard Process for Data Mining)
- **Produto final:** Dashboard interativo para apoio à decisão em saúde pública

---

## 📊 Resultados

| Modelo | RMSE | MAE | MAPE (%) | AIC |
|---|---|---|---|---|
| **SARIMA** | **434,48** | **292,64** | **3,69** | **1345,73** |
| Holt-Winters | 478,06 | 324,14 | 4,03 | 1350,22 |
| LSTM Adam (E1) | 592,83 | 437,07 | 5,43 | N/A |
| LSTM AdamW (E2) | 571,81 | 420,43 | 5,36 | N/A |
| LSTM Adam (E3) | 578,94 | 479,57 | 6,12 | N/A |
| LSTM AdamW (E4) | 704,56 | 525,12 | 6,43 | N/A |

> 🏆 **Modelo vencedor: SARIMA(1,1,1)(1,1,1)₁₂** com MAPE de 3,69%, 
> classificado como previsão altamente precisa (MAPE < 10%) 
> segundo a escala de Lewis (1982).

---

## 🔧 Tecnologias

| Categoria | Ferramentas |
|---|---|
| Linguagem | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Séries Temporais | Statsmodels |
| Dashboard | Streamlit |
| Dados | Pandas / NumPy |
| Visualização | Plotly / Matplotlib |
| Pré-processamento | Scikit-learn |

---

## 📁 Estrutura do Projeto
├── src/
│   ├── init.py         # Inicialização do pacote
│   ├── data_loader.py      # Carregamento e validação dos dados
│   ├── preprocessing.py    # Pré-processamento e normalização
│   ├── forecasting.py      # Modelos SARIMA, Holt-Winters e LSTM
│   └── metrics.py          # Métricas de avaliação (MAE, RMSE, MAPE, AIC)
├── Notebooks/
│   ├── EDA_SIM_I219.ipynb          # Análise exploratória dos dados
│   ├── tratamento_sim_i219.ipynb   # Tratamento e limpeza dos dados
│   └── series_temporais.ipynb      # Modelagem e análise dos resultados
├── Data/
│   ├── raw/                # Dados brutos do SIM
│   └── processed/          # Dados processados e previsões
├── figuras/                # Imagens e gráficos gerados
├── app.py                  # Dashboard interativo (Streamlit)
├── main.py                 # Pipeline principal de execução
└── requirements.txt        # Dependências do projeto

---

## 🚀 Como executar

### 1. Clone o repositório
```bash
git clone https://github.com/seuusuario/mortality-forecast-brasil.git
cd mortality-forecast-brasil
```

### 2. Crie o ambiente virtual
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Execute o pipeline de modelagem
```bash
python main.py
```

### 5. Inicie o dashboard interativo
```bash
streamlit run app.py
```

---

## 📈 Modelos Implementados

### Modelos Clássicos
- **Holt-Winters** — Suavização exponencial tripla com tendência 
  aditiva e sazonalidade multiplicativa (período = 12 meses)
- **SARIMA(1,1,1)(1,1,1)₁₂** — Modelo autorregressivo integrado 
  de médias móveis sazonal, identificado via análise ACF/PACF

### Redes Neurais LSTM

| Config | Arquitetura LSTM | Dense | Otimizador | Épocas | Batch |
|---|---|---|---|---|---|
| E1 | 128→128→64 | 32→16 | Adam | 120 | 16 |
| E2 | 128→128→64 | 32→16 | AdamW | 120 | 16 |
| E3 | 64→32 | 16 | Adam | 60 | 32 |
| E4 | 64→32 | 16 | AdamW | 60 | 32 |

> Todas as configurações utilizam ativação `tanh` nas camadas LSTM, 
> `relu` nas camadas Dense ocultas e ativação linear na camada de saída.

---

## 🗂️ Fonte dos Dados

- **SIM — Sistema de Informações sobre Mortalidade**
  - Portal de Dados Abertos do Governo Federal
  - Código CID-10: I219 (IAM não especificado)
  - Período: 2010–2022
- **IBGE** — Dados populacionais para padronização da taxa de mortalidade

---

## 👤 Autor

**Vitor Hugo Amadeu da Silva**  
Engenharia de Computação — UTFPR Cornélio Procópio  
Orientadora: Profª. Drª. Elisangela Aparecida da Silva Lizzi

---

## 📄 Licença

MIT License — sinta-se livre para usar, modificar e distribuir 
este projeto com os devidos créditos ao autor.
