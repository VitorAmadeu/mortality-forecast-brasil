# 📊 Mortality Forecast Brasil

Análise comparativa de modelos estatísticos (SARIMA, Holt-Winters) e Deep Learning (LSTM) para previsão de séries temporais de mortalidade no Brasil utilizando dados do SIM/DATASUS.

## 🎯 Objetivo

Comparar a performance de modelos clássicos de séries temporais com redes neurais LSTM na previsão de óbitos mensais.

## 🔧 Tecnologias

- **Python 3.10+**
- **TensorFlow/Keras** - LSTM
- **Statsmodels** - SARIMA & Holt-Winters
- **Streamlit** - Dashboard interativo
- **Pandas/NumPy** - Manipulação de dados
- **Plotly** - Visualizações

## 📁 Estrutura do Projeto
```
├── src/                    # Módulos do projeto
│   ├── data_loader.py      # Carregamento de dados
│   ├── preprocessing.py    # Pré-processamento
│   ├── forecasting.py      # Modelos de previsão
│   └── metrics.py          # Métricas de avaliação
├── Notebooks/              # Análises exploratórias
├── Data/                   # Dados brutos e processados
├── app.py                  # Dashboard Streamlit
├── main.py                 # Pipeline principal
└── requirements.txt        # Dependências
```

## 🚀 Como executar

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/mortality-forecast-brasil.git
cd mortality-forecast-brasil
```

### 2. Crie ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instale dependências
```bash
pip install -r requirements.txt
```

### 4. Execute o pipeline
```bash
python main.py
```

### 5. Rode o dashboard
```bash
streamlit run app.py
```

## 📊 Modelos Implementados

- **SARIMA (1,1,1)(1,1,1,12)** - Modelo autoregressivo sazonal
- **Holt-Winters** - Suavização exponencial tripla
- **LSTM E1** - 3 camadas LSTM (128-128-64) + Adam
- **LSTM E2** - 3 camadas LSTM (128-128-64) + AdamW
- **LSTM E3** - 2 camadas LSTM (64-32) + Adam
- **LSTM E4** - 2 camadas LSTM (64-32) + AdamW

## 📈 Métricas de Avaliação

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coeficiente de Determinação)

## 👤 Autor

**Vitor Hugo Amadeu da Silva**

## 📄 Licença

MIT License
