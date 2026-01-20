# Data Science End-to-End Project

Projeto de Ciência de Dados estruturado seguindo boas práticas de Machine Learning, com separação clara entre experimentação, código reutilizável e execução do pipeline.

## Objetivo

Demonstrar um pipeline completo de Machine Learning com:
- Geração de dados sintéticos
- Preprocessamento e feature engineering
- Treinamento com XGBoost
- Avaliação de modelos
- Rastreamento com MLflow 2.22.4
- Inferência com signature automática

## Stack Tecnológico

- **Python**: 3.11+
- **Data**: Pandas 2.0+, NumPy 1.24+
- **ML**: Scikit-learn 1.3+, XGBoost 2.0+
- **Validação**: Cross-validation com StratifiedKFold
- **Experimentos**: MLflow 2.22.4
- **Interpretabilidade**: SHAP, Feature Importance
- **Monitoramento**: Streamlit, PSI (Population Stability Index)
- **Logging**: JSON estruturado
- **Visualização**: Matplotlib 3.8+, Seaborn 0.13+
- **Ambiente**: Poetry 1.7+

## Estrutura do Projeto

```
modelo_projetos_ds/
├── data/
│   ├── raw/              # Dados brutos
│   ├── processed/        # Dados processados
│   └── external/         # Dados externos
├── notebooks/
│   ├── 01_eda.ipynb                    # Análise exploratória
│   ├── 02_feature_engineering.ipynb    # Engenharia de features
│   └── 03_modeling.ipynb              # Modelagem
├── src/
│   ├── __init__.py
│   ├── config.py         # Configurações centralizadas
│   ├── preprocessing.py  # Preprocessamento de dados (com logging)
│   ├── features.py       # Feature engineering com SimpleImputer
│   ├── train.py          # Treinamento + Cross-validation com logging
│   ├── evaluate.py       # Avaliação de métricas com logging
│   ├── monitoring.py     # Monitoramento de drift com logging
│   ├── inference.py      # Inferência com logging
│   ├── interpret.py      # Feature importance e SHAP
│   └── logger.py         # Sistema genérico de logging JSON
├── scripts/
│   ├── run_pipeline.py       # Pipeline completo de produção
│   ├── train_pipeline.py     # Script de treinamento com CV
│   ├── test_pipeline.py      # Testes end-to-end
│   ├── monitoring_pipeline.py # Monitoramento de drift
│   └── dashboard.py          # Dashboard Streamlit
├── models/               # Modelos treinados (versionados)
├── reports/
│   ├── metrics.json      # Métricas de desempenho
│   ├── drift.json        # Detecção de data drift
│   └── figures/          # Gráficos e visualizações
├── mlruns/               # Artefatos MLflow
├── pyproject.toml        # Configuração Poetry
├── requirements.txt      # Dependências pip
├── generate_data.py      # Gerador de dados sintéticos
├── POETRY_GUIDE.md       # Guia de uso do Poetry
└── README.md             # Este arquivo
```

## Instalação

### Opção 1: Poetry (Recomendado)

```bash
# Instalar dependências e criar ambiente virtual
poetry install

# Ativar ambiente virtual
poetry shell

# Ou executar comandos sem ativar
poetry run python scripts/run_pipeline.py
```

### Opção 2: Pip + Venv

```bash
# Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Guia Rápido

### 1. Gerar Dados de Teste

```bash
poetry run python generate_data.py
```

Gera 500 amostras com:
- 5 features numéricas
- 3 features categóricas
- Target binário (classificação)

Salva em:
- `data/raw/raw_data.csv` - Dados brutos
- `data/processed/data.csv` - Dados processados

### 2. Executar Pipeline Completo

```bash
# Com Poetry
poetry run python scripts/run_pipeline.py

# Com pip
python scripts/run_pipeline.py
```

**O que o pipeline faz:**
1. ✓ Carrega e preprocessa dados
2. ✓ Constrói preprocessador (StandardScaler + OneHotEncoder)
3. ✓ Treina modelo XGBoost com 300 estimadores
4. ✓ Avalia modelo (ROC-AUC, Precision, Recall, F1)
5. ✓ Registra métricas em `reports/metrics.json`
6. ✓ Salva modelo com MLflow (com signature automática)
7. ✓ Registra tags e parâmetros no MLflow

### 3. Executar Testes End-to-End

```bash
poetry run python scripts/test_pipeline.py
```

Executa teste completo com dados sintéticos:
- Geração de dados
- Divisão treino/teste
- Treinamento
- Avaliação
- Inferência em amostras
- Rastreamento com MLflow

### 4. Visualizar Resultados com MLflow

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Acesse `http://127.0.0.1:5000` para ver:
- ✓ Parâmetros do modelo
- ✓ Métricas de desempenho
- ✓ Artefatos salvos (modelo + preprocessador)
- ✓ Tags e descrição do modelo
- ✓ Histórico de todas as runs
- ✓ Comparação entre experimentos

## Configuração

Editar `src/config.py` para ajustar:

```python
# Configuração de Dados
TARGET = "target"                    # Coluna alvo
TEST_SIZE = 0.2                      # 20% para teste
RANDOM_STATE = 42                    # Seed para reprodutibilidade

# Configuração MLflow
MLFLOW_TRACKING_URI = "..."          # Diretório local ou servidor remoto
MLFLOW_EXPERIMENT = "mlflow_test_experiments"
MODEL_NAME = "xgboost_classifier"

# Parâmetros do XGBoost
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss"
}
```

## Desenvolvimento

### Uso em Notebooks

Os notebooks em `notebooks/` podem ser executados para exploração e experimentação:

```bash
jupyter notebook notebooks/
```

- **01_eda.ipynb**: Exploração e análise dos dados
- **02_feature_engineering.ipynb**: Criação e seleção de features
- **03_modeling.ipynb**: Experimentação com diferentes modelos

### Adicionando Novas Dependências

**Com Poetry:**
```bash
# Adicionar ao projeto
poetry add nome-do-pacote

# Adicionar apenas para desenvolvimento (testes, linting, etc)
poetry add --group dev nome-do-pacote

# Atualizar e sincronizar dependências
poetry lock
poetry install
```

**Com pip:**
```bash
pip install nome-do-pacote
pip freeze > requirements.txt
```

Para mais detalhes, veja [POETRY_GUIDE.md](POETRY_GUIDE.md).

## Boas Práticas Implementadas

✅ **Separação clara** entre código de experimentação (notebooks) e produção (scripts)  
✅ **Rastreamento de experimentos** com MLflow (versão 2.22.4)  
✅ **Cross-validation** com StratifiedKFold (5-fold) e 4 métricas  
✅ **Feature Importance** nativa do XGBoost + SHAP values  
✅ **Preprocessamento robusto** com SimpleImputer (mediana + constant)  
✅ **Dashboard interativo** com Streamlit (métricas, gráficos, alertas)  
✅ **Logging estruturado** em JSON (genérico em todos os módulos)  
✅ **Monitoramento de data drift** com PSI por feature  
✅ **Versionamento automático** de modelos e artefatos  
✅ **Configuração centralizada** em `src/config.py`  
✅ **Signature automática** para modelos (sem warnings)  
✅ **Tags e descrição** de modelos para rastreabilidade  
✅ **Testes end-to-end** para validar pipeline completo  
✅ **Testes unitários** com pytest e fixtures centralizadas

## Fluxo de Trabalho Recomendado

### 1. Exploração e Análise
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Experimentação
```bash
jupyter notebook notebooks/03_modeling.ipynb
```

### 3. Rastreamento com MLflow
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

### 4. Testes
```bash
poetry run python scripts/test_pipeline.py
```

### 5. Produção
```bitash
poetry run python scripts/run_pipeline.py
```

### 6. Monitoramento de Drift
```bash
poetry run python scripts/monitoring_pipeline.py
```

### 7. Dashboard de Performance
```bash
poetry run streamlit run scripts/dashboard.py
```
Acesse `http://localhost:8501` para visualizar:
- Métricas em tempo real (ROC-AUC, F1, Precision, Recall)
- Evolução temporal das métricas
- Histórico de runs do MLflow
- Alertas automáticos de degradação

## Troubleshooting

### Erro: FileNotFoundError para dados
```bash
poetry run python generate_data.py
```

### Erro: "ModuleNotFoundError: No module named 'src'"
Execute os scripts **sempre** a partir do diretório raiz do projeto:
```bash
cd /path/to/modelo_projetos_ds
poetry run python scripts/run_pipeline.py
```

### Erro ao instalar dependências
```bash
# Limpar cache e reinstalar
poetry install --no-cache
poetry lock --no-cache
```

### MLflow não está salvando artefatos corretamente
O tracking URI está configurado em `src/config.py`:
```python
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")
```
Verifique se o diretório `mlruns/` existe e tem permissões de escrita.

### Modelo salvo sem signature (MLflow)
Certifique-se de passar `X_example` ao chamar `save_model()`:
```python
save_model(pipeline, X_example=X_test)
```

## Métricas de Desempenho

Após executar o pipeline, verifique `reports/metrics.json`:

```json
{
    "roc_auc": 0.596,
    "report": {
        "0": {"precision": 0.47, "recall": 0.57, "f1-score": 0.51},
        "1": {"precision": 0.48, "recall": 0.37, "f1-score": 0.42},
        "accuracy": 0.47
    }
}
```

### Métricas Rastreadas no MLflow
- ✓ ROC-AUC
- ✓ Precision, Recall, F1-score por classe
- ✓ Número de amostras de treinamento
- ✓ Número de features

### Gerenciamento de projeto, Testes Unitários e CI
✔️ Testes unitários com pytest  
✔️ Pipeline CI com GitHub Actions  
✔️ Coverage automatizado  
✔️ Gerenciamento de dependências com Poetry

### Tags Registradas
- `model_type`: xgboost_classifier
- `framework`: scikit-learn
- `preprocessing`: StandardScaler + OneHotEncoder

## Documentação Adicional

- **[pyproject.toml](pyproject.toml)** - Configuração de dependências e metadados
- **[POETRY_GUIDE.md](POETRY_GUIDE.md)** - Guia completo de uso do Poetry
- **[SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md)** - Resumo técnico da solução

## Próximos Passos

- [ ] Integrar validação com evidently para detectar data drift
- [ ] Adicionar testes unitários com pytest
- [ ] Implementar CI/CD com GitHub Actions
- [ ] Deploy em produção com FastAPI
- [ ] Criar dashboard com Streamlit
- [ ] Adicionar log e tratamento de erros
- [ ] Documentar API de inferência

## Licença

MIT License - Veja [LICENSE](LICENSE) para detalhes

## Autor

Gabriela - 2026
