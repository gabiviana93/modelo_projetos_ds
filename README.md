# Data Science End-to-End Project

Projeto de Ciência de Dados estruturado seguindo boas práticas de Machine Learning, com separação clara entre experimentação, código reutilizável e execução do pipeline.

## Objetivo

Demonstrar um pipeline completo de Machine Learning com:
- Geração de dados sintéticos
- Preprocessamento e feature engineering
- Treinamento com XGBoost
- Avaliação de modelos
- Rastreamento com MLflow
- Inferência

## Stack

- **Python**: 3.11+
- **Data**: Pandas, NumPy
- **ML**: Scikit-learn, XGBoost
- **Experimentos**: MLflow
- **Visualização**: Matplotlib, Seaborn

## Estrutura do Projeto

```
ds-end-to-end/
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
│   ├── config.py         # Configurações
│   ├── preprocessing.py  # Preprocessamento
│   ├── features.py       # Feature engineering
│   ├── train.py          # Treinamento
│   ├── evaluate.py       # Avaliação
│   └── inference.py      # Inferência
├── scripts/
│   ├── run_pipeline.py   # Pipeline completo
│   └── train_pipeline.py # Script de treinamento
├── models/               # Modelos treinados
├── reports/
│   ├── metrics.json      # Métricas de desempenho
│   └── figures/          # Gráficos e visualizações
├── pyproject.toml        # Configuração Poetry
├── requirements.txt      # Dependências pip
├── generate_data.py      # Gera dados de teste
└── test_pipeline.py      # Script de teste end-to-end
```

## Instalação

### Opção 1: Poetry (Recomendado)

```bash
# Instalar dependências
poetry install

# Ativar ambiente virtual
poetry shell
```

### Opção 2: Pip

```bash
pip install -r requirements.txt
```

## Executando o Projeto

### 1. Gerar Dados de Teste

```bash
python generate_data.py
```

Gera 500 amostras com:
- 5 features numéricas
- 3 features categóricas  
- Target binário (classificação)

Saved in:
- `data/raw/raw_data.csv` - Dados brutos
- `data/processed/data.csv` - Dados processados

### 2. Executar Pipeline Completo

```bash
# Com Poetry
poetry run python test_pipeline.py

# Com pip
python test_pipeline.py
```

O script executa:
1. Geração de dados sintéticos
2. Divisão treino/teste
3. Preprocessamento
4. Treinamento do modelo XGBoost
5. Avaliação e métricas
6. Inferência em amostras
7. Rastreamento com MLflow

### 3. Visualizar Resultados com MLflow

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Acesse http://127.0.0.1:5000 para ver:
- Parâmetros do modelo
- Métricas de desempenho (ROC-AUC, Precision, Recall, F1)
- Artefatos salvos
- Histórico de experimentos

## Uso em Notebooks

Os notebooks em `notebooks/` podem ser executados para:
- **01_eda.ipynb**: Exploração e análise dos dados
- **02_feature_engineering.ipynb**: Criação e seleção de features
- **03_modeling.ipynb**: Experimentação com diferentes modelos

```bash
jupyter notebook notebooks/
```

## Configuração

Editar `src/config.py` para ajustar:
- `TARGET`: Nome da coluna alvo
- `TEST_SIZE`: Proporção de dados de teste
- `RANDOM_STATE`: Seed para reprodutibilidade
- `MLFLOW_EXPERIMENT`: Nome do experimento MLflow
- `MODEL_NAME`: Nome do modelo

## Adicionando Novas Dependências

Com Poetry:
```bash
poetry add nome-do-pacote
poetry add --group dev nome-do-pacote  # Para desenvolvimento
```

Com pip:
```bash
pip install nome-do-pacote
pip freeze > requirements.txt
```

## Boas Práticas

✓ Separação clara entre código de experimentação e produção  
✓ Rastreamento de experimentos com MLflow  
✓ Preprocessamento e feature engineering reutilizáveis  
✓ Versionamento de dados e modelos  
✓ Configuração centralizada  
✓ Notebooks para exploração, scripts para produção

## Troubleshooting

### Erro: FileNotFoundError para dados
```bash
python generate_data.py
```

### Erro ao instalar dependências com Poetry
```bash
poetry install --no-cache
```

### MLflow não está salvando artefatos
Verifique se o diretório `mlruns/` foi criado e tem permissões de escrita.

## Licença

MIT License - Veja LICENSE para detalhes
