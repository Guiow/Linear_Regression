# Análise de Engajamento de Influenciadores no Instagram

## Descrição

Projeto de análise preditiva usando modelos de Regressão Linear para prever taxas de engajamento de influenciadores no Instagram. O projeto implementa uma análise completa, desde o pré-processamento dos dados até a avaliação detalhada dos modelos, incluindo visualizações e documentação abrangente.

## Estrutura e Funcionalidade dos Arquivos

### Análise e Visualização (`src/`)

#### `data_visualization.py`

- Implementa funções especializadas para visualização dos dados
- Gera gráficos de distribuição de engajamento
- Cria matrizes de correlação
- Produz boxplots para análise de outliers
- Visualiza relações entre variáveis
- Funções principais:
  - `visualize_all()`: Coordena todas as visualizações
  - `eng_rate_distribution()`: Análise da variável dependente
  - `relation_between_variables()`: Análise de correlações
  - `correlation_matrix()`: Heatmap de correlações

#### `exploratory_analysis.py`

- Realiza análise exploratória inicial dos dados
- Examina distribuições e estatísticas básicas
- Identifica valores ausentes
- Funções principais:
  - `initial_data_analysis()`: Estatísticas descritivas
  - `num_values_analysis()`: Análise de valores nulos
  - `calculate_results()`: Métricas resumidas

### Pré-processamento (`src/preprocessing.py`)

- Implementa pipeline completo de pré-processamento
- Trata valores ausentes e outliers
- Normaliza features
- Converte métricas (K, M, B para valores numéricos)
- Funções principais:
  - `load_preprocessed_data()`: Carrega e processa dados
  - `convert_metrics()`: Converte unidades
  - `normalize_features()`: Padronização de features
  - `handle_outliers()`: Remoção de outliers

### Modelagem e Avaliação

#### `train_model.py`

- Implementa três modelos de regressão:
  - Regressão Linear
  - Ridge (L2)
  - Lasso (L1)
- Treina modelos com validação cruzada
- Gera métricas de performance
- Classes e funções:
  - `EngagementModel`: Classe principal de modelagem
  - `plot_regularization_tec()`: Visualiza predições
  - `save_results()`: Documenta resultados

#### `evaluate_model.py`

- Avalia performance dos modelos
- Calcula métricas (R², MSE, MAE)
- Analisa resíduos
- Funções principais:
  - `calculate_metrics()`: Métricas de avaliação
  - `plot_residuals()`: Análise de resíduos
  - `plot_feature_importance()`: Importância das features

#### `model_comparison.py`

- Compara performance entre modelos
- Gera visualizações comparativas
- Analisa coeficientes
- Funções principais:
  - `create_comparison_plots()`: Gráficos comparativos
  - `format_metrics_results_values()`: Formatação de métricas

#### `model_visualization.py`

- Gera visualizações específicas dos modelos
- Analisa homocedasticidade
- Funções principais:
  - `visualize_homosterasticity_grapth()`: Análise de resíduos

### Documentação

#### `generate_report.py`

- Gera relatório técnico em PDF
- Documenta metodologia e resultados
- Inclui visualizações e análises
- Estrutura:
  - Capa e resumo
  - Metodologia detalhada
  - Resultados e discussão
  - Conclusões

## Instalação

```bash

git clone https://github.com/seu-usuario/analise-engajamento-influenciadores

cd analise-engajamento-influenciadores

python -m virtualenv venv

source venv/bin/activate

pip install -r requirements.txt

```

## Execução

1. Pré-processamento e análise exploratória:

```bash
python src/preprocessing.py
python src/exploratory_analysis.py
```

2. Treinamento e avaliação:

```bash
python src/train_model.py
python src/evaluate_model.py
```

3. Geração de relatório:

```bash
python src/generate_report.py
```

## Requisitos

- Python 3.12
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy
- reportlab

## Estrutura de Diretórios

```
analise-engajamento-influenciadores/
├── data/
│   └── influencers.csv              # Dataset original
├── docs/
│   ├── raw_df/                      # Visualizações dos dados brutos
│   │   ├── engagement_distribution.png
│   │   ├── correlation_matrix.png
│   │   └── boxplot_before_outliers.png
│   ├── processed_df/                # Visualizações após processamento
│   │   ├── boxplot_after_outliers.png
│   │   └── model_comparison.png
│   ├── model_results.txt            # Métricas detalhadas
│   └── Relatorio_Tecnico_RegLinear.pdf
├── src/                             # Códigos fonte
│   ├── data_visualization.py
│   ├── evaluate_model.py
│   ├── exploratory_analysis.py
│   ├── generate_report.py
│   ├── model_comparison.py
│   ├── model_visualization.py
│   ├── preprocessing.py
│   └── train_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Autores

- Matheus Queiroz
- Guilherme Oliveira

## Licença

MIT
