# Análise de Engajamento de Influenciadores no Instagram

## Descrição

Projeto de análise preditiva usando Regressão Linear para modelar e prever taxas de engajamento de influenciadores no Instagram. O modelo analisa métricas como número de seguidores, posts e likes para prever o engajamento dos usuários.

## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/analise-engajamento-influenciadores
cd analise-engajamento-influenciadores

# Crie e ative o ambiente virtual
python -m virtualenv venv
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

## Estrutura

```
analise-engajamento-influenciadores/
├── data/
│   └── influencers.csv          # Dataset com métricas dos influenciadores
├── docs/
│   ├── boxplot_before_outliers.png  # Visualização antes do tratamento
│   ├── boxplot_after_outliers.png   # Visualização após tratamento
│   ├── model_comparison.png         # Comparação de performance
│   ├── model_results.txt            # Métricas detalhadas
│   └── Relatorio_Tecnico_RegLinear.pdf  # Documentação completa
├── src/
│   ├── exploratory_analysis.py      # Análise exploratória
│   ├── preprocessing.py             # Pré-processamento
│   ├── train_model.py              # Treinamento dos modelos
│   └── evaluate_model.py           # Avaliação de performance
├── .gitignore
├── README.md
└── requirements.txt
```

## Execução

```bash
# Análise exploratória
python src/exploratory_analysis.py

# Treinamento e avaliação dos modelos
python src/train_model.py

# Avaliação detalhada
python src/evaluate_model.py

# Gerar relatório
python src/generate_report.py
```

## Funcionalidades

- Pré-processamento automático de dados
  - Tratamento de valores ausentes
  - Remoção de outliers (Z-score)
  - Normalização de features
- Implementação de três modelos:
  - Regressão Linear
  - Ridge Regression
  - Lasso Regression
- Avaliação comparativa com métricas:
  - R² (coeficiente de determinação)
  - MSE (erro quadrático médio)
  - MAE (erro absoluto médio)
- Visualizações:
  - Distribuição de features
  - Comparação de modelos
  - Predições vs valores reais

## Tecnologias

- Python 3.12
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

## Autor

- Matheus Queiroz
- Guilherme Oliveira
