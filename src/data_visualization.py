import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_all(df, docs_path):
    """Para visualizar os dados em diferentes condições"""
    eng_rate_distribution(df, docs_path)# 3.1 Distribuição da taxa de engajamento
    relation_between_variables(df, docs_path)# 3.2 Relação entre seguidores e engajamento
    top_ten_contry_most_influencers(df, docs_path) #3.3 Top 10 países com mais influenciadores
    correlation_matrix(df, docs_path)  # 3.4 Matriz de correlação

def eng_rate_distribution(df, docs_path):
    """Para visualizar a distribuição da variavel dependente"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='60_day_eng_rate', bins=30)
    plt.title('Distribuição da Taxa de Engajamento')
    plt.xlabel('Taxa de Engajamento')
    plt.ylabel('Frequência')
    plt.savefig(f'{docs_path}/engagement_distribution.png')
    plt.close()

def relation_between_variables(df, docs_path):
    """Para visualizar a distribuição e relação entre todas as variaveis"""
    plt.figure(figsize=(10, 6))
    sns.pairplot(df)
    plt.savefig(f'{docs_path}/relation_between_variables.png')
    plt.close()

def top_ten_contry_most_influencers(df, docs_path):
    """Destaca os principais paises com mais influenciadores"""
    plt.figure(figsize=(12, 6))
    df['country'].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Países com Mais Influenciadores')
    plt.xlabel('País')
    plt.ylabel('Número de Influenciadores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{docs_path}/top_countries.png')
    plt.close()

def correlation_matrix(df, docs_path):
    """Para visualizar a matriz de correlacao entre todas as variaveis"""
    correlation_matrix = df.select_dtypes(include=['number']).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig(f'{docs_path}/correlation_matrix.png')
    plt.close()

def visualize_boxplots(df, file_name):
    """Para visualizar o grafico boxplot de todas as variaveis em comparacao"""
    plt.figure(figsize=(15, 5))
    df.boxplot()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'docs/{file_name}.png')
    plt.close()