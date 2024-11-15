from preprocessing import load_preprocessed_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Carregando os dados
df = load_preprocessed_data('data/influencers.csv')

# Função para análise exploratória
def exploratory_analysis():
    initial_data_analysis() # 1. Análise inicial dos dados
    num_values_analysis()# 2. Análise de valores nulos
    visualization()# 3. Visualizações

def initial_data_analysis():
    print("Dimensões do dataset:", df.shape)
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    print("\nInformações do dataset:")
    print(df.info())
    print("\nEstatísticas descritivas:")
    print(df.describe())

def num_values_analysis():
    print("\nValores nulos por coluna:")
    print(df.isnull().sum())

def visualization():
    eng_rate_distribution()# 3.1 Distribuição da taxa de engajamento
    relation_between_followers_eng()# 3.2 Relação entre seguidores e engajamento
    top_ten_contry_most_influencers() #3.3 Top 10 países com mais influenciadores
    correlation_matrix()  # 3.4 Matriz de correlação

def eng_rate_distribution():
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='60_day_eng_rate', bins=30) # OK
    plt.title('Distribuição da Taxa de Engajamento') # !! TEMOS UM OUTLIER DE 0.25 QUE DEVEMOS TRATAR
    plt.xlabel('Taxa de Engajamento (%)')
    plt.ylabel('Frequência')
    plt.savefig('docs/engagement_distribution.png')
    plt.close()

def relation_between_followers_eng():
    plt.figure(figsize=(10, 6))
    plt.scatter(df['followers'], df['60_day_eng_rate'], alpha=0.5)# O
    plt.title('Relação entre Número de Seguidores e Taxa de Engajamento')
    plt.xlabel('Número de Seguidores')
    plt.ylabel('Taxa de Engajamento (%)')
    plt.savefig('docs/followers_vs_engagement.png')
    plt.close()

def top_ten_contry_most_influencers():
    plt.figure(figsize=(12, 6))
    df['country'].value_counts().head(10).plot(kind='bar') #!!! PERFEITO
    plt.title('Top 10 Países com Mais Influenciadores')
    plt.xlabel('País')
    plt.ylabel('Número de Influenciadores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('docs/top_countries.png')
    plt.close()

def correlation_matrix():
    correlation_matrix = df[['influence_score', 'posts', 'followers', 
                           'avg_likes', '60_day_eng_rate']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig('docs/correlation_matrix.png')
    plt.close()

def calculate_results():
        return {
        'total_influencers': len(df),
        'avg_engagement': df['60_day_eng_rate'].mean(),
        'median_followers': df['followers'].median(), # !!!! A MEDIANA DEVE SER CALCULADA ANTES DA NORMALIZAÇÃO
        'top_countries': df['country'].value_counts().head(5).to_dict(),
    }

if __name__ == "__main__":
    exploratory_analysis()
    results = calculate_results()
    print("\nResultados da análise:")
    for key, value in results.items():
        print(f"\n{key}:")
        print(value)