import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Configuração para exibir gráficos em estilo mais moderno
plt.style.use('seaborn')

# Carregando os dados
df = pd.read_csv('data/influencers.csv')

# Função para análise exploratória
def exploratory_analysis(df):
    # 1. Análise inicial dos dados
    print("Dimensões do dataset:", df.shape)
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    print("\nInformações do dataset:")
    print(df.info())
    print("\nEstatísticas descritivas:")
    print(df.describe())
    
    # 2. Análise de valores nulos
    print("\nValores nulos por coluna:")
    print(df.isnull().sum())
    
    # 3. Visualizações
    
    # 3.1 Distribuição da taxa de engajamento
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='60_day_eng_rate', bins=30)
    plt.title('Distribuição da Taxa de Engajamento')
    plt.xlabel('Taxa de Engajamento (%)')
    plt.ylabel('Frequência')
    plt.savefig('docs/engagement_distribution.png')
    plt.close()
    
    # 3.2 Relação entre seguidores e engajamento
    plt.figure(figsize=(10, 6))
    plt.scatter(df['followers'], df['60_day_eng_rate'], alpha=0.5)
    plt.xscale('log')
    plt.title('Relação entre Número de Seguidores e Taxa de Engajamento')
    plt.xlabel('Número de Seguidores (log)')
    plt.ylabel('Taxa de Engajamento (%)')
    plt.savefig('docs/followers_vs_engagement.png')
    plt.close()
    
    # 3.3 Top 10 países com mais influenciadores
    plt.figure(figsize=(12, 6))
    df['country'].value_counts().head(10).plot(kind='bar')
    plt.title('Top 10 Países com Mais Influenciadores')
    plt.xlabel('País')
    plt.ylabel('Número de Influenciadores')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('docs/top_countries.png')
    plt.close()
    
    # 3.4 Matriz de correlação
    correlation_matrix = df[['influence_score', 'posts', 'followers', 
                           'avg_likes', '60_day_eng_rate']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.savefig('docs/correlation_matrix.png')
    plt.close()
    
    return {
        'total_influencers': len(df),
        'avg_engagement': df['60_day_eng_rate'].mean(),
        'median_followers': df['followers'].median(),
        'top_countries': df['country'].value_counts().head(5).to_dict(),
        'correlation_matrix': correlation_matrix
    }

if __name__ == "__main__":
    results = exploratory_analysis(df)
    print("\nResultados da análise:")
    for key, value in results.items():
        print(f"\n{key}:")
        print(value)