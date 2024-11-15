from preprocessing import load_data
from data_visualization import visualize_all

# Carregando os dados
df = load_data('data/influencers.csv')

# Função para análise exploratória
def exploratory_analysis():
    initial_data_analysis() # 1. Análise inicial dos dados
    num_values_analysis()# 2. Análise de valores nulos
    visualize_all(df, 'docs/raw_df')

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

def calculate_results():
        return {
        'total_influencers': len(df),
        'avg_engagement': df['60_day_eng_rate'].mean(),
        'median_followers': df['followers'].median(),
        'top_countries': df['country'].value_counts().head(5).to_dict(),
    }

if __name__ == "__main__":
    exploratory_analysis()
    results = calculate_results()
    print("\nResultados da análise:")
    for key, value in results.items():
        print(f"\n{key}:")
        print(value)