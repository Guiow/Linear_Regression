from sklearn.preprocessing import StandardScaler
from data_visualization import visualize_all, visualize_boxplots
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def load_preprocessed_data(file_path):
    """Carrega os dados com preprocessamento"""
    df = load_data(file_path)
    df_processed = preprocess_data(df)
    return df_processed

def load_data(file_path):
    """Carrega os dados"""
    df = pd.read_csv(file_path)
    df = convert_metrics(df) # Converter colunas com 'k', 'm', 'b', '%' para valores numéricos
    return df

def preprocess_data(df):
    """Pipeline completo de pré-processamento"""
    print("Iniciando pré-processamento dos dados...")
    print(f"Número inicial de registros: {len(df)}")

    df = drop_unecessary_columns(df) # Remover colunas não necessárias para o modelo
    df = treat_null_values(df)
    df = normalize_features(df) # Normalizar features
    df = scale_features(df)
    #df = merge_features(df)
    
    visualize_boxplots(df, 'boxplot_before_outliers')
    df = handle_outliers(df, 2)
    visualize_boxplots(df, 'boxplot_after_outliers')

    print(f"Número final de registros: {len(df)}")
    return df

def merge_features(df):
    pca = PCA(n_components=1)

    df['followers_total_likes'] = pca.fit_transform(df[[ 'followers', 'total_likes']])
    df = df.drop(['followers', 'total_likes'], axis=1)

    df['combined_likes'] = pca.fit_transform(df[[ 'avg_likes', 'new_post_avg_like']])
    df = df.drop(['avg_likes', 'new_post_avg_like'], axis=1)
    return df

def convert_metrics(df):
    """Converte métricas para formato numérico"""

    def convert_to_number(value):
        if pd.isna(value):
            return value
        return calculate_conversion(value)
    
    cols = ['posts', 'followers', 'avg_likes', 'total_likes', '60_day_eng_rate', 'new_post_avg_like']
    for col in cols:
        df[col] = df[col].apply(convert_to_number)
    return df

def calculate_conversion(value):
    multipliers = {'k': 1e3, 'm': 1e6, 'b': 1e9, '%' : 0.01}
    for suffix, multiplier in multipliers.items():
        if suffix in value:
            return float(value.replace(suffix, '')) * multiplier
    return float(value)

def drop_unecessary_columns(df):
    columns_to_drop = ['rank', 'channel_info']
    return df.drop(columns=columns_to_drop)

def treat_null_values(df):
    df['country'] = df['country'].fillna('Unknown')
    return df.dropna() # Tratar valores ausentes

def normalize_features(df):
    scaler = StandardScaler()
    features = df.select_dtypes(include=[np.number]).columns
    df[features] = scaler.fit_transform(df[features])
    return df

def scale_features(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def handle_outliers(df, threshold):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < threshold]
    return df

if __name__ == "__main__":
    # Teste do pipeline
    file_path = 'data/influencers.csv'

    #Teste df sem processamento
    df = load_data(file_path)
    print(f"Dataset carregado: {df.shape[0]} registros")

    #Teste df com processamento
    df_processed = load_preprocessed_data(file_path)
    print(f"Dataset processado: {df_processed.shape[0]} registros")
    print("\nAmostras dos dados processados:")
    print(df_processed)
    visualize_all(df_processed, 'docs/processed_df')
    