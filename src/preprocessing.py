import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Carrega e realiza limpeza inicial dos dados"""
    df = pd.read_csv(file_path)
    print(f"Dataset carregado: {df.shape[0]} registros")
    return df

def convert_metrics(df):
    """Converte métricas para formato numérico"""
    # Converter colunas com 'k', 'm', 'b' para valores numéricos
    cols = ['posts', 'followers', 'avg_likes', 'total_likes']
    
    def convert_to_number(value):
        if pd.isna(value):
            return value
        if isinstance(value, (int, float)):
            return value
        
        value = str(value).lower()
        multipliers = {'k': 1e3, 'm': 1e6, 'b': 1e9}
        
        for suffix, multiplier in multipliers.items():
            if suffix in value:
                try:
                    return float(value.replace(suffix, '')) * multiplier
                except ValueError:
                    return np.nan
        return float(value)

    for col in cols:
        df[col] = df[col].apply(convert_to_number)
    
    return df

def clean_engagement_rate(df):
    """Limpa e converte taxa de engajamento"""
    df['60_day_eng_rate'] = df['60_day_eng_rate'].str.rstrip('%').astype(float)
    return df

def preprocess_data(df):
    """Pipeline completo de pré-processamento"""
    df = convert_metrics(df)
    df = clean_engagement_rate(df)
    
    # Remover colunas não necessárias para o modelo
    columns_to_drop = ['rank', 'channel_info', 'new_post_avg_like', 'country']
    df = df.drop(columns=columns_to_drop)
    
    # Tratar valores ausentes
    df = df.dropna()
    
    # Normalizar features
    scaler = StandardScaler()
    features = ['influence_score', 'posts', 'followers', 'avg_likes']
    df[features] = scaler.fit_transform(df[features])
    
    return df

if __name__ == "__main__":
    # Teste do pipeline
    df = load_data('data/influencers.csv')
    df_processed = preprocess_data(df)
    print(f"Dataset processado: {df_processed.shape[0]} registros")
    print("\nAmostras dos dados processados:")
    print(df_processed.head())