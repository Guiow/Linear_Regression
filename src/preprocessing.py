from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_preprocessed_data(file_path):
    """Carrega os dados com preprocessamento"""
    df = load_data(file_path)
    df_processed = preprocess_data(df)
    return df_processed

def load_data(file_path):
    """Carrega os dados"""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Pipeline completo de pré-processamento"""
    df = convert_metrics(df) # Converter colunas com 'k', 'm', 'b', '%' para valores numéricos
    df = drop_unecessary_columns(df) # Remover colunas não necessárias para o modelo
    print(df)
    df = treat_null_values(df)
    print(df)
    df = normalize_features(df) # Normalizar features
    return df

def convert_metrics(df):
    """Converte métricas para formato numérico"""
    
    def convert_to_number(value):
        if pd.isna(value):
            return value
        return calculate_conversion(value)
    
    cols = ['posts', 'followers', 'avg_likes', 'total_likes', '60_day_eng_rate']
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
    columns_to_drop = ['rank', 'channel_info', 'new_post_avg_like']
    return df.drop(columns=columns_to_drop)

def treat_null_values(df):
    df['country'] = df['country'].fillna('Unknown')
    return df.dropna() # Tratar valores ausentes

def normalize_features(df):
    scaler = StandardScaler()
    features = ['influence_score', 'posts', 'followers', 'avg_likes', 'total_likes',]
    df[features] = scaler.fit_transform(df[features])
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