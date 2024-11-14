import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.features = ['influence_score', 'posts', 'followers', 'avg_likes']
        self.target = '60_day_eng_rate'
        
    def handle_missing_values(self):
        # Análise de valores ausentes
        missing_stats = self.df.isnull().sum()
        print("Valores ausentes por coluna:")
        print(missing_stats)
        
        # Para colunas numéricas, preencher com mediana
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        # Para colunas categóricas, preencher com moda
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        self.df[categorical_cols] = self.df[categorical_cols].fillna(self.df[categorical_cols].mode().iloc[0])
        
        return self.df
    
    def handle_outliers(self, threshold=3):
        """
        Remove outliers usando o método Z-score
        threshold: número de desvios padrão para considerar outlier
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Plot boxplots antes da remoção
        plt.figure(figsize=(15, 5))
        self.df[self.features].boxplot()
        plt.title('Distribuição das Features antes do tratamento de Outliers')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('docs/boxplot_before_outliers.png')
        plt.close()
        
        # Remover outliers usando Z-score
        for col in numeric_cols:
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            self.df = self.df[z_scores < threshold]
        
        # Plot boxplots depois da remoção
        plt.figure(figsize=(15, 5))
        self.df[self.features].boxplot()
        plt.title('Distribuição das Features após tratamento de Outliers')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('docs/boxplot_after_outliers.png')
        plt.close()
        
        print(f"Registros removidos: {len(self.df) - self.df.shape[0]}")
        return self.df
    
    def scale_features(self):
        """
        Normaliza as features numéricas
        """
        scaler = StandardScaler()
        self.df[self.features] = scaler.fit_transform(self.df[self.features])
        return self.df

    def process(self):
        """
        Executa todo o pipeline de pré-processamento
        """
        print("Iniciando pré-processamento dos dados...")
        print(f"Número inicial de registros: {len(self.df)}")
        
        self.df = self.handle_missing_values()
        self.df = self.handle_outliers()
        self.df = self.scale_features()
        
        print(f"Número final de registros: {len(self.df)}")
        return self.df

# Atualização da classe EngagementModel para usar o DataPreprocessor
class EngagementModel:
    def __init__(self, data_path='data/influencers.csv'):
        self.df = pd.read_csv(data_path)
        self.features = ['influence_score', 'posts', 'followers', 'avg_likes']
        self.target = '60_day_eng_rate'
        
    def preprocess_data(self):
        preprocessor = DataPreprocessor(self.df)
        self.df = preprocessor.process()
        
        X = self.df[self.features]
        y = self.df[self.target]
        
        return train_test_split(X, y, test_size=0.2, random_state=42)