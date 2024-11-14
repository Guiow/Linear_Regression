import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.features = ['influence_score', 'posts', 'followers', 'avg_likes']
        self.target = '60_day_eng_rate'
        
    def handle_missing_values(self):
        missing_stats = self.df.isnull().sum()
        print("Valores ausentes por coluna:")
        print(missing_stats)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        self.df[categorical_cols] = self.df[categorical_cols].fillna(self.df[categorical_cols].mode().iloc[0])
        
        return self.df
    
    def handle_outliers(self, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        plt.figure(figsize=(15, 5))
        self.df[self.features].boxplot()
        plt.title('Distribuição das Features antes do tratamento de Outliers')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('docs/boxplot_before_outliers.png')
        plt.close()
        
        for col in numeric_cols:
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            self.df = self.df[z_scores < threshold]
        
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
        scaler = StandardScaler()
        self.df[self.features] = scaler.fit_transform(self.df[self.features])
        return self.df

    def process(self):
        print("Iniciando pré-processamento dos dados...")
        print(f"Número inicial de registros: {len(self.df)}")
        
        self.df = self.handle_missing_values()
        self.df = self.handle_outliers()
        self.df = self.scale_features()
        
        print(f"Número final de registros: {len(self.df)}")
        return self.df

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
        
    def train_models(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results[name] = {
                'r2': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'coef': dict(zip(self.features, model.coef_))
            }
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            results[name]['cv_mean'] = cv_scores.mean()
            results[name]['cv_std'] = cv_scores.std()
            
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Valor Real')
            plt.ylabel('Valor Predito')
            plt.title(f'Real vs Predito - {name}')
            plt.savefig(f'docs/{name.lower().replace(" ", "_")}_predictions.png')
            plt.close()
        
        return results

def plot_model_comparison(results):
    models = list(results.keys())
    metrics = ['R²', 'MSE', 'MAE']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(['r2', 'mse', 'mae']):
        values = [results[model][metric] for model in models]
        
        axes[i].bar(models, values)
        axes[i].set_title(metrics[i])
        axes[i].set_xticklabels(models, rotation=45)
    
    plt.tight_layout()
    plt.savefig('docs/model_comparison.png')
    plt.close()

if __name__ == "__main__":
    model = EngagementModel()
    results = model.train_models()
    plot_model_comparison(results)
    
    with open('docs/model_results.txt', 'w') as f:
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")