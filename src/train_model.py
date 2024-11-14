import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

class EngagementModel:
    def __init__(self, data_path='data/influencers.csv'):
        self.df = pd.read_csv(data_path)
        self.features = ['influence_score', 'posts', 'followers', 'avg_likes']
        self.target = '60_day_eng_rate'
        
    def preprocess_data(self):
        # Remove linhas com valores nulos
        self.df = self.df.dropna(subset=self.features + [self.target])
        
        # Normalização dos dados
        scaler = StandardScaler()
        X = scaler.fit_transform(self.df[self.features])
        y = self.df[self.target]
        
        # Split dos dados
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
            # Treinar modelo
            model.fit(X_train, y_train)
            
            # Predições
            y_pred = model.predict(X_test)
            
            # Métricas
            results[name] = {
                'r2': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'coef': dict(zip(self.features, model.coef_))
            }
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            results[name]['cv_mean'] = cv_scores.mean()
            results[name]['cv_std'] = cv_scores.std()
            
            # Plot real vs predito
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Valor Real')
            plt.ylabel('Valor Predito')
            plt.title(f'Real vs Predito - {name}')
            plt.savefig(f'docs/{name.lower().replace(" ", "_")}_predictions.png')
            plt.close()
        
        return results

if __name__ == "__main__":
    model = EngagementModel()
    results = model.train_models()
    
    # Salvar resultados
    with open('docs/model_results.txt', 'w') as f:
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")