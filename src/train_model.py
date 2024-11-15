import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from preprocessing import load_preprocessed_data
from model_visualization import plot_model_comparison

class EngagementModel:
    def __init__(self, data_path='data/influencers.csv'):
        self.df = load_preprocessed_data(data_path)
        self.features = self.df.select_dtypes(include=[np.number]).drop('60_day_eng_rate', axis=1).columns
        self.target = '60_day_eng_rate'
        
    def preprocess_data(self):
        X = self.df[self.features]
        y = self.df[self.target]
        return train_test_split(X, y, test_size=0.25, random_state=42)
        
    def train_models(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01)# !! Esse só fica bom com alpha baixo não sei porque ainda
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

            plot_regularization_tec(name, y_test, y_pred)
        return results
    
def plot_regularization_tec(name, y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predito')
    plt.title(f'Real vs Predito - {name}')
    plt.savefig(f'docs/{name.lower().replace(" ", "_")}_predictions.png')
    plt.close()

def save_results(results):
    """Salva os resultados dos modelos em um arquivo texto"""
    with open('docs/model_results.txt', 'w') as f:
        f.write("RESULTADOS DOS MODELOS\n")
        f.write("=====================\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * (len(model_name) + 1) + "\n")
            
            # Métricas principais
            f.write(f"R²: {metrics['r2']:.4f}\n")
            f.write(f"MSE: {metrics['mse']:.4f}\n")
            f.write(f"MAE: {metrics['mae']:.4f}\n")
            f.write(f"CV Score (média): {metrics['cv_mean']:.4f}\n")
            f.write(f"CV Score (desvio): {metrics['cv_std']:.4f}\n")
            
            # Coeficientes
            f.write("\nCoeficientes:\n")
            for feature, coef in metrics['coef'].items():
                f.write(f"{feature}: {coef:.4f}\n")
            
            f.write("\n" + "="*50 + "\n")

if __name__ == "__main__":
    model = EngagementModel()
    results = model.train_models()
    plot_model_comparison(results)
    save_results(results)