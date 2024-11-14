import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_model_predictions(model_path, test_data_path):
    """Carrega modelo e faz predições"""
    model = joblib.load(model_path)
    X_test = pd.read_csv(test_data_path)
    y_pred = model.predict(X_test)
    return X_test, y_pred

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de avaliação"""
    return {
        'R²': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

def plot_residuals(y_true, y_pred, save_path='docs/residuals.png'):
    """Plota análise de resíduos"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Valores Preditos')
    plt.ylabel('Resíduos')
    plt.title('Análise de Resíduos')
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, save_path='docs/feature_importance.png'):
    """Plota importância das features"""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(model.coef_)
    })
    importance = importance.sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    importance.plot(kind='barh', x='feature', y='importance')
    plt.title('Importância das Features')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_evaluation_report(metrics, output_path='docs/evaluation_report.txt'):
    """Gera relatório de avaliação"""
    with open(output_path, 'w') as f:
        f.write("Relatório de Avaliação do Modelo\n")
        f.write("=" * 30 + "\n\n")
        
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

if __name__ == "__main__":
    # Carregar modelo e dados
    model = joblib.load('models/linear_regression_model.joblib')
    test_data = pd.read_csv('data/test_data.csv')
    
    # Fazer predições
    X_test = test_data.drop('60_day_eng_rate', axis=1)
    y_test = test_data['60_day_eng_rate']
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    metrics = calculate_metrics(y_test, y_pred)
    
    # Gerar visualizações
    plot_residuals(y_test, y_pred)
    plot_feature_importance(model, X_test.columns)
    
    # Gerar relatório
    generate_evaluation_report(metrics)