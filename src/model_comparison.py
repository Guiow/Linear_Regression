import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_comparison_plots(results):
    """Cria um grafico de comparacao do modelo"""
    # Dados dos modelos
    models_data = {
        'Modelo': ['Linear', 'Ridge', 'Lasso'] * 5,
        'Métrica': ['R²'] * 3 + ['MSE'] * 3 + ['MAE'] * 3 + ['CrossV_mean'] * 3 + ['CrossV_std']  * 3,
        'Valor': format_metrics_results_values(results)
    }

    df = pd.DataFrame(models_data)
    
    # Criar figura com subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Gráfico de barras para todas as métricas
    sns.barplot(data=df, x='Modelo', y='Valor', hue='Métrica', ax=ax1)
    ax1.set_title('Comparação de Métricas por Modelo')
    ax1.set_ylabel('Valor')
    

    features_name = results['Linear Regression']['coef'].keys()
    # 2. Coeficientes dos modelos
    coef_data = {
        'Feature': features_name,
        'Linear': format_coefficient_results_values(results, 'Linear Regression', features_name),
        'Ridge': format_coefficient_results_values(results, 'Ridge', features_name),
        'Lasso': format_coefficient_results_values(results, 'Lasso', features_name)
    }
    
    coef_df = pd.DataFrame(coef_data)
    coef_df.set_index('Feature', inplace=True)
    
    sns.heatmap(coef_df, annot=True, cmap='coolwarm', center=0, ax=ax2)
    ax2.set_title('Coeficientes dos Modelos por Feature')
    
    plt.tight_layout()
    plt.savefig('docs/model_comparison_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

def format_metrics_results_values(results):
    """Formata os valores de metrica dos modelos para serem visualizados no grafico"""
    models_names = ['Linear Regression', 'Ridge', 'Lasso']
    metrics = ['r2', 'mse', 'mae', 'cv_mean', 'cv_std']
    values = [results[mn][metric] for metric in metrics for mn in models_names]
    return values

def format_coefficient_results_values(result, model_name, features_name):
    """Formata os valores dos coeficientes das features para serem visualizados no grafico"""
    features_coefficient_values = [result[model_name]['coef'][feature] for feature in features_name]
    return features_coefficient_values