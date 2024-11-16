import matplotlib.pyplot as plt
import numpy as np

def visualize_heterosterasticity_grapth(y_test, y_pred, title):
    residuals = y_test - y_pred

    # Calcula o desvio padrão dos residuos
    std_residuals = np.std(residuals)
    interval = 1.96
    
    # Cria intervalo de confiança
    lower_bound = -interval * std_residuals
    upper_bound = interval * std_residuals

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, label='Residuals')
    plt.axhline(0, color='red', linestyle='--', label='Zero Line')
    plt.axhline(lower_bound, color='blue', linestyle='--', label=f'Lower {interval:.2f}')
    plt.axhline(upper_bound, color='blue', linestyle='--', label=f'Upper {interval:.2f}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f"Homoscedasticity {title}")
    plt.legend()
    plt.savefig(f"docs/Homoscedasticity{title}")
    plt.close()

def plot_model_comparison(results):
    models = list(results.keys())
    metrics = ['R²', 'MSE', 'MAE']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(['r2', 'mse', 'mae']):
        values = [results[model][metric] for model in models]
        
        # Criar as barras
        bars = axes[i].bar(range(len(models)), values)
        
        # Configurar os ticks
        axes[i].set_xticks(range(len(models)))
        axes[i].set_xticklabels(models, rotation=45)
        
        # Adicionar título
        axes[i].set_title(metrics[i])
        
        # Adicionar valores sobre as barras
        for bar in bars:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('docs/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()