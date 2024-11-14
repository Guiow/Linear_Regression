import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo
plt.style.use('seaborn')

def create_comparison_plots():
    # Dados dos modelos
    models_data = {
        'Modelo': ['Linear', 'Ridge', 'Lasso'] * 3,
        'Métrica': ['R²'] * 3 + ['MSE'] * 3 + ['MAE'] * 3,
        'Valor': [0.724, 0.721, 0.715, 0.156, 0.158, 0.162, 0.312, 0.315, 0.319]
    }
    
    df = pd.DataFrame(models_data)
    
    # Criar figura com subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Gráfico de barras para todas as métricas
    sns.barplot(data=df, x='Modelo', y='Valor', hue='Métrica', ax=ax1)
    ax1.set_title('Comparação de Métricas por Modelo')
    ax1.set_ylabel('Valor')
    
    # 2. Coeficientes dos modelos
    coef_data = {
        'Feature': ['avg_likes', 'followers', 'influence_score', 'posts'],
        'Linear': [0.452, -0.328, 0.245, -0.156],
        'Ridge': [0.448, -0.325, 0.242, -0.154],
        'Lasso': [0.445, -0.320, 0.240, -0.150]
    }
    
    coef_df = pd.DataFrame(coef_data)
    coef_df.set_index('Feature', inplace=True)
    
    sns.heatmap(coef_df, annot=True, cmap='coolwarm', center=0, ax=ax2)
    ax2.set_title('Coeficientes dos Modelos por Feature')
    
    plt.tight_layout()
    plt.savefig('docs/model_comparison_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_comparison_plots()