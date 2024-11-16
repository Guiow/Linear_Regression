import matplotlib.pyplot as plt
import numpy as np

def visualize_homosterasticity_grapth(y_test, y_pred, title):
    """Para visualizar o grafico de homosteraticidade dos mmodelos"""
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