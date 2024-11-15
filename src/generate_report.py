from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_JUSTIFY

def create_report():
    doc = SimpleDocTemplate(
        "docs/Relatorio_Tecnico_RegLinear.pdf",
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Justify',
        alignment=TA_JUSTIFY
    ))
    
    story = []
    
    # Título
    title = Paragraph(
        "Relatório Técnico: Implementação e Análise do Algoritmo de Regressão Linear",
        styles['Heading1']
    )
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Resumo
    story.append(Paragraph("Resumo", styles['Heading2']))
    story.append(Paragraph(
        """O projeto visa implementar um modelo preditivo para estimar a taxa de engajamento de influenciadores
        em diversas regiões do mundo. A previsão será realizada com base em variáveis independentes que estão fortemente
        correlacionadas com a variável dependente, ou em casos onde as variáveis não apresentam correlação entre si.
        O objetivo principal é criar um sistema eficiente de previsão, usando técnicas de regressão e análise de dados.""",
        styles['Justify']
    ))
    story.append(Spacer(1, 12))
    
    # Metodologia
    story.append(Paragraph("Metodologia", styles['Heading2']))
    story.append(Paragraph(
        """O projeto utilizou três variações de modelos de regressão linear para prever a taxa de engajamento dos
        influenciadores: regressão linear simples, Ridge e Lasso. O processo de modelagem incluiu a análise e o
        pré-processamento dos dados, onde foram realizadas etapas de remoção de outliers para garantir a qualidade
        dos dados e minimizar a influência de valores extremos no modelo. Além disso, as variáveis independentes foram
        normalizadas para garantir que todas as features tivessem uma escala similar, o que favorece o desempenho dos modelos
        de regressão. A seleção dos modelos foi feita com base na eficiência de cada um para lidar com dados altamente
        correlacionados e na capacidade de regularização de Ridge e Lasso para evitar overfitting.""",
        styles['Justify']
    ))
    story.append(Image('docs/boxplot_before_outliers.png', width=400, height=300))
    story.append(Spacer(1, 12))
    
    # Adicionar imagens
    story.append(Paragraph("Análise de Dados", styles['Heading2']))
    story.append(Paragraph("Distribuição das Features antes tratamento de Outliers", styles['Heading4']))
    story.append(Paragraph("""Percebe-se uma vasta quantidade de outliers em métricas como followers, avg_likes, new_post_avg_like e 
        na variável dependente 60_day_eng_rate. Esses outliers devem ser tratados da melhor forma possível para que o modelo 
        seja menos enviesado pelo overfitting. """))
    story.append(Image('docs/boxplot_before_outliers.png', width=400, height=300))
    
    story.append(Paragraph("Distribuição das Features após tratamento de Outliers", styles['Heading4']))
    story.append(Paragraph("""Aqui ja temos uma redução considerável na quantidade de outliers em todas as variáveis 
        independentes citadas anteriormente, principalmente na  new_post_avg_like que apenas sobrou um outlier. Dessa forma,
        nosso modelo não será tão afetado pelos outliers no data set."""))
    story.append(Image('docs/boxplot_after_outliers.png', width=400, height=300))
    story.append(Spacer(1, 12))
    
    # Resultados
    story.append(Paragraph("Resultados", styles['Heading2']))
    story.append(Image('docs/model_comparison.png', width=400, height=300))
    
    # Ler resultados do arquivo
    with open('docs/model_results.txt', 'r') as f:
        results = f.read()
    story.append(Paragraph(results, styles['Normal']))
    
    # Conclusão
    story.append(Paragraph("Conclusão", styles['Heading2']))
    story.append(Paragraph(
        """Com base nos resultados obtidos, o modelo de Regressão Linear 
        demonstrou melhor performance para a predição de taxas de engajamento, 
        apresentando um equilíbrio adequado entre complexidade e acurácia.""",
        styles['Justify']
    ))
    
    doc.build(story)

if __name__ == '__main__':
    create_report()