from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
import matplotlib.pyplot as plt
import pandas as pd

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
        """Este projeto implementa modelos de regressão linear para prever taxas de 
        engajamento de influenciadores no Instagram, utilizando métricas como número 
        de seguidores, posts e likes.""",
        styles['Justify']
    ))
    story.append(Spacer(1, 12))
    
    # Metodologia
    story.append(Paragraph("Metodologia", styles['Heading2']))
    story.append(Paragraph(
        """O projeto utilizou três variações de modelos de regressão linear: 
        Linear simples, Ridge e Lasso. Os dados foram pré-processados para 
        remoção de outliers e normalização das features.""",
        styles['Justify']
    ))
    story.append(Spacer(1, 12))
    
    # Adicionar imagens
    story.append(Paragraph("Análise de Dados", styles['Heading2']))
    story.append(Image('docs/boxplot_before_outliers.png', width=400, height=300))
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