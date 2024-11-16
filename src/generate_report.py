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
    story.append(Spacer(1, 12))
    
    # Titulo Analise Exploratoria -----------------------
    story.append(Paragraph("Análise Exploratória", styles['Heading2']))

    # Topico conhecendo os dados ------------------------
    story.append(Paragraph("Conhecendo os Dados", styles['Heading3']))
    story.append(Paragraph("1° Analisamos a distribuição da nossa variável dependente", styles['Heading4']))
    story.append(Paragraph("""Percebe-mos alguns outliers com taxa de engajamento acima de 25%, no qual
        é muito alto comparado ao resto dos dados. Devemos tratar isso."""))
    story.append(Image('docs/raw_df/engagement_distribution.png', width=370, height=260))

    story.append(Paragraph("2° Analisamos a distribuição e relação de cada uma das variáveis", styles['Heading4']))
    story.append(Paragraph("""Percebe-se que a variável independente rank segue uma distribuição qualse constante, e tem uma
        relação não linear com a váriavel dependente. A variável independente new_post_avg_like, segue uma distribuição bastante
        semelhante ao da nossa variável dependente, e de uma forma não tão clara, parece se relacionar com a variável dependente.
        de forma linear. E também, a variável new_post_avg_like parece se relacionar de forma qualse linear com avg_likes. Outra
        observação, é praticamente todas as variáveis exceto rank, possuem outliers que devem ser tratados para melhorar
        o desempenho do nosso modelo.
        """))
    story.append(Image('docs/raw_df/relation_between_variables.png', width=450, height=450))

    story.append(Paragraph("3° Analisamos a matriz de correlação", styles['Heading4']))
    story.append(Paragraph("""Como esperado a variável rank possui correlação qualse nula
        com a variável dependente oque nos permite remover tranquilamente essa variável, da mesma forma as variáveis independentes
        channel_info e country foram removidas pelo mesmo motivo em testes anteriores. E percebemos uma forte correlação entre
        a variável dependente 60_day_eng_rate, e duas variáveis independentes new_post_avg_like e avg_likes. A correlação entre
        as variáveis independentes new_post_avg_like e avg_likes também está alta, devemos reduzi-la no preprocessamento. A variável
        followers também está um pouco correlacionada com total_likes, mas nada muito alarmante."""))
    story.append(Image('docs/raw_df/correlation_matrix.png', width=450, height=450))
    
    story.append(Paragraph("4° Distribuição das Features antes tratamento de Outliers", styles['Heading4']))
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