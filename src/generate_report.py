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
    story.append(Image('docs/raw_df/boxplot_before_outliers.png', width=400, height=300))

    # Topico Dados Apos Pre-processamento -----------------
    story.append(Paragraph("Dados Após Pré-processamento", styles['Heading3']))
    story.append(Paragraph("1° Nova distribuição da variável dependente", styles['Heading4']))
    story.append(Paragraph("""Após a remoção de outliers e a normalização dos dados a nova distribuição
        ficou mais clara. O pico inicial de dados foi reduzido, e a distribuição ficou mais suave
        nos evidenciando que temos muito mais dados com taxa de engajamento de porcentagem baixa doque 
        de porcentagem alta."""))
    story.append(Image('docs/processed_df/engagement_distribution.png', width=370, height=260))

    story.append(Paragraph("2° Nova distribuição e relação entre variáveis", styles['Heading4']))
    story.append(Paragraph("""Conseguimos evidenciar mais ainda a relação linear entre a variável independente 
        new_post_avg_like e a variável dependente 60_day_eng_rate, oque é ótimo para precisão do nosso modelo.
        Percebe-se também, que qualse todas as distribuições de variáveis seguem o mesmo padrão com
        muitos valores baixos e menos valores altos, exceto pela variável independente influence_score, que segue
        uma distribuição mais central. As variáveis rank, country e channel_info foram removidas por 
        terem relação qualse nula com a variável dependente."""))
    story.append(Image('docs/processed_df/relation_between_variables.png', width=450, height=450))

    story.append(Paragraph("3° Nova Matriz de Correlação", styles['Heading4']))
    story.append(Paragraph("""Percebe-se que a relação entre a variável dependente 60_day_eng_rate e as variáveis independentes
        new_post_avg_like e avg_likes permanecem da mesma forma. Entretanto, conseguimos reduzir a correlação entre
        new_post_avg_like e avg_likes para um nível aceitável, abaixo de 0.75. Decidimos por não usar tecnicas de redução
        de dimensionalidade em variáveis independetes com correlação abaixo de 0.8, e também optamos por não fundir
        as duas variáveis porque avg_likes também está com uma correlação considerável com a variável dependente.
        A variável independente total_likes que possuia uma correlação considerável com followers não possui mais,
        a correlação entre as duas foi bastante reduzida. Dessa forma, os dados ficaram melhores para treinar o modelo."""))
    story.append(Image('docs/processed_df/correlation_matrix.png', width=450, height=450))
    
    story.append(Paragraph("4° Distribuição das Features após tratamento de Outliers", styles['Heading4']))
    story.append(Paragraph("""Aqui ja temos uma redução considerável na quantidade de outliers em todas as variáveis 
        independentes citadas anteriormente, principalmente na  new_post_avg_like que apenas sobrou um outlier. Dessa forma,
        nosso modelo não será tão afetado pelos outliers no data set."""))
    story.append(Image('docs/processed_df/boxplot_after_outliers.png', width=400, height=300))
    story.append(Spacer(1, 12))
    
    # Titulo Implementacao do Algoritmo -----------------------
    story.append(Paragraph("Implementação do Algoritmo", styles['Heading2']))
    story.append(Paragraph("""O algoritmo foi implementado usando 3 modelos diferentes da biblioteca 
        sklearn.linear_model sendo eles o modelo regular de regressão linear, o ~de Lasso e o de Ridge.
        Eles foram treinados com os mesmos dados para comparação posteriormente. Depois de alguns testes decidimos que
        0.75% é o valor ideal para treinamento do nosso modelo. E claro, todos os modelos foram treinados com dados
        preprocessados."""))
    
    # Titulo Validacao e Ajuste de Hiperparametros ------------
    story.append(Paragraph("Validação e Ajuste de Hiperparâmetros", styles['Heading2']))

    # Topico Escolha das variaveis ----------------------------
    story.append(Paragraph("Escolha das variaveis", styles['Heading3']))
    story.append(Paragraph("""As variáveis independentes foram escolhidas por dois principais motivos, 1° forte correlação com
        a variável dependente, 2° baixa correlação entre as outras variáveis independentes. Dessa forma temos, que as variáveis
        independentes como influence_score, followers e total_likes, atendem o primeiro critério, enquanto as avg_likes e 
        new_post_arg_like atendem o segundo e por fim a variável independente post que é um meio termo entre o primeiro e segundo.
        Variáveis como rank, country e channel_info possuiam uma correlação com a variável dependente muito baixa, por isso
        foram removidas e como elas não eram correlacionadas também, tecnicas de redução de dimensionalidade não iriam ajudar."""))
    
    #Topico Validacao dos Modelos ------------------------------------------
    story.append(Paragraph("Validação dos Modelos", styles['Heading3']))
    story.append(Paragraph("""Para validar os modelos foram usadas métricas de validação comuns como R², MSE e MAE.
        E para evitar o maximo de overfitting usamos validação cruzada com o R², e calculamos a média e o desvio padrão,
        da validação cruzada, para análise."""))
    
    #Topico Otimizacao dos Modelos
    story.append(Paragraph("Otimização dos Modelos", styles['Heading3']))
    story.append(Paragraph("""Para otimizar todos os modelos, além de ajustar os hiperparametros nós deveriamos otimizar a base de dados, e assim fizemos.
        A base de dados foi melhorada com os seguintes passos:"""))
    
    story.append(Paragraph("    - Garantimos que todos os dados estão normalizados.", styles['Heading3']))

    story.append(Paragraph("    - Evitamos a multicolineariedade entre variáveis independentes muito correlacionadas.", styles['Heading3']))
    story.append(Paragraph("""Entretando, permitimos que avg_like e new_post_avg_like pois ambas compartilhavam uma correlação de alto valor
        com a variável dependente."""))
    
    story.append(Paragraph("    - Ajustamos o hiperparamêtro alpha de Ridge e Lasso.", styles['Heading3']))
    story.append(Paragraph("""Para o modelo Ridge um hiperparamêtro alpha = 1 consideramos ideal, penalizando moderadamente
        os coeficientes do modelo. E para o modelo de Lasso, alpha = 0.01 ja é o suficiente para zerar alguns coeficientes e
        e fornecer boas métricas de desvio padrão e média do R² da validação cruzada."""))

    story.append(Paragraph("    - Evitamos que os dados sofram de heteroscedasticidade.", styles['Heading3']))
    story.append(Paragraph("""Percebe-se que temos pouquissimos dados que saem do intervalo de confiança dos modelos, 
        e ainda assim os valores que saem do intervalo de confiança não são extrapolantes, dessa forma eles podem ser aceitos."""))
    story.append(Image('docs/HomoscedasticityLinear Regression.png', width=400, height=300))
    story.append(Image('docs/HomoscedasticityRidge.png', width=400, height=300))
    story.append(Image('docs/HomoscedasticityLasso.png', width=400, height=300))

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