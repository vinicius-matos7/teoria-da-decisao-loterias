# Teoria da Decisão & Modelagem Preditiva de Loterias



Projeto analítico e probabilístico para modelagem de loterias brasileiras (**Mega-Sena**, **Lotofácil** e **Quina**) utilizando conceitos de **Inferência Bayesiana**, **Teoria da Decisão Estatística** e **Aprendizado de Máquina**, fundamentados na literatura dos professores **Luís Gustavo Esteves, Rafael Izbicki e Rafael Bassi Stern** (*Inferência Bayesiana: Notas de Aula*) e **Rafael Izbicki e Tiago Mendonça dos Santos** (*Aprendizado de máquina: uma abordagem estatística*).



---



## 📊 Painéis Decisórios & Avaliações Empíricas (Visualização Direta)



Abaixo estão disponibilizados diretamente os dois painéis visuais gerados pelos modelos para cada modalidade, permitindo consultar as estimativas e a validação retroativa sem necessidade de clonar o repositório ou executar os notebooks:



1. **Painel Decisório & Recomendação Oficial:** Apresenta a **Decisão Ótima de Bayes (Bilhete Oficial)**, o ranking de votos do consenso, as probabilidades a posteriori com intervalos de credibilidade (95% CI) e o gabarito diofantino de cores de Gianella.

2. **Avaliação Empírica (Acertos vs. Erros):** Mapa visual completo com dezenas preditas e sorteadas em geometria perfeitamente circular (*Zero Lookahead Bias*).

3. **Ranking de Performance & Distribuição Hipergeométrica:** Painel estatístico dedicado confrontando a assertividade de cada modelo contra a linha de corte teórica $E[X]$ e as faixas oficiais de premiação da Caixa.



---



### 🟢 1. Lotofácil



#### (A) Painel Decisório & Bilhete Oficial Recomendado

O bilhete de consenso maximiza a utilidade esperada a posteriori combinando a concordância entre os modelos analíticos:

![Painel Decisório - Lotofácil](graficos/lotofacil_dashboard_decisao.png)



#### (B) Avaliação Empírica: Estimativas dos Modelos vs. Resultado Oficial

Validação retroativa sem viés de antecipação (*Zero Lookahead Bias*):

![Avaliação de Acertos e Erros - Lotofácil](graficos/lotofacil_avaliacao_acertos_erros.png)



#### (C) Ranking de Performance & Distribuição Hipergeométrica Teórica

Comparativo estatístico de assertividade versus a linha de corte esperada $E[X]$ e faixas oficiais de premiação:

![Ranking e Distribuição Estatística - Lotofácil](graficos/lotofacil_ranking_performance.png)



---



### 🔵 2. Mega-Sena



#### (A) Painel Decisório & Bilhete Oficial Recomendado

Síntese multicritério das 6 dezenas recomendadas com base nas probabilidades a posteriori e no gabarito de cores:

![Painel Decisório - Mega-Sena](graficos/mega_sena_dashboard_decisao.png)



#### (B) Avaliação Empírica: Estimativas dos Modelos vs. Resultado Oficial

Confronto com o último sorteio apurado da Caixa e mapa visual de dezenas:

![Avaliação de Acertos e Erros - Mega-Sena](graficos/mega_sena_avaliacao_acertos_erros.png)



#### (C) Ranking de Performance & Distribuição Hipergeométrica Teórica

Comparativo estatístico de assertividade versus a linha de corte esperada $E[X]$ e faixas oficiais de premiação:

![Ranking e Distribuição Estatística - Mega-Sena](graficos/mega_sena_ranking_performance.png)



---



### 🟣 3. Quina



#### (A) Painel Decisório & Bilhete Oficial Recomendado

Bilhete de 5 dezenas ótimo sob a regra de decisão bayesiana:

![Painel Decisório - Quina](graficos/quina_dashboard_decisao.png)



#### (B) Avaliação Empírica: Estimativas dos Modelos vs. Resultado Oficial

Desempenho dos modelos individuais versus o concurso apurado da Caixa:

![Avaliação de Acertos e Erros - Quina](graficos/quina_avaliacao_acertos_erros.png)



#### (C) Ranking de Performance & Distribuição Hipergeométrica Teórica

Comparativo estatístico de assertividade versus a linha de corte esperada $E[X]$ e faixas oficiais de premiação:

![Ranking e Distribuição Estatística - Quina](graficos/quina_ranking_performance.png)



---



### 📋 Histórico Consolidado de Acertos dos Modelos

O repositório mantém um registro contínuo e auditável do desempenho retroativo de cada abordagem preditiva frente aos resultados oficiais sorteados pela Caixa Econômica Federal (*Walk-Forward Validation* estrito sem viés de antecipação):

* **Planilhas Consolidadas:** [`excel/historico_acertos_modelos.xlsx`](excel/historico_acertos_modelos.xlsx) e [`excel/historico_acertos_modelos.csv`](excel/historico_acertos_modelos.csv).

* **Comparativo de Premiações (Últimos 20 Jogos):** Painel com a apuração das faixas oficiais da Caixa disponível em [`graficos/comparativo_premiacoes_20_jogos.png`](graficos/comparativo_premiacoes_20_jogos.png).

* **Métricas Registradas por Concurso:** Dezenas previstas por cada modelo, dezenas reais sorteadas, contagem exata de acertos, taxa de aproveitamento (%) e faixa de premiação oficial obtida.



---



## 🧠 Metodologia e Fundamentação em Teoria da Decisão



### A Decisão Ótima de Bayes (Consenso Multicritério)

Sob a formulação formal da Teoria da Decisão Estatística (*Abraham Wald, 1950*), a escolha ótima sob incerteza estocástica severa corresponde à alternativa $d^*$ que maximiza a utilidade esperada com respeito à distribuição preditiva a posteriori:



$$d^* = \arg\max_{d \in \mathcal{D}} \mathbb{E}_{\theta \mid \mathcal{D}}[U(d, \theta)]$$



Ao ponderar as probabilidades a posteriori de múltiplos estimadores complementares, o método atua como uma aproximação de *Bayesian Model Averaging (BMA)*:

* **Filtro de Ruído Estocástico:** Anula variações idiossincráticas e sobreajustes de curto prazo presentes em modelos individuais isolados.

* **Redução da Variância do Estimador:** Seleciona as dezenas que apresentam maior robustez conjunta, posicionando o bilhete sistematicamente nas caudas superiores de acerto em relação à esperança combinatória teórica $\mathbb{E}[X]$.



---



### Os Métodos Analíticos Implementados



Cada notebook executa e compara **4 abordagens metodológicas**:



1. **Método 1 (Estatístico - Frequência Ponderada & Atraso):**

   * Ponderação temporal com decaimento exponencial ($w_t = 2^{-(T - t) / 100}$).

   * Métrica de atraso relativo ($R_i = d_i / \mu_{\text{gap}, i}$).

   * Subestratégias: Balanceada, Mais Atrasadas (*Overdue*) e Frequentes Recentes (*Momentum*).



2. **Método 2 (Machine Learning Multi-label Calibrado):**

   * Classificador supervisionado (*Binary Relevance* com Regressão Logística $L_2$).

   * *Features* dinâmicas de médias móveis multiescala (5, 10, 25, 50 e 100 sorteios) e atrasos correntes.

   * Validação cronológica estrita (*Zero Lookahead Bias*).



3. **Método 3 (Inferência Bayesiana & Priori Conjugada Beta-Binomial):**

   * Priori racional calibrada pela probabilidade neutra combinatória de cada jogo:

     * **Mega-Sena** ($6/60 = 0.10$): $\text{Beta}(1, 9)$

     * **Lotofácil** ($15/25 = 0.60$): $\text{Beta}(6, 4)$

     * **Quina** ($5/80 = 0.0625$): $\text{Beta}(1, 15)$

   * Atualização analítica a posteriori via modelo Beta-Binomial Global e modelo Dinâmico Adaptativo com meia-vida temporal.



4. **Método 4 (A Geometria do Acaso - Renato Gianella):**

   * Modelagem combinatória baseada nas cores e gabaritos diofantinos de partição de inteiros.

   * Restrição do espaço de busca para as partições de maior probabilidade teórica.



5. **Métodos Híbridos (Grade Combinatória Completa — 15 Métodos):**

   * **Família Gianella (Geometria do Acaso & Gabaritos Diofantinos de Cores):**

     * **Gianella + Bayes Global:** Aloca as cotas diofantinas do gabarito combinatório de Gianella selecionando as dezenas que maximizam a esperança a posteriori conjugada global $\mathbb{E}[\theta_i \mid \text{dados}]$.

     * **Gianella + Bayes Adaptativo:** Aplica o gabarito ótimo sobre as probabilidades a posteriori do modelo Bayesiano dinâmico com ponderação temporal exponencial ($\tau = 150$).

     * **Gianella + Melhor Decisão de Bayes:** Maximiza a utilidade esperada a posteriori sob a restrição combinatória do macro-estado mais provável: $\max_{a \in \mathcal{T}^*} \mathbb{E}[U(a, s) \mid \text{dados}]$, integrando o consenso bayesiano e minimização de risco/covariância.

     * **Gianella + Regressão Logística:** Preenche o gabarito diofantino selecionando as dezenas com maiores probabilidades calibradas pelo classificador supervisionado de Machine Learning.

     * **Gianella + Freq/Atraso (Momentum):** Preenche o gabarito de cores priorizando as dezenas de maior intensidade de frequência ponderada recente.

     * **Gianella + Freq/Atraso (Balanceado):** Preenche o gabarito de cores combinando o momentum recente com a pressão de atraso relativo via escore padronizado.

   * **Família Melhor Decisão de Bayes (Consenso e Teoria da Decisão sob Incerteza):**

     * **Melhor Decisão de Bayes + Regressão Logística:** *Linear Opinion Pool* combinando 50% utilidade bayesiana de consenso com 50% probabilidades calibradas de ML.

     * **Melhor Decisão de Bayes + Freq/Atraso (Momentum):** Combinação linear do consenso ótimo de Bayes com a frequência recente ponderada.

     * **Melhor Decisão de Bayes + Freq/Atraso (Balanceado):** Integração do consenso decisório ótimo com o escore composto balanceado de frequência e atraso.

   * **Família Bayes Global (Conjugada Beta-Binomial Estática):**

     * **Bayes Global + Regressão Logística:** Combinação convexa da posteriori Beta-Binomial global com probabilidades preditivas de ML.

     * **Bayes Global + Frequência/Atraso (Momentum):** Ponderação da probabilidade a posteriori estática com o momentum temporal.

     * **Bayes Global + Frequência/Atraso (Balanceado):** Combinação da posteriori global com a métrica padronizada de frequência e atraso.

   * **Família Bayes Adaptativo (Decaimento Temporal Exponencial):**

     * **Bayes Adaptativo + Regressão Logística:** Combinação da evidência temporal a posteriori com probabilidades de ML supervisionado.

     * **Bayes Adaptativo + Frequência/Atraso (Momentum):** Fusão da posteriori adaptativa dinâmica com a taxa recente de sorteios.

     * **Bayes Adaptativo + Frequência/Atraso (Balanceado):** Combinação da evidência bayesiana dinâmica com o equilíbrio entre frequência e atraso.



6. **Métodos Ensemble (Aurélien Géron, Cap. 7 & Jerome H. Friedman — 18 Métodos):**
   * **Ensemble Bagging (Bootstrap Aggregating):** Gera réplicas com reposição (*bootstrap*) dos concursos históricos, ajustando preditores em cada partição e agregando votos via probabilidade média ou votação majoritária (*soft/hard voting*).
   * **Ensemble Pasting (Non-replacement Aggregating):** Amostragem estocástica de concursos históricos sem reposição (*Breiman, 1999*), reduzindo a dependência e variância entre estimadores.
   * **Ensemble AdaBoost (Adaptive Boosting):** Ajuste sequencial reponderando as instâncias históricas com base no erro de previsão de cada concurso ($w_i^{(m+1)} \propto w_i^{(m)} \exp(\alpha_m e_{m, i})$), forçando o modelo a focar nos sorteios mais anômalos (*Freund & Schapire, 1997*).
   * **Ensemble Gradient Boosting (Residual Boosting Padrão):** Correção sequencial do gradiente de resíduos ($r = Y - F_{m-1}$) com regularização por encolhimento (*shrinkage* $\eta$) e decaimento temporal adaptativo (*Friedman, 1999*).
   * **Ensemble Stochastic Gradient Boosting (Subsample Boosting):** A cada estágio de boosting, sorteia aleatoriamente uma fração subsample (ex.: 70%) dos concursos históricos para ajustar o vetor residual, reduzindo substancialmente a variância e mitigando sobreajuste (*Géron p. 207; Friedman, 2002*).
   * **Ensemble Logit Gradient Boosting (Log-Loss / Binomial Deviance):** Opera no espaço contínuo dos logits/log-odds com link sigmoide $\sigma(z) = \frac{1}{1 + e^{-z}}$ e resíduo como o gradiente negativo da perda de entropia cruzada binária ($y - \sigma(F)$) via passos de Newton-Raphson calibrados (*Friedman, 2001*).
   * **Ensemble Multi-Model Residual Cycling Gradient Boosting:** Gradient Boosting com preditores heterogêneos onde cada estágio de boosting corrige os resíduos empregando ciclicamente um paradigma analítico distinto do projeto (M2 ML Projeção Residual $\rightarrow$ M4 Gianella Cores $\rightarrow$ M1 Delay/Gap Momentum $\rightarrow$ M3 Bayesiano Adaptativo).
   * **Ensemble Stacking (Stacked Generalization / Blender Ridge L2):** Arquitetura em camadas (*layered ensemble*) que divide a base em treino e *holdout*, gerando matriz de meta-features e calibrando um meta-aprendiz Ridge $L_2$ para encontrar a combinação convexa ótima (*Wolpert, 1992; Géron p. 208-210*).
   * **Ensemble Frequência/Atraso (Momentum Multi-Janela com Bootstrap):** Comitê que combina múltiplas escalas de decaimento temporal ($\lambda \in [15, 30, 50, 80, 120, 180, 250]$) sob amostragem *Bootstrap* cronológica, estabilizando acelerações de curto prazo e tendências de longo prazo.
   * **Ensemble Frequência/Atraso (Balanceado Multi-Escala com Bootstrap):** Comitê que varia sistematicamente o peso de equilíbrio $\alpha \in [0{,}25, 0{,}35, 0{,}50, 0{,}65, 0{,}75]$ e meias-vidas de atraso sob amostragem *Bootstrap*, eliminando o viés de parametrização estática.
   * **Ensemble Batch Gradient Descent (Otimização Convexa Global):** Otimiza os pesos de combinação dos modelos-base calculando o gradiente exato da perda log-loss em todo o histórico de treino sob restrição simplex.
   * **Ensemble Mini-Batch Gradient Descent (Otimização por Blocos Temporais):** Atualiza os pesos através de subamostras temporais compactas, acelerando a convergência e mitigando mínimos locais espúrios.
   * **Ensemble Stochastic Gradient Descent (SGD com Agendamento de Taxa):** Atualiza o vetor de pesos concurso a concurso com agendamento adaptativo de taxa de aprendizado $\eta_t = \frac{\eta_0}{1 + \lambda t}$.
   * **Ensemble Bagged Logit Gradient Boosting (Comitê de Boosters com Bootstrap):** Comitê $M=10$ de estimadores Logit Gradient Boosting treinados independentemente sobre réplicas com reposição (*bootstrap*), estabilizando as probabilidades preditivas.
   * **Ensemble Dual Logit GB (Combinação Convexa Base + Bagged):** Fusão convexa ponderada entre a projeção precisa do Logit GB determinístico ($w_1 = 0{,}45$) e a robustez contra sobreajuste do Bagged Logit GB ($w_2 = 0{,}55$), operando no espaço de probabilidades calibradas com mitigação de ruído estocástico.
   * **Ensemble Meta-Comitê Pré-Treinado ($M=100$ padrão; $M=50$ opcional — Método 35, [`LotteryPretrainedMetaCommitteeModel`](mega_sena_models.py)):** Meta-ensemble de alta cardinalidade baseado diretamente nos **15 ensembles já treinados** e em seus sub-estimadores internos (`estimators_`, `sub_models_`, `staged_scores_`), utilizando *Random Subspace* ($65\%$), *Bayesian Model Averaging* com priori de Dirichlet informada por OOB (imputação justa pela média observada), calibração térmica via softmax integrada ao escore híbrido e agregação (*Soft + Probabilidade + Borda + Hard*).
   * **Ensemble Comitê Profundo Multi-Família Re-Estimado ($M=100$ padrão; $M=50$ opcional — Método 36, [`LotteryDeepEnsembleCommitteeModel`](mega_sena_models.py)):** Comitê *bootstrap* heterogêneo padronizado com $M=100$ (suportando $M=50$) estimadores treinados sobre réplicas temporais distintas em 6 famílias (`logit_gb`, `cycling_gb`, `momentum_bootstrap`, `balanced_bootstrap`, `gd_bootstrap`, `bayes_gianella_bootstrap`), ponderado por acurácia *Out-of-Bag* (`oob_score_`) e ancorado no consenso dos 15 ensembles pré-treinados.
   * **Ensemble Comitê de Gianella ($M=100$ padrão — Método 37, [`LotteryGianellaCommitteeModel`](mega_sena_models.py)):** Meta-comitê guiado pela Geometria do Acaso multi-gabarito (Top-3 partições diofantinas de Gianella com viés markoviano suave), encolhimento Bayesiano (*shrinkage*) unificando os 15 ensembles pré-treinados e estimadores fundamentais (Bayes Global/Adaptativo, ML e Frequência/Atraso), e projeção estrutural ortogonal satisfazendo os limites centrais da Soma (TCL) e Paridade.
   * **Comparativo Empírico $M=50$ vs $M=100$ e Padronização em $M=100$ (Walk-Forward Zero-Lookahead em 300 Concursos + 10 Sementes):**
     - **Padronização em $M=100$ (Estabilidade & Reprodutibilidade):** Nos últimos 300 concursos oficiais (100 Mega-Sena + 100 Lotofácil + 100 Quina; `window_train=600`), os acertos médios fora da amostra foram estatisticamente indistinguíveis entre `M=50` e `M=100` (`3,2800` vs `3,2700`, $p=0{,}317$ no Meta-Comitê; `3,2333` vs `3,2500`, $p=0{,}446$ no Comitê Profundo; concursos premiados `18 vs 15` e `11 vs 15`). Contudo, em 10 sementes sobre toda a série histórica (`3083` Mega-Sena, `3824` Lotofácil, `7162` Quina), **`M=100` reduziu o erro padrão Monte Carlo (`SE`) em ~28–29%** ($\approx 1/\sqrt{2}$: `0,0103->0,0074`, `0,0095->0,0068`, `0,0122->0,0088` no Meta-Comitê; `0,0326->0,0230`, `0,0349->0,0248`, `0,0319->0,0225` no Comitê Profundo), elevou a convergência de cauda até `0,957–1,000` (Jaccard) e aumentou a reprodutibilidade exata inter-semente (`15,6%->22,2%`, `80,0%->100,0%` e `48,9%->64,4%` no Meta-Comitê; `2,2%->8,9%` e `4,4%->8,9%` em Mega-Sena/Quina no Comitê Profundo), com custo extra mínimo no Meta-Comitê (`0,157s->0,190s` na Mega-Sena; `0,204s->0,235s` na Quina) e aceitável no Comitê Profundo (`2,51s–7,91s`). Assim, os métodos ativos no pipeline (`Métodos 35 e 36`) ficam padronizados em **`M=100`**, mantendo `n_committees=50` e `compare_committee_sizes` disponíveis via API.
     - **Escalonamento de Ensembles Base (`M=50` vs `M=100`):** Escalar ensembles base individuais (`Scaled_StochGB` $p=0{,}486$, `Scaled_LogitGB` $p=0{,}655$, `Scaled_Momentum` $p=0{,}857$, `Scaled_Balanced` $p=0{,}639$) também não mostrou ganho estatisticamente significativo em acertos médios fora da amostra.
---



## 📁 Estrutura do Repositório



```text

decision/

├── excel/                # Bases de dados oficiais atualizadas (.xlsx e .csv)

│   ├── mega_sena_asloterias_ate_concurso_3080_sorteio.xlsx

│   ├── loto_facil_asloterias_ate_concurso_3819_sorteio.xlsx

│   ├── quina_asloterias_ate_concurso_7157_sorteio.xlsx

│   └── historico_acertos_modelos.xlsx # Registro consolidado de acertos dos modelos

│

├── graficos/             # Gráficos e painéis decisórios exportados em alta resolução

│   ├── lotofacil_dashboard_decisao.png

│   ├── lotofacil_avaliacao_acertos_erros.png

│   ├── mega_sena_dashboard_decisao.png

│   ├── mega_sena_avaliacao_acertos_erros.png

│   ├── quina_dashboard_decisao.png

│   └── quina_avaliacao_acertos_erros.png

│

├── notebooks/            # Notebooks Jupyter pré-executados com gráficos completos

│   ├── lotofacil.ipynb   # Modelagem da Lotofácil (25 dezenas, 15 sorteadas)

│   ├── megasena.ipynb    # Modelagem da Mega-Sena (60 dezenas, 6 sorteadas)

│   └── quina.ipynb       # Modelagem da Quina (80 dezenas, 5 sorteadas)

│

├── tests/                # Suíte automatizada de testes (pytest)

│   ├── test_lottery_models.py

│   └── test_backtest_monitor.py

│

├── scripts/              # Rotinas operacionais e agendamentos de atualização (locais)

└── mega_sena_models.py   # Motor analítico em Python com todos os modelos probabilísticos

```



---



## 🚀 Como Executar Localmente



Para abrir os notebooks no Jupyter:

```bash

jupyter notebook notebooks/

```

Ou abra os arquivos `.ipynb` diretamente no VS Code. Todos os notebooks já estão pré-executados com todas as saídas, tabelas e gráficos renderizados.



Para rodar os testes automatizados:

```bash

pytest tests/

```



---



## 📚 Referências Bibliográficas



* ESTEVES, Luís Gustavo; IZBICKI, Rafael; STERN, Rafael Bassi. *Inferência Bayesiana: Notas de Aula*. IME-USP / UFSCar.

* IZBICKI, Rafael; SANTOS, Tiago Mendonça dos. *Aprendizado de máquina: uma abordagem estatística*. 1. ed. 2020.

* DEGROOT, Morris H. *Optimal Statistical Decisions*. John Wiley & Sons, 1970.

* GIANELLA, Renato. *The Geometry of Chance: Lotto Numbers Patterns*. 2013.

* GÉRON, Aurélien. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 2. ed. O'Reilly, 2019. Capítulo 7: *Ensemble Learning and Random Forests*.

* FRIEDMAN, Jerome H. *Greedy Function Approximation: A Gradient Boosting Machine*. The Annals of Statistics, v. 29, n. 5, p. 1189–1232, 2001.

* FRIEDMAN, Jerome H. *Stochastic Gradient Boosting*. Computational Statistics & Data Analysis, v. 38, n. 4, p. 367–378, 2002.

* BREIMAN, Leo. *Bagging Predictors*. Machine Learning, v. 24, n. 2, p. 123–140, 1996.

* BREIMAN, Leo. *Pasting Small Votes for Classification in Large Databases and On-Line*. Machine Learning, v. 36, p. 85–103, 1999.

* WOLPERT, David H. *Stacked Generalization*. Neural Networks, v. 5, n. 2, p. 241–259, 1992.

