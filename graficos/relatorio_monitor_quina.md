# Relatório de Monitoramento e Backtest Walk-Forward — Quina
**Período Avaliado:** Últimos 25 concursos (7110 a 7134)
**Data da Execução:** 06/10/2026 16:27:09
**Metodologia:** Validação Walk-Forward estrita sem viés de antecipação (Zero Lookahead Bias).

---

## 1. Resumo Executivo e Tabela de Desempenho de Acertos

A tabela abaixo consolida a performance empírica de cada método nos concursos testados:

| Modelo                           |   Acertos Médios |   Desvio Padrão |   IC 95% Inf |   IC 95% Sup |   Acertos Acumulados |   Máx em 1 Sorteio |   Mín em 1 Sorteio |   Jogos Premiados |   Taxa de Premiação (%) |   Quina (5 acertos) |   Quadra (4 acertos) |   Terno (3 acertos) |   Duque (2 acertos) |
|:---------------------------------|-----------------:|----------------:|-------------:|-------------:|---------------------:|-------------------:|-------------------:|------------------:|------------------------:|--------------------:|---------------------:|--------------------:|--------------------:|
| Freq/Atraso (Balanceado)         |             0.24 |          0.5228 |       0.0242 |       0.4558 |                    6 |                  2 |                  0 |                 1 |                       4 |                   0 |                    0 |                   0 |                   1 |
| Freq/Atraso (Mais Atrasadas)     |             0.36 |          0.6377 |       0.0968 |       0.6232 |                    9 |                  2 |                  0 |                 2 |                       8 |                   0 |                    0 |                   0 |                   2 |
| Freq/Atraso (Momentum)           |             0.24 |          0.5228 |       0.0242 |       0.4558 |                    6 |                  2 |                  0 |                 1 |                       4 |                   0 |                    0 |                   0 |                   1 |
| ML Regressão Logística           |             0.32 |          0.6904 |       0.035  |       0.605  |                    8 |                  2 |                  0 |                 3 |                      12 |                   0 |                    0 |                   0 |                   3 |
| Bayesiano Global (Beta-Binomial) |             0.2  |          0.4082 |       0.0315 |       0.3685 |                    5 |                  1 |                  0 |                 0 |                       0 |                   0 |                    0 |                   0 |                   0 |
| Bayesiano Dinâmico Adaptativo    |             0.28 |          0.5416 |       0.0564 |       0.5036 |                    7 |                  2 |                  0 |                 1 |                       4 |                   0 |                    0 |                   0 |                   1 |
| Gianella (Geometria do Acaso)    |             0.16 |          0.4726 |       0      |       0.3551 |                    4 |                  2 |                  0 |                 1 |                       4 |                   0 |                    0 |                   0 |                   1 |
| Aposta Aleatória Uniforme        |             0.24 |          0.5228 |       0.0242 |       0.4558 |                    6 |                  2 |                  0 |                 1 |                       4 |                   0 |                    0 |                   0 |                   1 |

**Benchmark Teórico Hipergeométrico:** E[X] = 0.3125 acertos por aposta simples.

---

## 2. Testes Formais de Significância Estatística

Comparações formais pareadas de cada modelo contra a **Aposta Aleatória Uniforme** e contra o valor esperado combinatório teórico:

| Modelo                           |   Média |   Dif vs Aleatório |   p-valor (t-pareado) |   p-valor (Wilcoxon) |   p-valor (vs Teórico) |   Cohen d | Significativo (alpha=0.05)   | Conclusão                                           |
|:---------------------------------|--------:|-------------------:|----------------------:|---------------------:|-----------------------:|----------:|:-----------------------------|:----------------------------------------------------|
| Freq/Atraso (Balanceado)         |    0.24 |               0    |                1      |               1      |                 0.4947 |    0      | Não                          | Estatisticamente indistinguível do acaso (p=1.0000) |
| Freq/Atraso (Mais Atrasadas)     |    0.36 |               0.12 |                0.5238 |               0.5094 |                 0.7128 |    0.1294 | Não                          | Estatisticamente indistinguível do acaso (p=0.5238) |
| Freq/Atraso (Momentum)           |    0.24 |               0    |                1      |               1      |                 0.4947 |    0      | Não                          | Estatisticamente indistinguível do acaso (p=1.0000) |
| ML Regressão Logística           |    0.32 |               0.08 |                0.6469 |               0.6234 |                 0.9571 |    0.0928 | Não                          | Estatisticamente indistinguível do acaso (p=0.6469) |
| Bayesiano Global (Beta-Binomial) |    0.2  |              -0.04 |                0.7878 |               0.7815 |                 0.181  |   -0.0544 | Não                          | Estatisticamente indistinguível do acaso (p=0.7878) |
| Bayesiano Dinâmico Adaptativo    |    0.28 |               0.04 |                0.8022 |               0.8028 |                 0.7667 |    0.0507 | Não                          | Estatisticamente indistinguível do acaso (p=0.8022) |
| Gianella (Geometria do Acaso)    |    0.16 |              -0.08 |                0.5743 |               0.5887 |                 0.1197 |   -0.1139 | Não                          | Estatisticamente indistinguível do acaso (p=0.5743) |
| Aposta Aleatória Uniforme        |    0.24 |               0    |                1      |               1      |                 0.4947 |    0      | Não                          | Estatisticamente indistinguível do acaso (p=1.0000) |

> **Interpretação sob a Teoria da Decisão e Inferência Estatística (Esteves, Izbicki & Stern):**
> Em um jogo de loteria matematicamente justo e sem vício mecânico, os sorteios sucessivos formam uma sequência
> de variáveis aleatórias independentes e identicamente distribuídas (i.i.d.).
> Qualquer variação no número médio de acertos em amostras finitas (ex: 20 a 50 concursos) constitui flutuação
> amostral se os p-valores dos testes pareados forem superiores ao nível de significância convencional (alpha = 0.05).

---

## 3. Comparação de Características Estruturais (Real vs Modelos vs Teórico)

Avaliação das propriedades combinatórias dos bilhetes gerados por cada modelo em relação aos sorteios reais:

| Origem / Modelo                  |   Pares Médios |   Soma Média |   Amplitude Média |   Consecutivos Médios |   Atraso Médio |   Entropia Gianella Média |   Aderência Gabarito Real (%) | Gabarito Diofantino Mais Frequente   |
|:---------------------------------|---------------:|-------------:|------------------:|----------------------:|---------------:|--------------------------:|------------------------------:|:-------------------------------------|
| Sorteio Real                     |           2.68 |       214.4  |             52.76 |                  0.36 |          14.49 |                     1.378 |                           100 | 2-2-1-0                              |
| Freq/Atraso (Balanceado)         |           3.44 |       220.52 |             36.68 |                  1    |          37.51 |                     1.079 |                            20 | 3-2-0-0                              |
| Freq/Atraso (Mais Atrasadas)     |           2.44 |       245.36 |             45.24 |                  0.24 |          46.78 |                     1.382 |                            20 | 3-1-1-0                              |
| Freq/Atraso (Momentum)           |           3.4  |       183.92 |             50.28 |                  0.16 |          10.11 |                     1.706 |                            24 | 2-1-1-1                              |
| ML Regressão Logística           |           2.4  |       143.24 |             44    |                  0.4  |           6.48 |                     1.354 |                            16 | 3-1-1-0                              |
| Bayesiano Global (Beta-Binomial) |           4    |       175    |             48    |                  0    |          13.74 |                     1.371 |                            20 | 3-1-1-0                              |
| Bayesiano Dinâmico Adaptativo    |           3    |       175.52 |             42.84 |                  0.56 |          13.66 |                     1.522 |                            56 | 2-2-1-0                              |
| Gianella (Geometria do Acaso)    |           3.96 |       158.32 |             51.84 |                  0    |          25.29 |                     1.522 |                            56 | 2-2-1-0                              |
| Aposta Aleatória Uniforme        |           2.32 |       198.2  |             53.92 |                  0.2  |          13.39 |                     1.382 |                            20 | 2-2-1-0                              |
| Benchmark Teórico Combinatório   |           2.5  |       202.5  |             54    |                  0.25 |          15    |                     2     |                           100 | Balanceado Ideal                     |

### Dimensões Analisadas:
1. **Paridade:** Número médio de dezenas pares (esperança teórica = 2.50).
2. **Soma das Dezenas:** Soma total das dezenas (esperança teórica combinatória = 202.50).
3. **Amplitude / Dispersão:** Diferença entre a maior e menor dezena (esperança teórica = 54.00).
4. **Dezenas Consecutivas:** Quantidade média de pares adjacentes sorteados (esperança teórica = 0.25).
5. **Atraso Médio:** Média de concursos que as dezenas estavam sem sair no momento pré-sorteio (esperança teórica E[D] = 15.00; gap de recorrência E[Gap] = 16.00).
6. **Gabarito Diofantino de Cores (Gianella):** Partição das dezenas nos 4 quadrantes do volante e entropia de Shannon espacial.

---

## 4. Recomendações de Decisão sob Incerteza

- **Coerência Estrutural:** Modelos Bayesianos e Estatísticos tendem a gerar bilhetes cujas características estruturais
  (soma, amplitude e gabarito diofantino) se alinham com a região de maior densidade probabilística combinatória.
- **Gerenciamento de Expectativa:** Como comprovado pelos testes de hipótese formais, nenhum algoritmo baseado em dados
  históricos altera a probabilidade a priori de ganhar o prêmio principal em loterias justas.
- **Seleção de Utilidade:** O modelo Bayesiano com aversão a risco permite diversificar os números selecionados,
  reduzindo co-ocorrência histórica desnecessária.