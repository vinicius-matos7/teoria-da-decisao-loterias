# Relatório de Monitoramento e Backtest Walk-Forward — Lotofácil
**Período Avaliado:** Últimos 25 concursos (3772 a 3796)
**Data da Execução:** 06/10/2026 16:26:07
**Metodologia:** Validação Walk-Forward estrita sem viés de antecipação (Zero Lookahead Bias).

---

## 1. Resumo Executivo e Tabela de Desempenho de Acertos

A tabela abaixo consolida a performance empírica de cada método nos concursos testados:

| Modelo                           |   Acertos Médios |   Desvio Padrão |   IC 95% Inf |   IC 95% Sup |   Acertos Acumulados |   Máx em 1 Sorteio |   Mín em 1 Sorteio |   Jogos Premiados |   Taxa de Premiação (%) |   15 acertos |   14 acertos |   13 acertos |   12 acertos |   11 acertos |
|:---------------------------------|-----------------:|----------------:|-------------:|-------------:|---------------------:|-------------------:|-------------------:|------------------:|------------------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
| Freq/Atraso (Balanceado)         |             8.84 |          0.8505 |       8.4889 |       9.1911 |                  221 |                 11 |                  7 |                 1 |                       4 |            0 |            0 |            0 |            0 |            1 |
| Freq/Atraso (Mais Atrasadas)     |             9.16 |          0.9866 |       8.7528 |       9.5672 |                  229 |                 11 |                  7 |                 2 |                       8 |            0 |            0 |            0 |            0 |            2 |
| Freq/Atraso (Momentum)           |             8.92 |          1.222  |       8.4156 |       9.4244 |                  223 |                 11 |                  6 |                 2 |                       8 |            0 |            0 |            0 |            0 |            2 |
| ML Regressão Logística           |             8.8  |          1.0801 |       8.3541 |       9.2459 |                  220 |                 11 |                  7 |                 1 |                       4 |            0 |            0 |            0 |            0 |            1 |
| Bayesiano Global (Beta-Binomial) |             8.72 |          1.2423 |       8.2072 |       9.2328 |                  218 |                 10 |                  6 |                 0 |                       0 |            0 |            0 |            0 |            0 |            0 |
| Bayesiano Dinâmico Adaptativo    |             8.8  |          1.0408 |       8.3704 |       9.2296 |                  220 |                 11 |                  6 |                 1 |                       4 |            0 |            0 |            0 |            0 |            1 |
| Gianella (Geometria do Acaso)    |             9    |          1.118  |       8.5385 |       9.4615 |                  225 |                 11 |                  6 |                 2 |                       8 |            0 |            0 |            0 |            0 |            2 |
| Aposta Aleatória Uniforme        |             8.88 |          1.3329 |       8.3298 |       9.4302 |                  222 |                 11 |                  6 |                 2 |                       8 |            0 |            0 |            0 |            0 |            2 |

**Benchmark Teórico Hipergeométrico:** E[X] = 9.0000 acertos por aposta simples.

---

## 2. Testes Formais de Significância Estatística

Comparações formais pareadas de cada modelo contra a **Aposta Aleatória Uniforme** e contra o valor esperado combinatório teórico:

| Modelo                           |   Média |   Dif vs Aleatório |   p-valor (t-pareado) |   p-valor (Wilcoxon) |   p-valor (vs Teórico) |   Cohen d | Significativo (alpha=0.05)   | Conclusão                                           |
|:---------------------------------|--------:|-------------------:|----------------------:|---------------------:|-----------------------:|----------:|:-----------------------------|:----------------------------------------------------|
| Freq/Atraso (Balanceado)         |    8.84 |              -0.04 |                0.9152 |               0.9472 |                 0.3563 |   -0.0215 | Não                          | Estatisticamente indistinguível do acaso (p=0.9152) |
| Freq/Atraso (Mais Atrasadas)     |    9.16 |               0.28 |                0.3062 |               0.2548 |                 0.4254 |    0.2091 | Não                          | Estatisticamente indistinguível do acaso (p=0.3062) |
| Freq/Atraso (Momentum)           |    8.92 |               0.04 |                0.9162 |               0.8336 |                 0.7463 |    0.0213 | Não                          | Estatisticamente indistinguível do acaso (p=0.9162) |
| ML Regressão Logística           |    8.8  |              -0.08 |                0.8303 |               0.7319 |                 0.3638 |   -0.0433 | Não                          | Estatisticamente indistinguível do acaso (p=0.8303) |
| Bayesiano Global (Beta-Binomial) |    8.72 |              -0.16 |                0.6599 |               0.7711 |                 0.2709 |   -0.0891 | Não                          | Estatisticamente indistinguível do acaso (p=0.6599) |
| Bayesiano Dinâmico Adaptativo    |    8.8  |              -0.08 |                0.8166 |               0.9156 |                 0.3462 |   -0.0469 | Não                          | Estatisticamente indistinguível do acaso (p=0.8166) |
| Gianella (Geometria do Acaso)    |    9    |               0.12 |                0.7179 |               0.8205 |                 1      |    0.0731 | Não                          | Estatisticamente indistinguível do acaso (p=0.7179) |
| Aposta Aleatória Uniforme        |    8.88 |               0    |                1      |               1      |                 0.6566 |    0      | Não                          | Estatisticamente indistinguível do acaso (p=1.0000) |

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
| Sorteio Real                     |           6.96 |       197.8  |             22.6  |                  8.12 |           0.67 |                     1.928 |                           100 | 5-4-3-3                              |
| Freq/Atraso (Balanceado)         |           5.44 |       187    |             24    |                  8.88 |           0.99 |                     1.875 |                             4 | 5-5-3-2                              |
| Freq/Atraso (Mais Atrasadas)     |           7.76 |       196.44 |             21.92 |                  8.56 |           1.13 |                     1.934 |                            16 | 5-4-3-3                              |
| Freq/Atraso (Momentum)           |           4.92 |       177.52 |             24    |                  8.84 |           0.6  |                     1.852 |                             4 | 6-4-4-1                              |
| ML Regressão Logística           |           6.96 |       196.92 |             23.2  |                  8.48 |           0.83 |                     1.937 |                            20 | 5-4-3-3                              |
| Bayesiano Global (Beta-Binomial) |           7.4  |       172.96 |             24    |                 10.6  |           0.7  |                     1.914 |                            20 | 5-5-3-2                              |
| Bayesiano Dinâmico Adaptativo    |           4.96 |       176.36 |             24    |                  8.72 |           0.64 |                     1.811 |                             0 | 6-4-4-1                              |
| Gianella (Geometria do Acaso)    |           7.32 |       195.36 |             24    |                  7.68 |           0.66 |                     1.933 |                            24 | 5-4-4-2                              |
| Aposta Aleatória Uniforme        |           7.08 |       194.24 |             23.16 |                  7.76 |           0.73 |                     1.928 |                             8 | 5-4-4-2                              |
| Benchmark Teórico Combinatório   |           7.2  |       195    |             22.75 |                  8.4  |           0.67 |                     2     |                           100 | Balanceado Ideal                     |

### Dimensões Analisadas:
1. **Paridade:** Número médio de dezenas pares (esperança teórica = 7.20).
2. **Soma das Dezenas:** Soma total das dezenas (esperança teórica combinatória = 195.00).
3. **Amplitude / Dispersão:** Diferença entre a maior e menor dezena (esperança teórica = 22.75).
4. **Dezenas Consecutivas:** Quantidade média de pares adjacentes sorteados (esperança teórica = 8.40).
5. **Atraso Médio:** Média de concursos que as dezenas estavam sem sair no momento pré-sorteio (esperança teórica E[D] = 0.67; gap de recorrência E[Gap] = 1.67).
6. **Gabarito Diofantino de Cores (Gianella):** Partição das dezenas nos 4 quadrantes do volante e entropia de Shannon espacial.

---

## 4. Recomendações de Decisão sob Incerteza

- **Coerência Estrutural:** Modelos Bayesianos e Estatísticos tendem a gerar bilhetes cujas características estruturais
  (soma, amplitude e gabarito diofantino) se alinham com a região de maior densidade probabilística combinatória.
- **Gerenciamento de Expectativa:** Como comprovado pelos testes de hipótese formais, nenhum algoritmo baseado em dados
  históricos altera a probabilidade a priori de ganhar o prêmio principal em loterias justas.
- **Seleção de Utilidade:** O modelo Bayesiano com aversão a risco permite diversificar os números selecionados,
  reduzindo co-ocorrência histórica desnecessária.