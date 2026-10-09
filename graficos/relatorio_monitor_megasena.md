# Relatório de Monitoramento e Backtest Walk-Forward — Mega-Sena
**Período Avaliado:** Últimos 25 concursos (3042 a 3066)
**Data da Execução:** 06/10/2026 16:25:45
**Metodologia:** Validação Walk-Forward estrita sem viés de antecipação (Zero Lookahead Bias).

---

## 1. Resumo Executivo e Tabela de Desempenho de Acertos

A tabela abaixo consolida a performance empírica de cada método nos concursos testados:

| Modelo                           |   Acertos Médios |   Desvio Padrão |   IC 95% Inf |   IC 95% Sup |   Acertos Acumulados |   Máx em 1 Sorteio |   Mín em 1 Sorteio |   Jogos Premiados |   Taxa de Premiação (%) |   Sena (6 acertos) |   Quina (5 acertos) |   Quadra (4 acertos) |
|:---------------------------------|-----------------:|----------------:|-------------:|-------------:|---------------------:|-------------------:|-------------------:|------------------:|------------------------:|-------------------:|--------------------:|---------------------:|
| Freq/Atraso (Balanceado)         |             0.44 |          0.7681 |       0.1229 |       0.7571 |                   11 |                  3 |                  0 |                 0 |                       0 |                  0 |                   0 |                    0 |
| Freq/Atraso (Mais Atrasadas)     |             0.48 |          0.6532 |       0.2104 |       0.7496 |                   12 |                  2 |                  0 |                 0 |                       0 |                  0 |                   0 |                    0 |
| Freq/Atraso (Momentum)           |             0.56 |          0.5831 |       0.3193 |       0.8007 |                   14 |                  2 |                  0 |                 0 |                       0 |                  0 |                   0 |                    0 |
| ML Regressão Logística           |             0.72 |          0.7916 |       0.3932 |       1.0468 |                   18 |                  3 |                  0 |                 0 |                       0 |                  0 |                   0 |                    0 |
| Bayesiano Global (Beta-Binomial) |             0.84 |          0.8981 |       0.4693 |       1.2107 |                   21 |                  3 |                  0 |                 0 |                       0 |                  0 |                   0 |                    0 |
| Bayesiano Dinâmico Adaptativo    |             0.4  |          0.5    |       0.1936 |       0.6064 |                   10 |                  1 |                  0 |                 0 |                       0 |                  0 |                   0 |                    0 |
| Gianella (Geometria do Acaso)    |             0.88 |          1.0132 |       0.4618 |       1.2982 |                   22 |                  4 |                  0 |                 1 |                       4 |                  0 |                   0 |                    1 |
| Aposta Aleatória Uniforme        |             0.52 |          0.7141 |       0.2252 |       0.8148 |                   13 |                  2 |                  0 |                 0 |                       0 |                  0 |                   0 |                    0 |

**Benchmark Teórico Hipergeométrico:** E[X] = 0.6000 acertos por aposta simples.

---

## 2. Testes Formais de Significância Estatística

Comparações formais pareadas de cada modelo contra a **Aposta Aleatória Uniforme** e contra o valor esperado combinatório teórico:

| Modelo                           |   Média |   Dif vs Aleatório |   p-valor (t-pareado) |   p-valor (Wilcoxon) |   p-valor (vs Teórico) |   Cohen d | Significativo (alpha=0.05)   | Conclusão                                           |
|:---------------------------------|--------:|-------------------:|----------------------:|---------------------:|-----------------------:|----------:|:-----------------------------|:----------------------------------------------------|
| Freq/Atraso (Balanceado)         |    0.44 |              -0.08 |                0.7136 |               0.6467 |                 0.308  |   -0.0743 | Não                          | Estatisticamente indistinguível do acaso (p=0.7136) |
| Freq/Atraso (Mais Atrasadas)     |    0.48 |              -0.04 |                0.8239 |               0.7907 |                 0.3675 |   -0.045  | Não                          | Estatisticamente indistinguível do acaso (p=0.8239) |
| Freq/Atraso (Momentum)           |    0.56 |               0.04 |                0.7698 |               0.763  |                 0.7346 |    0.0592 | Não                          | Estatisticamente indistinguível do acaso (p=0.7698) |
| ML Regressão Logística           |    0.72 |               0.2  |                0.38   |               0.4314 |                 0.4559 |    0.1789 | Não                          | Estatisticamente indistinguível do acaso (p=0.3800) |
| Bayesiano Global (Beta-Binomial) |    0.84 |               0.32 |                0.2352 |               0.2986 |                 0.1941 |    0.2435 | Não                          | Estatisticamente indistinguível do acaso (p=0.2352) |
| Bayesiano Dinâmico Adaptativo    |    0.4  |              -0.12 |                0.4498 |               0.4386 |                 0.0569 |   -0.1536 | Não                          | Estatisticamente indistinguível do acaso (p=0.4498) |
| Gianella (Geometria do Acaso)    |    0.88 |               0.36 |                0.195  |               0.2513 |                 0.1798 |    0.2666 | Não                          | Estatisticamente indistinguível do acaso (p=0.1950) |
| Aposta Aleatória Uniforme        |    0.52 |               0    |                1      |               1      |                 0.5806 |    0      | Não                          | Estatisticamente indistinguível do acaso (p=1.0000) |

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
| Sorteio Real                     |           3    |       176.2  |             43.48 |                  0.52 |           8.35 |                     1.628 |                           100 | 2-2-1-1                              |
| Freq/Atraso (Balanceado)         |           2.64 |       178.88 |             42    |                  0.28 |          23.93 |                     1.661 |                            28 | 3-2-1-0                              |
| Freq/Atraso (Mais Atrasadas)     |           2.2  |       188.92 |             40.8  |                  0.12 |          29.83 |                     1.536 |                            28 | 4-1-1-0                              |
| Freq/Atraso (Momentum)           |           3.84 |       166.96 |             52    |                  0    |           4.7  |                     1.768 |                            44 | 2-2-1-1                              |
| ML Regressão Logística           |           2.56 |       168.64 |             39.92 |                  0.6  |           4.01 |                     1.492 |                            28 | 3-2-1-0                              |
| Bayesiano Global (Beta-Binomial) |           1.12 |       165.92 |             48    |                  0.04 |           6.39 |                     1.918 |                            36 | 2-2-1-1                              |
| Bayesiano Dinâmico Adaptativo    |           3.56 |       155.8  |             44.56 |                  0.08 |           4.06 |                     1.716 |                            32 | 2-2-1-1                              |
| Gianella (Geometria do Acaso)    |           2.08 |       189.36 |             46.8  |                  0.24 |           6.18 |                     1.409 |                            36 | 3-2-1-0                              |
| Aposta Aleatória Uniforme        |           3.12 |       194.96 |             45.08 |                  0.48 |           9.72 |                     1.514 |                            20 | 3-2-1-0                              |
| Benchmark Teórico Combinatório   |           3    |       183    |             43.57 |                  0.5  |           9    |                     2     |                           100 | Balanceado Ideal                     |

### Dimensões Analisadas:
1. **Paridade:** Número médio de dezenas pares (esperança teórica = 3.00).
2. **Soma das Dezenas:** Soma total das dezenas (esperança teórica combinatória = 183.00).
3. **Amplitude / Dispersão:** Diferença entre a maior e menor dezena (esperança teórica = 43.57).
4. **Dezenas Consecutivas:** Quantidade média de pares adjacentes sorteados (esperança teórica = 0.50).
5. **Atraso Médio:** Média de concursos que as dezenas estavam sem sair no momento pré-sorteio (esperança teórica E[D] = 9.00; gap de recorrência E[Gap] = 10.00).
6. **Gabarito Diofantino de Cores (Gianella):** Partição das dezenas nos 4 quadrantes do volante e entropia de Shannon espacial.

---

## 4. Recomendações de Decisão sob Incerteza

- **Coerência Estrutural:** Modelos Bayesianos e Estatísticos tendem a gerar bilhetes cujas características estruturais
  (soma, amplitude e gabarito diofantino) se alinham com a região de maior densidade probabilística combinatória.
- **Gerenciamento de Expectativa:** Como comprovado pelos testes de hipótese formais, nenhum algoritmo baseado em dados
  históricos altera a probabilidade a priori de ganhar o prêmio principal em loterias justas.
- **Seleção de Utilidade:** O modelo Bayesiano com aversão a risco permite diversificar os números selecionados,
  reduzindo co-ocorrência histórica desnecessária.