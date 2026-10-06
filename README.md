# Teoria da Decisão & Modelagem Preditiva de Loterias

Projeto analítico e probabilístico para modelagem de loterias brasileiras (**Mega-Sena**, **Lotofácil** e **Quina**) utilizando conceitos de **Inferência Bayesiana** e **Teoria da Decisão**, fundamentados na literatura dos professores **Luís Gustavo Esteves, Rafael Izbicki e Rafael Bassi Stern** (*Aprendizado de Máquina sob a Ótica Bayesiana* e *Inferência Bayesiana e Teoria da Decisão*).

---

## 📁 Estrutura do Repositório

```text
decision/
├── excel/                # Bases de dados oficiais e planilhas atualizadas
│   ├── mega_sena_asloterias_ate_concurso_3066_sorteio.xlsx (e .csv)
│   ├── loto_facil_asloterias_ate_concurso_3796_sorteio.xlsx (e .csv)
│   └── quina_asloterias_ate_concurso_7134_sorteio.xlsx (e .csv)
│
├── notebooks/            # Notebooks Jupyter para cada modalidade (executados com gráficos)
│   ├── megasena.ipynb    # Modelagem para a Mega-Sena (60 dezenas, 6 sorteadas) -> Alvo 3067
│   ├── lotofacil.ipynb   # Modelagem para a Lotofácil (25 dezenas, 15 sorteadas) -> Alvo 3797
│   └── quina.ipynb       # Modelagem para a Quina (80 dezenas, 5 sorteadas) -> Alvo 7135
│
└── mega_sena_models.py   # Motor analítico em Python com os 3 métodos probabilísticos
```

---

## 🧠 Métodos Implementados

Cada notebook executa e compara **3 métodos complementares**:

1. **Método 1 (Estatístico - Frequência Ponderada & Atraso):**
   * Ponderação temporal com decaimento exponencial ($w_t = 2^{-(T - t) / 100}$).
   * Métrica de atraso relativo ($R_i = d_i / \mu_{\text{gap}, i}$).
   * Seleção balanceada, atrasada ou de momento recente (*momentum*).

2. **Método 2 (Machine Learning Multi-label):**
   * Modelo supervisionado (*Binary Relevance* com Regressão Logística calibrada $L_2$).
   * Janelas temporais móveis multiescala (5, 10, 25, 50 e 100 concursos) e defasagens (*lags*).
   * Validação cronológica rigorosa sem vazamento de dados futuros (*zero lookahead bias*).

3. **Método 3 (Bayesiano & Teoria da Decisão):**
   * Modelo conjugado Beta-Binomial com **priori racional justa** informada pela probabilidade teórica neutra de cada modalidade:
     * Mega-Sena ($6/60 = 0.10$): $\text{Beta}(1, 9)$
     * Lotofácil ($15/25 = 0.60$): $\text{Beta}(6, 4)$
     * Quina ($5/80 = 0.0625$): $\text{Beta}(1, 15)$
   * Formulação do problema formal de decisão $(\mathcal{A}, \Theta, P, U)$ e seleção pela **Alternativa de Bayes** (maximização da utilidade esperada a posteriori).
   * Comparação analítica entre o **Conjugado Global** (tendência acumulada de longo prazo) e o **Dinâmico Adaptativo** (regime físico e mecânico recente).

---

## 🚀 Como Executar

Para abrir os notebooks no Jupyter:
```bash
jupyter notebook notebooks/
```
Ou abra qualquer um dos notebooks diretamente no VS Code. Todos os notebooks já estão pré-executados e com seus gráficos, intervalos de credibilidade e estimativas renderizados.

---

## 📚 Referências Bibliográficas
* ESTEVES, Luís Gustavo; IZBICKI, Rafael; STERN, Rafael Bassi. *Aprendizado de Máquina sob a Ótica Bayesiana*.
* STERN, Rafael Bassi. *Inferência Bayesiana e Teoria da Decisão*.
