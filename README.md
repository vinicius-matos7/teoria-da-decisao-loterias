# Teoria da Decisão & Modelagem Preditiva de Loterias

Projeto analítico e probabilístico para modelagem de loterias brasileiras (**Mega-Sena**, **Lotofácil** e **Quina**) utilizando conceitos de **Inferência Bayesiana** e **Teoria da Decisão**.

---

## 📁 Estrutura do Repositório

```text
decision/
├── excel/                # Bases de dados oficiais e planilhas atualizadas
│   ├── mega_sena_asloterias_ate_concurso_3065_sorteio.xlsx (e .csv)
│   ├── loto_facil_asloterias_ate_concurso_3794_sorteio.xlsx (e .csv)
│   └── quina_asloterias_ate_concurso_7132_sorteio.xlsx (e .csv)
│
├── notebooks/            # Notebooks Jupyter para cada modalidade (executados com gráficos)
│   ├── megasena.ipynb    # Modelagem para a Mega-Sena (60 dezenas, 6 sorteadas) -> Alvo 3066
│   ├── lotofacil.ipynb   # Modelagem para a Lotofácil (25 dezenas, 15 sorteadas) -> Alvo 3795
│   └── quina.ipynb       # Modelagem para a Quina (80 dezenas, 5 sorteadas) -> Alvo 7133
│
└── mega_sena_models.py   # Motor analítico em Python com os 3 métodos probabilísticos
```

---

## 🧠 Métodos Implementados

Cada notebook executa e compara **3 métodos complementares**:

1. **Método 1 (Estatístico - Frequência Ponderada & Atraso):**
   * Ponderação temporal com decaimento exponencial ($w_t = 2^{-(T - t) / 100}$).
   * Métrica de atraso relativo ($R_i = d_i / \mu_{\text{gap}, i}$).
   * Seleção balanceada, atrasada ou de momento recente.

2. **Método 2 (Machine Learning Multi-label):**
   * Modelo supervisionado (*Binary Relevance* com Regressão Logística calibrada).
   * Janelas temporais móveis multiescala (5, 10, 25, 50 concursos) e defasagens (*lags*).
   * Validação cronológica sem vazamento de dados (*lookahead bias*).

3. **Método 3 (Bayesiano & Teoria da Decisão):**
   * Modelo conjugado Beta-Binomial com priori justa informada pela probabilidade teórica neutra de cada modalidade:
     * Mega-Sena ($6/60 = 0.10$): $\text{Beta}(1, 9)$
     * Lotofácil ($15/25 = 0.60$): $\text{Beta}(6, 4)$
     * Quina ($5/80 = 0.0625$): $\text{Beta}(1, 15)$
   * Formulação do problema de decisão $(\mathcal{A}, \Theta, P, U)$ e seleção pela **Alternativa de Bayes** (maximização da utilidade esperada a posteriori).

---

## 🚀 Como Executar

Para abrir os notebooks no Jupyter:
```bash
jupyter notebook notebooks/
```
Ou abra qualquer um dos notebooks diretamente no VS Code. Todos os notebooks já estão pré-executados e com seus gráficos e previsões renderizados.
