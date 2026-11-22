"""
metodo_39_nobre_stephens.py
===========================
Método 39: Modelo Bayesiano Causal Duplamente Robusto com Bootstrap de Dirichlet
Baseado em:
  1. Stephens, D. A., Nobre, W. S., Moodie, E. E. M., & Schmidt, A. M. (2023).
     "Causal Inference Under Mis-Specification: Adjustment Based on the Propensity Score".
     Bayesian Analysis, 18(2), 639-694.
  2. Nobre, W. da S. (2021).
     "A Bayesian View on Causal Inference for Observational Data".
     Tese de Doutorado em Estatística, Instituto de Matemática, UFRJ.

Pilares:
  - Cutting Feedback (Separação estanque entre escore basal estrutural e momentum de curto prazo)
  - Escore de Propensão b(X) via invariantes do Teorema Central do Limite e Bernoulli
  - Regressão Duplamente Robusta (Double Robustness)
  - Bayesian Bootstrap via Processo de Dirichlet (w ~ Dirichlet(1, ..., 1))
  - Filtro combinatório de Soma (TCL) e Paridade
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from mega_sena_models import (
    LotteryCausalPropensityModel,
    predict_causal_propensity,
    load_lottery_data,
    find_latest_lottery_file
)

def executar_metodo_39(jogo: str = 'lotofacil', n_bootstrap: int = 50):
    filepath = find_latest_lottery_file(jogo)
    df, binary_matrix, cfg = load_lottery_data(filepath, game=jogo)
    k = cfg['k']
    n_dezenas = cfg['n_dezenas']
    ultimo_concurso = int(df['Concurso'].max())
    proximo_concurso = ultimo_concurso + 1
    
    print("=" * 80)
    print(f"MÉTODO 39 (NOBRE-STEPHENS) — {cfg['name'].upper()}")
    print(f"Base de dados: {filepath} ({binary_matrix.shape[0]} sorteios históricos)")
    print(f"Próximo Concurso Previsto: {proximo_concurso}")
    print("=" * 80)
    
    # Instanciação e Treinamento do Modelo
    modelo = LotteryCausalPropensityModel(n_bootstrap=n_bootstrap, random_state=42)
    modelo.fit(binary_matrix, cfg=cfg)
    
    # Palpite Ótimo
    palpite = modelo.predict_top_k(k)
    metrics = modelo.top_ticket_metrics_
    
    print(f"\n★ PALPITE FINAL RECOMENDADO (TOP {k} DEZENAS):")
    print(f"   -> {palpite}")
    print(f"\n📊 Métricas Estruturais da Aposta:")
    print(f"   • Soma das Dezenas: {metrics['sum']} (Faixa Central TCL: {metrics['sum_bounds']}) -> Válido: {metrics['passes_tcl_sum']}")
    print(f"   • Dezenas Pares: {metrics['evens']} (Faixa Esperada: {metrics['even_bounds']}) -> Válido: {metrics['passes_parity']}")
    print(f"   • Erro Padrão Médio (SE Monte Carlo): {metrics['mean_se']:.6f}")
    
    print("\n📋 Top 10 Dezenas por Escore Causal de Propensão:")
    df_scores = modelo.get_scores_df()
    print(df_scores.head(10).to_string(index=False))
    return palpite

if __name__ == '__main__':
    for game in ['lotofacil', 'megasena', 'quina']:
        executar_metodo_39(game)
        print("\n" + "#" * 80 + "\n")
