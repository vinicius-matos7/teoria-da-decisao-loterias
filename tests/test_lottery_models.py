import os
import sys
import pytest
import numpy as np
import pandas as pd
import nbformat

# Adiciona o diretório raiz ao path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)

from mega_sena_models import (
    load_lottery_data,
    detect_lottery_game,
    find_latest_lottery_file,
    FrequencyDelayModel,
    MultiLabelMLModel,
    BayesianDecisionModel,
    GianellaColorTemplateModel,
    GianellaBayesHybridModel,
    run_all_lottery_methods
)


class TestLotteryLoading:
    @pytest.mark.parametrize("game,expected_dezenas,expected_k", [
        ("megasena", 60, 6),
        ("lotofacil", 25, 15),
        ("quina", 80, 5)
    ])
    def test_load_all_games(self, game, expected_dezenas, expected_k):
        filepath = find_latest_lottery_file(game, search_dir=os.path.join(BASE_DIR, 'excel'))
        assert os.path.exists(filepath), f"Arquivo para {game} não encontrado: {filepath}"
        
        df, binary_matrix, cfg = load_lottery_data(filepath, game=game)
        assert len(df) > 0
        assert binary_matrix.shape[1] == expected_dezenas
        assert cfg['k'] == expected_k
        assert np.all(np.isin(binary_matrix, [0, 1]))


class TestModelsExecution:
    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        # 150 sorteios sintéticos de 6 dezenas em 60 (> min_train_history=100)
        n_draws = 150
        n_dezenas = 60
        matrix = np.zeros((n_draws, n_dezenas), dtype=int)
        for i in range(n_draws):
            draw = np.random.choice(n_dezenas, size=6, replace=False)
            matrix[i, draw] = 1
        return matrix

    def test_frequency_delay_model(self, synthetic_data):
        m1 = FrequencyDelayModel(half_life=50.0, alpha=0.5).fit(synthetic_data)
        top_bal = m1.predict_top_k(6, strategy='balanced')
        top_overdue = m1.predict_top_k(6, strategy='overdue')
        top_momentum = m1.predict_top_k(6, strategy='momentum')
        
        assert len(top_bal) == 6
        assert len(top_overdue) == 6
        assert len(top_momentum) == 6
        assert all(1 <= d <= 60 for d in top_bal)
        assert len(set(top_bal)) == 6

        # Edge cases
        assert len(m1.predict_top_k(0)) == 0
        assert len(m1.predict_top_k(70)) == 60

    def test_ml_model(self, synthetic_data):
        m2 = MultiLabelMLModel(estimator='logistic_regression', random_state=42).fit(synthetic_data)
        top6 = m2.predict_top_k(6)
        assert len(top6) == 6
        assert all(1 <= d <= 60 for d in top6)
        assert len(set(top6)) == 6

    def test_bayesian_model(self, synthetic_data):
        m3 = BayesianDecisionModel(alpha_0=1.0, beta_0=9.0, decay_half_life=50.0).fit(synthetic_data)
        top6 = m3.predict_top_k(6)
        assert len(top6) == 6
        assert all(1 <= d <= 60 for d in top6)
        assert len(set(top6)) == 6
        
        u = m3.evaluate_expected_utility(top6)
        assert isinstance(u, float)
        assert u > 0.0

    def test_gianella_model(self, synthetic_data):
        m4 = GianellaColorTemplateModel(n_dezenas=60, k=6, group_size=10).fit(synthetic_data)
        top6 = m4.predict_top_k(6)
        assert len(top6) == 6
        assert len(set(top6)) == 6
        assert all(1 <= d <= 60 for d in top6)
        
        # Edge cases para predict_top_k
        assert len(m4.predict_top_k(0)) == 0
        top7 = m4.predict_top_k(7)
        assert len(top7) == 7
        assert len(set(top7)) == 7
        top70 = m4.predict_top_k(70)
        assert len(top70) == 60
        assert len(set(top70)) == 60

        # Validação de entradas
        with pytest.raises(ValueError):
            GianellaColorTemplateModel(n_dezenas=60).fit(np.zeros((0, 60)))
        with pytest.raises(ValueError):
            GianellaColorTemplateModel(n_dezenas=60).fit(np.zeros((10, 25)))

        df_templates = m4.get_template_stats_df()
        assert len(df_templates) > 0
        assert 'prob_teorica' in df_templates.columns
        assert 'freq_empirica_pct' in df_templates.columns

    def test_gianella_bayes_hybrid_model(self, synthetic_data):
        m_hybrid = GianellaBayesHybridModel(
            n_dezenas=60, k=6, group_size=10, alpha_0=1.0, beta_0=9.0, decay_half_life=50.0
        ).fit(synthetic_data)

        # 1. Teste das três estratégias
        top_global = m_hybrid.predict_gianella_bayes_global(6)
        top_adapt = m_hybrid.predict_gianella_bayes_adaptive(6)
        top_best = m_hybrid.predict_gianella_best_bayes_decision(6)

        for name, bet in [('global', top_global), ('adaptive', top_adapt), ('best_bayes', top_best)]:
            assert len(bet) == 6, f"Tamanho incorreto para {name}"
            assert len(set(bet)) == 6, f"Dezenas duplicadas em {name}"
            assert all(1 <= d <= 60 for d in bet), f"Dezenas fora do intervalo em {name}"

        # 2. Teste via predict_top_k com strategies
        assert m_hybrid.predict_top_k(6, strategy='bayes_global') == top_global
        assert m_hybrid.predict_top_k(6, strategy='bayes_adaptive') == top_adapt
        assert m_hybrid.predict_top_k(6, strategy='bayes_decision') == top_best

        # 3. Teste com escores de consenso externos
        consensus = np.random.uniform(0.1, 5.0, size=60)
        m_hybrid_cons = GianellaBayesHybridModel(
            n_dezenas=60, k=6, group_size=10
        ).fit(synthetic_data, consensus_scores=consensus)
        top_cons = m_hybrid_cons.predict_gianella_best_bayes_decision(6)
        assert len(top_cons) == 6
        assert len(set(top_cons)) == 6

        # 4. Teste com penalidade de risco / covariância
        m_risk = GianellaBayesHybridModel(
            n_dezenas=60, k=6, group_size=10, risk_penalty=0.5
        ).fit(synthetic_data)
        top_risk = m_risk.predict_top_k(6, strategy='bayes_decision')
        assert len(top_risk) == 6
        assert len(set(top_risk)) == 6

        # 5. Edge cases
        assert len(m_hybrid.predict_top_k(0)) == 0
        top70 = m_hybrid.predict_top_k(70)
        assert len(top70) == 60
        assert len(set(top70)) == 60

        with pytest.raises(ValueError):
            m_hybrid.predict_top_k(6, strategy='invalid_strategy')
        with pytest.raises(ValueError):
            GianellaBayesHybridModel(n_dezenas=60).fit(np.zeros((0, 60)))
        with pytest.raises(ValueError):
            GianellaBayesHybridModel(n_dezenas=60).fit(np.zeros((10, 25)))

        # 6. DataFrame de sumário
        summary_df = m_hybrid.get_summary_df()
        assert len(summary_df) == 60
        assert 'prob_bayes_global' in summary_df.columns
        assert 'prob_bayes_adaptativo' in summary_df.columns
        assert 'escore_melhor_decisao' in summary_df.columns


class TestNotebooksIntegrity:
    @pytest.mark.parametrize("nb_name", [
        "megasena.ipynb",
        "lotofacil.ipynb",
        "quina.ipynb"
    ])
    def test_notebook_format_and_size(self, nb_name):
        path = os.path.join(BASE_DIR, 'notebooks', nb_name)
        assert os.path.exists(path)
        with open(path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        nbformat.validate(nb)
        assert len(nb.cells) >= 10
        
        # Kernelspec deve ser padronizado para python3
        ks = nb.metadata.get('kernelspec', {})
        assert ks is not None
        assert ks.get('name') == 'python3'
        
        # O notebook da megasena deve estar em tamanho razoável (< 1.2 MB) com todos os gráficos renderizados
        if nb_name == 'megasena.ipynb':
            assert os.path.getsize(path) < 1200000


class TestRunAllLotteryMethods:
    @pytest.mark.parametrize("game,expected_k,expected_n", [
        ("megasena", 6, 60),
        ("lotofacil", 15, 25),
        ("quina", 5, 80)
    ])
    def test_run_all_lottery_methods_execution(self, game, expected_k, expected_n):
        filepath = find_latest_lottery_file(game, search_dir=os.path.join(BASE_DIR, 'excel'))
        assert os.path.exists(filepath), f"Arquivo não encontrado: {filepath}"
        
        results = run_all_lottery_methods(filepath, game=game)
        assert isinstance(results, dict)
        
        required_keys = ['df_raw', 'binary_matrix', 'cfg', 'm1_model', 'm2_model',
                         'm3_static', 'm3_dyn', 'm4_gianella', 'predictions', 'comparison_df']
        for k in required_keys:
            assert k in results, f"Chave {k} ausente em results"
            
        assert results['cfg']['k'] == expected_k
        assert results['cfg']['n_dezenas'] == expected_n
        assert len(results['predictions']) == 7
        assert len(results['comparison_df']) == 7
        
        for name, bet in results['predictions'].items():
            assert len(bet) == expected_k, f"Tamanho incorreto para {name}: {len(bet)} != {expected_k}"
            assert len(set(bet)) == expected_k, f"Dezenas duplicadas em {name}: {bet}"
            assert all(1 <= d <= expected_n for d in bet), f"Dezena fora da faixa [1, {expected_n}] em {name}: {bet}"


class TestGeneratedGraphics:
    @pytest.mark.parametrize("game,concurso", [
        ("mega_sena", 3067),
        ("lotofacil", 3797),
        ("quina", 7135)
    ])
    def test_graphics_exist(self, game, concurso):
        graficos_dir = os.path.join(BASE_DIR, 'graficos')
        f1 = os.path.join(graficos_dir, f"{game}_{concurso}_probabilidades_posteriori.png")
        f2 = os.path.join(graficos_dir, f"{game}_{concurso}_consenso_modelos.png")
        f3 = os.path.join(graficos_dir, f"{game}_{concurso}_dashboard_decisao.png")
        
        for f in [f1, f2, f3]:
            assert os.path.exists(f), f"Arquivo ausente: {f}"
            assert os.path.getsize(f) > 50000, f"Arquivo corrompido ou vazio: {f}"
