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
    BayesianHybridModel,
    LotteryBaggingModel,
    LotteryPastingModel,
    LotteryAdaBoostModel,
    LotteryGradientBoostingModel,
    LotteryStackingModel,
    predict_gianella_bayes_global,
    predict_gianella_bayes_adaptive,
    predict_gianella_best_bayes_decision,
    predict_gianella_logistic_regression,
    predict_gianella_momentum,
    predict_gianella_balanced,
    predict_best_bayes_logistic_regression,
    predict_best_bayes_momentum,
    predict_best_bayes_balanced,
    predict_bayes_global_logistic_regression,
    predict_bayes_global_momentum,
    predict_bayes_global_balanced,
    predict_bayes_adaptive_logistic_regression,
    predict_bayes_adaptive_momentum,
    predict_bayes_adaptive_balanced,
    predict_bagging,
    predict_pasting,
    predict_adaboost,
    predict_gradient_boosting,
    predict_stacking,
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

        # 1. Teste de todas as estratégias híbridas no GianellaBayesHybridModel
        top_global = m_hybrid.predict_gianella_bayes_global(6)
        top_adapt = m_hybrid.predict_gianella_bayes_adaptive(6)
        top_best = m_hybrid.predict_gianella_best_bayes_decision(6)
        top_g_lr = m_hybrid.predict_gianella_logistic_regression(6)
        top_g_mom = m_hybrid.predict_gianella_momentum(6)
        top_g_bal = m_hybrid.predict_gianella_balanced(6)
        top_bb_lr = m_hybrid.predict_best_bayes_logistic_regression(6)
        top_bb_mom = m_hybrid.predict_best_bayes_momentum(6)
        top_bb_bal = m_hybrid.predict_best_bayes_balanced(6)
        top_bg_lr = m_hybrid.predict_bayes_global_logistic_regression(6)
        top_bg_mom = m_hybrid.predict_bayes_global_momentum(6)
        top_bg_bal = m_hybrid.predict_bayes_global_balanced(6)
        top_ba_lr = m_hybrid.predict_bayes_adaptive_logistic_regression(6)
        top_ba_mom = m_hybrid.predict_bayes_adaptive_momentum(6)
        top_ba_bal = m_hybrid.predict_bayes_adaptive_balanced(6)

        all_hybrid_bets = [
            ('g_global', top_global), ('g_adapt', top_adapt), ('g_best', top_best),
            ('g_lr', top_g_lr), ('g_mom', top_g_mom), ('g_bal', top_g_bal),
            ('bb_lr', top_bb_lr), ('bb_mom', top_bb_mom), ('bb_bal', top_bb_bal),
            ('bg_lr', top_bg_lr), ('bg_mom', top_bg_mom), ('bg_bal', top_bg_bal),
            ('ba_lr', top_ba_lr), ('ba_mom', top_ba_mom), ('ba_bal', top_ba_bal)
        ]
        for name, bet in all_hybrid_bets:
            assert len(bet) == 6, f"Tamanho incorreto para {name}"
            assert len(set(bet)) == 6, f"Dezenas duplicadas em {name}"
            assert all(1 <= d <= 60 for d in bet), f"Dezenas fora do intervalo em {name}"

        # 2. Teste via predict_top_k com strategies
        assert m_hybrid.predict_top_k(6, strategy='bayes_global') == top_global
        assert m_hybrid.predict_top_k(6, strategy='bayes_adaptive') == top_adapt
        assert m_hybrid.predict_top_k(6, strategy='bayes_decision') == top_best
        assert m_hybrid.predict_top_k(6, strategy='logistic_regression') == top_g_lr
        assert m_hybrid.predict_top_k(6, strategy='momentum') == top_g_mom
        assert m_hybrid.predict_top_k(6, strategy='balanced') == top_g_bal

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

        # 4b. Teste específico de penalização de covariância no grupo 0
        X_corr = np.zeros((100, 60))
        coin = np.random.binomial(1, 0.5, size=100)
        X_corr[:, 0] = coin
        X_corr[:, 1] = coin  # cov(0, 1) positiva alta (~0.25)
        X_corr[:, 2] = np.random.binomial(1, 0.48, size=100)  # cov com 0 ~ 0
        m_risk_corr = GianellaBayesHybridModel(
            n_dezenas=60, k=6, group_size=10, risk_penalty=10.0
        ).fit(X_corr)
        top_risk_corr = m_risk_corr.predict_top_k(6, strategy='bayes_decision', template=(2, 1, 1, 1, 1, 0))
        g0_picks = [b for b in top_risk_corr if b <= 10]
        assert g0_picks == [1, 3], f"Esperado penalizar dezena 2 em favor da dezena 3, obtido {g0_picks}"

        # 5. Funções utilitárias diretas
        cfg_test = {'n_dezenas': 60, 'k': 6, 'gianella_group_size': 10, 'alpha_0': 1.0, 'beta_0': 9.0}
        fn_global = predict_gianella_bayes_global(synthetic_data, cfg_test, k=6)
        fn_adapt = predict_gianella_bayes_adaptive(synthetic_data, cfg_test, k=6)
        fn_best = predict_gianella_best_bayes_decision(synthetic_data, cfg_test, k=6)
        for fn_bet in [fn_global, fn_adapt, fn_best]:
            assert len(fn_bet) == 6
            assert len(set(fn_bet)) == 6
            assert all(1 <= d <= 60 for d in fn_bet)

        # 6. Edge cases e validações
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
        with pytest.raises(ValueError):
            GianellaBayesHybridModel(n_dezenas=60, k=70)
        with pytest.raises(ValueError):
            GianellaBayesHybridModel(n_dezenas=53, group_size=10)
        with pytest.raises(ValueError):
            GianellaColorTemplateModel(n_dezenas=60, k=70)
        with pytest.raises(ValueError):
            GianellaColorTemplateModel(n_dezenas=53, group_size=10)

        # 7. DataFrame de sumário
        summary_df = m_hybrid.get_summary_df()
        assert len(summary_df) == 60
        assert 'prob_bayes_global' in summary_df.columns
        assert 'prob_bayes_adaptativo' in summary_df.columns
        assert 'escore_melhor_decisao' in summary_df.columns

    def test_bayesian_hybrid_model(self, synthetic_data):
        m_bayes_hyb = BayesianHybridModel(
            n_dezenas=60, k=6, alpha_0=1.0, beta_0=9.0, decay_half_life=50.0
        ).fit(synthetic_data)

        bb_lr = m_bayes_hyb.predict_best_bayes_logistic_regression(6)
        bb_mom = m_bayes_hyb.predict_best_bayes_momentum(6)
        bb_bal = m_bayes_hyb.predict_best_bayes_balanced(6)
        bg_lr = m_bayes_hyb.predict_bayes_global_logistic_regression(6)
        bg_mom = m_bayes_hyb.predict_bayes_global_momentum(6)
        bg_bal = m_bayes_hyb.predict_bayes_global_balanced(6)
        ba_lr = m_bayes_hyb.predict_bayes_adaptive_logistic_regression(6)
        ba_mom = m_bayes_hyb.predict_bayes_adaptive_momentum(6)
        ba_bal = m_bayes_hyb.predict_bayes_adaptive_balanced(6)

        for name, bet in [
            ('bb_lr', bb_lr), ('bb_mom', bb_mom), ('bb_bal', bb_bal),
            ('bg_lr', bg_lr), ('bg_mom', bg_mom), ('bg_bal', bg_bal),
            ('ba_lr', ba_lr), ('ba_mom', ba_mom), ('ba_bal', ba_bal)
        ]:
            assert len(bet) == 6, f"Tamanho incorreto para {name}"
            assert len(set(bet)) == 6, f"Dezenas duplicadas em {name}"
            assert all(1 <= d <= 60 for d in bet), f"Dezenas fora da faixa em {name}"

    def test_lottery_bagging_model(self, synthetic_data):
        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}
        m_bag = LotteryBaggingModel(n_estimators=5, max_samples=0.8, oob_score=True, random_state=42)
        m_bag.fit(synthetic_data, cfg=cfg)

        top_soft = m_bag.predict_top_k(6, voting='soft')
        top_hard = m_bag.predict_top_k(6, voting='hard')

        assert len(top_soft) == 6
        assert len(set(top_soft)) == 6
        assert all(1 <= d <= 60 for d in top_soft)

        assert len(top_hard) == 6
        assert len(set(top_hard)) == 6
        assert all(1 <= d <= 60 for d in top_hard)

        assert isinstance(m_bag.oob_score_, float)
        assert m_bag.oob_score_ >= 0.0

        scores_df = m_bag.get_scores_df()
        assert len(scores_df) == 60
        assert 'escore_bagging' in scores_df.columns
        assert 'votos_hard' in scores_df.columns

        # Edge cases & validation
        assert len(m_bag.predict_top_k(0)) == 0
        top70 = m_bag.predict_top_k(70)
        assert len(top70) == 60

        with pytest.raises(ValueError):
            LotteryBaggingModel(n_estimators=0)
        with pytest.raises(ValueError):
            LotteryBaggingModel(max_samples=0.0)
        with pytest.raises(ValueError):
            LotteryBaggingModel(voting='invalid')
        with pytest.raises(ValueError):
            LotteryBaggingModel().fit(np.zeros((0, 60)))

    def test_lottery_adaboost_model(self, synthetic_data):
        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}
        m_ada = LotteryAdaBoostModel(n_estimators=5, learning_rate=1.0, random_state=42)
        m_ada.fit(synthetic_data, cfg=cfg)

        top6 = m_ada.predict_top_k(6)
        assert len(top6) == 6
        assert len(set(top6)) == 6
        assert all(1 <= d <= 60 for d in top6)

        assert len(m_ada.estimator_weights_) == 5
        assert all(w > 0 for w in m_ada.estimator_weights_)
        assert len(m_ada.estimator_errors_) == 5

        scores_df = m_ada.get_scores_df()
        assert len(scores_df) == 60
        assert 'escore_adaboost' in scores_df.columns

        # Edge cases & validation
        assert len(m_ada.predict_top_k(0)) == 0
        assert len(m_ada.predict_top_k(70)) == 60

        with pytest.raises(ValueError):
            LotteryAdaBoostModel(n_estimators=0)
        with pytest.raises(ValueError):
            LotteryAdaBoostModel(learning_rate=0.0)
        with pytest.raises(ValueError):
            LotteryAdaBoostModel().fit(np.zeros((0, 60)))

    def test_lottery_gradient_boosting_model(self, synthetic_data):
        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}
        m_gb = LotteryGradientBoostingModel(n_estimators=5, learning_rate=0.5, random_state=42)
        m_gb.fit(synthetic_data, cfg=cfg)

        top6 = m_gb.predict_top_k(6)
        assert len(top6) == 6
        assert len(set(top6)) == 6
        assert all(1 <= d <= 60 for d in top6)

        staged = list(m_gb.staged_predict_top_k(6))
        assert len(staged) == 6  # F_0 + 5 stages
        for stage_bet in staged:
            assert len(stage_bet) == 6
            assert len(set(stage_bet)) == 6
            assert all(1 <= d <= 60 for d in stage_bet)

        scores_df = m_gb.get_scores_df()
        assert len(scores_df) == 60
        assert 'escore_gradboost' in scores_df.columns

        # Edge cases & validation
        assert len(m_gb.predict_top_k(0)) == 0
        assert len(m_gb.predict_top_k(70)) == 60

        with pytest.raises(ValueError):
            LotteryGradientBoostingModel(n_estimators=0)
        with pytest.raises(ValueError):
            LotteryGradientBoostingModel(learning_rate=0.0)
        with pytest.raises(ValueError):
            LotteryGradientBoostingModel().fit(np.zeros((0, 60)))

    def test_lottery_pasting_model(self, synthetic_data):
        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}
        m_paste = LotteryPastingModel(n_estimators=5, max_samples=0.8, random_state=42)
        m_paste.fit(synthetic_data, cfg=cfg)

        top6 = m_paste.predict_top_k(6)
        assert len(top6) == 6
        assert len(set(top6)) == 6
        assert all(1 <= d <= 60 for d in top6)

        top6_hard = m_paste.predict_top_k(6, voting='hard')
        assert len(top6_hard) == 6
        assert len(set(top6_hard)) == 6

        scores_df = m_paste.get_scores_df()
        assert len(scores_df) == 60
        assert 'escore_pasting' in scores_df.columns
        assert 'votos_hard' in scores_df.columns

        # Edge cases & validation
        assert len(m_paste.predict_top_k(0)) == 0
        top70 = m_paste.predict_top_k(70)
        assert len(top70) == 60

        with pytest.raises(ValueError):
            LotteryPastingModel(n_estimators=0)
        with pytest.raises(ValueError):
            LotteryPastingModel(max_samples=0.0)
        with pytest.raises(ValueError):
            LotteryPastingModel(voting='invalid')
        with pytest.raises(ValueError):
            LotteryPastingModel().fit(np.zeros((0, 60)))

    def test_lottery_stacking_model(self, synthetic_data):
        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}
        m_stack = LotteryStackingModel(holdout_ratio=0.25, random_state=42)
        m_stack.fit(synthetic_data, cfg=cfg)

        top6 = m_stack.predict_top_k(6)
        assert len(top6) == 6
        assert len(set(top6)) == 6
        assert all(1 <= d <= 60 for d in top6)

        assert len(m_stack.blender_weights_) == len(m_stack.base_models)
        scores_df = m_stack.get_scores_df()
        assert len(scores_df) == 60
        assert 'escore_stacking' in scores_df.columns

        # Edge cases & validation
        assert len(m_stack.predict_top_k(0)) == 0
        assert len(m_stack.predict_top_k(70)) == 60

        with pytest.raises(ValueError):
            LotteryStackingModel(holdout_ratio=0.01)
        with pytest.raises(ValueError):
            LotteryStackingModel(meta_learner='invalid')
        with pytest.raises(ValueError):
            LotteryStackingModel().fit(np.zeros((0, 60)))

    def test_ensemble_helper_functions(self, synthetic_data):
        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}
        p_bag = predict_bagging(synthetic_data, cfg, k=6, n_estimators=3)
        p_paste = predict_pasting(synthetic_data, cfg, k=6, n_estimators=3)
        p_ada = predict_adaboost(synthetic_data, cfg, k=6, n_estimators=3)
        p_gb = predict_gradient_boosting(synthetic_data, cfg, k=6, n_estimators=3)
        p_stack = predict_stacking(synthetic_data, cfg, k=6)

        for name, bet in [('bagging', p_bag), ('pasting', p_paste), ('adaboost', p_ada), ('gradboost', p_gb), ('stacking', p_stack)]:
            assert len(bet) == 6, f"Tamanho incorreto para {name}"
            assert len(set(bet)) == 6, f"Dezenas repetidas em {name}"
            assert all(1 <= d <= 60 for d in bet), f"Dezenas fora da faixa em {name}"


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
                         'm3_static', 'm3_dyn', 'm4_gianella', 'm_bagging', 'm_pasting', 'm_adaboost',
                         'm_gradboost', 'm_stacking', 'predictions', 'predictions_ensemble', 'comparison_df']
        for k in required_keys:
            assert k in results, f"Chave {k} ausente em results"
            
        assert results['cfg']['k'] == expected_k
        assert results['cfg']['n_dezenas'] == expected_n
        assert len(results['predictions']) == 7
        assert len(results['comparison_df']) == 7
        assert len(results['predictions_ensemble']) == 5
        
        for name, bet in results['predictions'].items():
            assert len(bet) == expected_k, f"Tamanho incorreto para {name}: {len(bet)} != {expected_k}"
            assert len(set(bet)) == expected_k, f"Dezenas duplicadas em {name}: {bet}"
            assert all(1 <= d <= expected_n for d in bet), f"Dezena fora da faixa [1, {expected_n}] em {name}: {bet}"

        for name, bet in results['predictions_ensemble'].items():
            assert len(bet) == expected_k, f"Tamanho incorreto para {name}: {len(bet)} != {expected_k}"
            assert len(set(bet)) == expected_k, f"Dezenas duplicadas em {name}: {bet}"
            assert all(1 <= d <= expected_n for d in bet), f"Dezena fora da faixa [1, {expected_n}] em {name}: {bet}"

    @pytest.mark.parametrize("game,expected_k,expected_n", [
        ("megasena", 6, 60),
        ("lotofacil", 15, 25),
        ("quina", 5, 80)
    ])
    def test_run_all_lottery_methods_with_hybrids(self, game, expected_k, expected_n):
        filepath = find_latest_lottery_file(game, search_dir=os.path.join(BASE_DIR, 'excel'))
        assert os.path.exists(filepath), f"Arquivo não encontrado: {filepath}"

        results = run_all_lottery_methods(filepath, game=game, include_hybrids=True)
        assert isinstance(results, dict)

        required_keys = ['df_raw', 'binary_matrix', 'cfg', 'm1_model', 'm2_model',
                         'm3_static', 'm3_dyn', 'm4_gianella', 'm_hybrid', 'm_bagging', 'm_pasting',
                         'm_adaboost', 'm_gradboost', 'm_stacking', 'pred_consenso',
                         'consensus_scores', 'predictions', 'predictions_hybrid',
                         'predictions_ensemble', 'comparison_df']
        for k in required_keys:
            assert k in results, f"Chave {k} ausente em results"

        assert len(results['predictions']) == 28
        assert len(results['comparison_df']) == 28
        assert len(results['predictions_hybrid']) == 15
        assert len(results['predictions_ensemble']) == 5

        for name, bet in results['predictions'].items():
            assert len(bet) == expected_k, f"Tamanho incorreto para {name}: {len(bet)} != {expected_k}"
            assert len(set(bet)) == expected_k, f"Dezenas duplicadas em {name}: {bet}"
            assert all(1 <= d <= expected_n for d in bet), f"Dezena fora da faixa [1, {expected_n}] em {name}: {bet}"

        for name, bet in results['predictions_hybrid'].items():
            assert len(bet) == expected_k
            assert len(set(bet)) == expected_k
            assert all(1 <= d <= expected_n for d in bet)

        for name, bet in results['predictions_ensemble'].items():
            assert len(bet) == expected_k
            assert len(set(bet)) == expected_k
            assert all(1 <= d <= expected_n for d in bet)


class TestGeneratedGraphics:
    @pytest.mark.parametrize("game", [
        "mega_sena",
        "lotofacil",
        "quina"
    ])
    def test_graphics_exist(self, game):
        graficos_dir = os.path.join(BASE_DIR, 'graficos')
        f1 = os.path.join(graficos_dir, f"{game}_dashboard_decisao.png")
        f2 = os.path.join(graficos_dir, f"{game}_avaliacao_acertos_erros.png")

        for f in [f1, f2]:
            assert os.path.exists(f), f"Arquivo ausente: {f}"
            assert os.path.getsize(f) > 50000, f"Arquivo corrompido ou vazio: {f}"
