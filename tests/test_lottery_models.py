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

    LotteryStochasticGradientBoostingModel,

    LotteryLogitGradientBoostingModel,

    LotteryBinomialDevianceGradientBoostingModel,

    LotteryResidualCyclingGradientBoostingModel,

    LotteryStackingModel,

    LotteryMomentumEnsembleModel,

    LotteryBalancedEnsembleModel,

    LotteryBatchGDEnsembleModel,

    LotteryMiniBatchGDEnsembleModel,

    LotteryStochasticGDEnsembleModel,

    LotteryBaggedLogitGBEnsembleModel,

    LotteryDualLogitGBEnsembleModel,

    LotteryPretrainedMetaCommitteeModel,

    LotterySuperEnsembleCommitteeModel,

    LotteryDeepEnsembleCommitteeModel,

    LotteryHighCommitteeBootstrappedMetaModel,

    predict_bagged_logit_gb_ensemble,

    predict_dual_logit_gb_ensemble,

    predict_pretrained_meta_committee,

    predict_deep_ensemble_committee,

    compare_committee_sizes,

    _extract_trained_ensemble_signals,

    predict_batch_gd_ensemble,

    predict_minibatch_gd_ensemble,

    predict_stochastic_gd_ensemble,

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

    predict_stochastic_gradient_boosting,

    predict_logit_gradient_boosting,

    predict_binomial_deviance_gradient_boosting,

    predict_residual_cycling_gradient_boosting,

    predict_stacking,

    predict_momentum_ensemble,

    predict_balanced_ensemble,

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



    def test_lottery_stochastic_gradient_boosting_model(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}

        m_sgb = LotteryStochasticGradientBoostingModel(n_estimators=5, learning_rate=0.5, subsample=0.7, random_state=42)

        m_sgb.fit(synthetic_data, cfg=cfg)



        top6 = m_sgb.predict_top_k(6)

        assert len(top6) == 6

        assert len(set(top6)) == 6

        assert all(1 <= d <= 60 for d in top6)



        staged = list(m_sgb.staged_predict_top_k(6))

        assert len(staged) == 6  # F_0 + 5 stages

        for stage_bet in staged:

            assert len(stage_bet) == 6

            assert len(set(stage_bet)) == 6

            assert all(1 <= d <= 60 for d in stage_bet)



        scores_df = m_sgb.get_scores_df()

        assert len(scores_df) == 60

        assert 'escore_stoch_gradboost' in scores_df.columns

        assert 'escore_acumulado' in scores_df.columns



        # Edge cases & validation

        assert len(m_sgb.predict_top_k(0)) == 0

        assert len(m_sgb.predict_top_k(70)) == 60



        with pytest.raises(ValueError):

            LotteryStochasticGradientBoostingModel(n_estimators=0)

        with pytest.raises(ValueError):

            LotteryStochasticGradientBoostingModel(learning_rate=0.0)

        with pytest.raises(ValueError):

            LotteryStochasticGradientBoostingModel(subsample=0.0)

        with pytest.raises(ValueError):

            LotteryStochasticGradientBoostingModel(subsample=1.5)

        with pytest.raises(ValueError):

            LotteryStochasticGradientBoostingModel().fit(np.zeros((0, 60)))



    def test_lottery_logit_gradient_boosting_model(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}

        m_lgb = LotteryLogitGradientBoostingModel(n_estimators=5, learning_rate=0.3, random_state=42)

        m_lgb.fit(synthetic_data, cfg=cfg)



        top6 = m_lgb.predict_top_k(6)

        assert len(top6) == 6

        assert len(set(top6)) == 6

        assert all(1 <= d <= 60 for d in top6)



        staged = list(m_lgb.staged_predict_top_k(6))

        assert len(staged) == 6  # F_0 + 5 stages

        for stage_bet in staged:

            assert len(stage_bet) == 6

            assert len(set(stage_bet)) == 6

            assert all(1 <= d <= 60 for d in stage_bet)



        scores_df = m_lgb.get_scores_df()

        assert len(scores_df) == 60

        assert 'escore_logit_gradboost' in scores_df.columns

        assert 'probabilidade_calibrada' in scores_df.columns

        assert 'logit_escore' in scores_df.columns

        assert all(0.0 <= p <= 1.0 for p in scores_df['probabilidade_calibrada'])



        # Edge cases & validation

        assert len(m_lgb.predict_top_k(0)) == 0

        assert len(m_lgb.predict_top_k(70)) == 60



        with pytest.raises(ValueError):

            LotteryLogitGradientBoostingModel(n_estimators=0)

        with pytest.raises(ValueError):

            LotteryLogitGradientBoostingModel(learning_rate=0.0)

        with pytest.raises(ValueError):

            LotteryLogitGradientBoostingModel().fit(np.zeros((0, 60)))



    def test_lottery_residual_cycling_gradient_boosting_model(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}

        m_rcgb = LotteryResidualCyclingGradientBoostingModel(n_estimators=8, learning_rate=0.4, random_state=42)

        m_rcgb.fit(synthetic_data, cfg=cfg)



        top6 = m_rcgb.predict_top_k(6)

        assert len(top6) == 6

        assert len(set(top6)) == 6

        assert all(1 <= d <= 60 for d in top6)



        assert len(m_rcgb.stage_models_) == 8

        assert set(m_rcgb.stage_models_) == {'ml_logistic', 'gianella', 'delay_momentum', 'bayes_adaptive'}



        staged = list(m_rcgb.staged_predict_top_k(6))

        assert len(staged) == 9  # F_0 + 8 stages

        for stage_bet in staged:

            assert len(stage_bet) == 6

            assert len(set(stage_bet)) == 6

            assert all(1 <= d <= 60 for d in stage_bet)



        scores_df = m_rcgb.get_scores_df()

        assert len(scores_df) == 60

        assert 'escore_cycling_gradboost' in scores_df.columns



        # Edge cases & validation

        assert len(m_rcgb.predict_top_k(0)) == 0

        assert len(m_rcgb.predict_top_k(70)) == 60



        with pytest.raises(ValueError):

            LotteryResidualCyclingGradientBoostingModel(n_estimators=0)

        with pytest.raises(ValueError):

            LotteryResidualCyclingGradientBoostingModel(learning_rate=0.0)

        with pytest.raises(ValueError):

            LotteryResidualCyclingGradientBoostingModel(base_cycle=[])

        with pytest.raises(ValueError):

            LotteryResidualCyclingGradientBoostingModel(base_cycle=['invalid_model'])

        with pytest.raises(ValueError):

            LotteryResidualCyclingGradientBoostingModel().fit(np.zeros((0, 60)))



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



    def test_lottery_momentum_ensemble_model(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60}

        m_mom = LotteryMomentumEnsembleModel(n_estimators=5, random_state=42)

        m_mom.fit(synthetic_data, cfg=cfg)



        top_soft = m_mom.predict_top_k(6, voting='soft')

        top_hard = m_mom.predict_top_k(6, voting='hard')



        assert len(top_soft) == 6

        assert len(set(top_soft)) == 6

        assert all(1 <= d <= 60 for d in top_soft)



        assert len(top_hard) == 6

        assert len(set(top_hard)) == 6

        assert all(1 <= d <= 60 for d in top_hard)



        assert isinstance(m_mom.oob_score_, float)

        assert m_mom.oob_score_ >= 0.0



        scores_df = m_mom.get_scores_df()

        assert len(scores_df) == 60

        assert 'escore_momentum_ensemble' in scores_df.columns



        # Validações de limites e erros

        assert len(m_mom.predict_top_k(0)) == 0

        assert len(m_mom.predict_top_k(70)) == 60

        with pytest.raises(ValueError):

            LotteryMomentumEnsembleModel(n_estimators=0)

        with pytest.raises(ValueError):

            LotteryMomentumEnsembleModel().fit(np.zeros((0, 60)))



    def test_lottery_balanced_ensemble_model(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60}

        m_bal = LotteryBalancedEnsembleModel(n_estimators=5, random_state=42)

        m_bal.fit(synthetic_data, cfg=cfg)



        top_soft = m_bal.predict_top_k(6, voting='soft')

        top_hard = m_bal.predict_top_k(6, voting='hard')



        assert len(top_soft) == 6

        assert len(set(top_soft)) == 6

        assert all(1 <= d <= 60 for d in top_soft)



        assert len(top_hard) == 6

        assert len(set(top_hard)) == 6

        assert all(1 <= d <= 60 for d in top_hard)



        assert isinstance(m_bal.oob_score_, float)

        assert m_bal.oob_score_ >= 0.0



        scores_df = m_bal.get_scores_df()

        assert len(scores_df) == 60

        assert 'escore_balanceado_ensemble' in scores_df.columns



        # Validações de limites e erros

        assert len(m_bal.predict_top_k(0)) == 0

        assert len(m_bal.predict_top_k(70)) == 60

        with pytest.raises(ValueError):

            LotteryBalancedEnsembleModel(n_estimators=-1)

        with pytest.raises(ValueError):

            LotteryBalancedEnsembleModel().fit(np.zeros((0, 60)))





    def test_lottery_batch_gd_ensemble_model(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60}

        m_bgd = LotteryBatchGDEnsembleModel(n_epochs=20, learning_rate=0.1, random_state=42)

        m_bgd.fit(synthetic_data, cfg=cfg)



        top6 = m_bgd.predict_top_k(6)

        assert len(top6) == 6

        assert len(set(top6)) == 6

        assert all(1 <= d <= 60 for d in top6)



        scores_df = m_bgd.get_scores_df()

        assert len(scores_df) == 60

        assert 'escore_batch_gd' in scores_df.columns

        assert len(m_bgd.loss_history_) == 20



        assert len(m_bgd.predict_top_k(0)) == 0

        assert len(m_bgd.predict_top_k(70)) == 60

        with pytest.raises(ValueError):

            LotteryBatchGDEnsembleModel().fit(np.zeros((0, 60)))



    def test_lottery_minibatch_gd_ensemble_model(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60}

        m_mbgd = LotteryMiniBatchGDEnsembleModel(n_epochs=15, batch_size=16, learning_rate=0.05, random_state=42)

        m_mbgd.fit(synthetic_data, cfg=cfg)



        top6 = m_mbgd.predict_top_k(6)

        assert len(top6) == 6

        assert len(set(top6)) == 6

        assert all(1 <= d <= 60 for d in top6)



        scores_df = m_mbgd.get_scores_df()

        assert len(scores_df) == 60

        assert 'escore_minibatch_gd' in scores_df.columns

        assert len(m_mbgd.loss_history_) == 15



        assert len(m_mbgd.predict_top_k(0)) == 0

        assert len(m_mbgd.predict_top_k(70)) == 60

        with pytest.raises(ValueError):

            LotteryMiniBatchGDEnsembleModel().fit(np.zeros((0, 60)))



    def test_lottery_bagged_logit_gb_ensemble_model(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60}

        m_bagged = LotteryBaggedLogitGBEnsembleModel(n_estimators=3, base_n_iterations=3, random_state=42)

        m_bagged.fit(synthetic_data, cfg=cfg)



        top6 = m_bagged.predict_top_k(6)

        assert len(top6) == 6

        assert len(set(top6)) == 6

        assert all(1 <= d <= 60 for d in top6)



        scores_df = m_bagged.get_scores_df()

        assert len(scores_df) == 60

        assert 'escore_meta_logit_gb' in scores_df.columns

        assert len(m_bagged.sub_models_) == 3



        assert len(m_bagged.predict_top_k(0)) == 0

        assert len(m_bagged.predict_top_k(70)) == 60

        with pytest.raises(ValueError):

            LotteryBaggedLogitGBEnsembleModel().fit(np.zeros((0, 60)))



    def test_lottery_dual_logit_gb_ensemble_model(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60}

        m_dual = LotteryDualLogitGBEnsembleModel(n_estimators_bagged=3, base_n_iterations=3, weight_base=0.45, random_state=42)

        m_dual.fit(synthetic_data, cfg=cfg)



        top6 = m_dual.predict_top_k(6)

        assert len(top6) == 6

        assert len(set(top6)) == 6

        assert all(1 <= d <= 60 for d in top6)



        scores_df = m_dual.get_scores_df()

        assert len(scores_df) == 60

        assert 'escore_dual_logit_gb' in scores_df.columns

        assert 'probabilidade_calibrada' in scores_df.columns



        assert len(m_dual.predict_top_k(0)) == 0

        assert len(m_dual.predict_top_k(70)) == 60

        with pytest.raises(ValueError):

            LotteryDualLogitGBEnsembleModel().fit(np.zeros((0, 60)))



    def test_lottery_stochastic_gd_ensemble_model(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60}

        m_sgd = LotteryStochasticGDEnsembleModel(n_epochs=10, learning_rate_0=0.05, random_state=42)

        m_sgd.fit(synthetic_data, cfg=cfg)



        top6 = m_sgd.predict_top_k(6)

        assert len(top6) == 6

        assert len(set(top6)) == 6

        assert all(1 <= d <= 60 for d in top6)



        scores_df = m_sgd.get_scores_df()

        assert len(scores_df) == 60

        assert 'escore_stochastic_gd' in scores_df.columns

        assert len(m_sgd.loss_history_) == 10



        assert len(m_sgd.predict_top_k(0)) == 0

        assert len(m_sgd.predict_top_k(70)) == 60

        with pytest.raises(ValueError):

            LotteryStochasticGDEnsembleModel().fit(np.zeros((0, 60)))



    def test_lottery_pretrained_meta_committee_model(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}

        # Pre-train compact ensembles to test direct reuse

        pre_lgb = LotteryLogitGradientBoostingModel(n_estimators=4, learning_rate=0.25, random_state=42).fit(synthetic_data, cfg)

        pre_rcgb = LotteryResidualCyclingGradientBoostingModel(n_estimators=4, random_state=42).fit(synthetic_data, cfg)

        pre_mom = LotteryMomentumEnsembleModel(n_estimators=4, random_state=42).fit(synthetic_data, cfg)

        pre_bal = LotteryBalancedEnsembleModel(n_estimators=4, random_state=42).fit(synthetic_data, cfg)

        trained_dict = {

            'Logit GB': pre_lgb,

            'Residual Cycling GB': pre_rcgb,

            'Momentum Ensemble': pre_mom,

            'Balanced Ensemble': pre_bal,

        }

        m50 = LotteryPretrainedMetaCommitteeModel(n_committees=50, random_state=42).fit(
            synthetic_data, cfg, trained_ensembles=trained_dict
        )

        m100 = LotterySuperEnsembleCommitteeModel(n_committees=100, random_state=42).fit(
            synthetic_data, cfg, trained_ensembles=trained_dict
        )

        assert m50.n_committees == 50

        assert m100.n_committees == 100

        assert len(m50.base_ensemble_names_) == 4

        assert m50.sub_estimator_count_ >= 4

        assert len(m50.committee_members_) == 50

        assert len(m100.committee_members_) == 100

        assert len(m50.convergence_curve_) == 50

        assert len(m100.convergence_curve_) == 100

        assert sum(m50.ensemble_contributions_.values()) == pytest.approx(1.0, abs=1e-4)

        assert np.isfinite(m50.mean_se_) and np.isfinite(m100.mean_se_)

        assert m100.mean_se_ <= m50.mean_se_ * 1.15 + 1e-6

        for voting_mode in ('hybrid', 'soft', 'borda', 'hard'):

            bet50 = m50.predict_top_k(6, voting=voting_mode)

            bet100 = m100.predict_top_k(6, voting=voting_mode)

            assert len(bet50) == 6 and len(set(bet50)) == 6

            assert len(bet100) == 6 and len(set(bet100)) == 6

            assert all(1 <= d <= 60 for d in bet50 + bet100)

        df_scores = m100.get_scores_df()

        assert len(df_scores) == 60

        for col in ['dezena', 'escore_meta_comite', 'probabilidade_comite', 'escore_soft', 'escore_borda', 'votos_hard', 'taxa_consenso_pct', 'incerteza_se']:

            assert col in df_scores.columns

        # Determinism check

        m50_dup = LotteryPretrainedMetaCommitteeModel(n_committees=50, random_state=42).fit(
            synthetic_data, cfg, trained_ensembles=trained_dict
        )

        assert m50.predict_top_k(6) == m50_dup.predict_top_k(6)

        # Standalone mode (trained_ensembles=None)

        m50_standalone = LotteryPretrainedMetaCommitteeModel(n_committees=50, random_state=42).fit(
            synthetic_data, cfg, trained_ensembles=None
        )

        assert len(m50_standalone.base_ensemble_names_) >= 3

        assert len(m50_standalone.predict_top_k(6)) == 6

        # Multi-lottery shapes (Lotofacil 25/15 and Quina 80/5)

        rng = np.random.RandomState(99)

        mat_lf = np.zeros((40, 25), dtype=int)

        for i in range(40):

            mat_lf[i, rng.choice(25, size=15, replace=False)] = 1

        cfg_lf = {'k': 15, 'n_dezenas': 25, 'alpha_0': 1.0, 'beta_0': 1.0, 'gianella_group_size': 5}

        lf_m50 = LotteryPretrainedMetaCommitteeModel(n_committees=50, random_state=7).fit(mat_lf, cfg_lf)

        assert len(lf_m50.predict_top_k(15)) == 15

        mat_qn = np.zeros((40, 80), dtype=int)

        for i in range(40):

            mat_qn[i, rng.choice(80, size=5, replace=False)] = 1

        cfg_qn = {'k': 5, 'n_dezenas': 80, 'alpha_0': 1.0, 'beta_0': 15.0, 'gianella_group_size': 10}

        qn_m50 = LotteryPretrainedMetaCommitteeModel(n_committees=50, random_state=7).fit(mat_qn, cfg_qn)

        assert len(qn_m50.predict_top_k(5)) == 5

        # Edge cases: small sample & zero matrix

        small_m = LotteryPretrainedMetaCommitteeModel(n_committees=50, random_state=42).fit(synthetic_data[:5], cfg)

        assert len(small_m.predict_top_k(6)) == 6

        one_m = LotteryPretrainedMetaCommitteeModel(n_committees=10, random_state=42).fit(synthetic_data[:1], cfg)

        assert len(one_m.predict_top_k(6)) == 6

        zero_m = LotteryPretrainedMetaCommitteeModel(n_committees=50, random_state=42).fit(np.zeros((20, 60), dtype=int), cfg)

        assert len(zero_m.predict_top_k(6)) == 6

        # Regression: top_k_ must respect configured self.voting across all modes

        for v_mode in ('hybrid', 'soft', 'borda', 'hard'):

            mv = LotteryPretrainedMetaCommitteeModel(n_committees=30, voting=v_mode, random_state=42).fit(
                synthetic_data, cfg, trained_ensembles=trained_dict
            )

            assert mv.top_k_ == mv.predict_top_k(6)

        # Regression: temperature_range actively modulates thermodynamic calibration in hybrid scores

        m_cold = LotteryPretrainedMetaCommitteeModel(
            n_committees=30, temperature_range=(0.05, 0.10), random_state=42
        ).fit(synthetic_data, cfg, trained_ensembles=trained_dict)

        m_hot = LotteryPretrainedMetaCommitteeModel(
            n_committees=30, temperature_range=(5.0, 10.0), random_state=42
        ).fit(synthetic_data, cfg, trained_ensembles=trained_dict)

        df_c = m_cold.get_scores_df().sort_values('dezena')['escore_meta_comite'].values

        df_h = m_hot.get_scores_df().sort_values('dezena')['escore_meta_comite'].values

        assert float(np.max(np.abs(df_c - df_h))) > 1e-3

        # Regression: ensembles lacking oob_score_ receive neutral prior rather than 0.0 penalty

        _, _, oob_mixed, _, _ = _extract_trained_ensemble_signals(
            {'Logit GB': pre_lgb, 'Momentum Ensemble': pre_mom}, 60, include_sub_estimators=True
        )

        assert len(oob_mixed) == 2 and oob_mixed[0] == pytest.approx(0.5, abs=1e-6) and oob_mixed[1] == pytest.approx(0.5, abs=1e-6)

        # Error checks

        with pytest.raises(RuntimeError):

            LotteryPretrainedMetaCommitteeModel(n_committees=50).predict_top_k(6)

        with pytest.raises(ValueError):

            LotteryPretrainedMetaCommitteeModel(n_committees=50).fit(np.zeros((0, 60)))

        with pytest.raises(ValueError):

            LotteryPretrainedMetaCommitteeModel(n_committees=0).fit(synthetic_data, cfg)

        with pytest.raises(ValueError):

            m50.predict_top_k(6, voting='invalid_mode')



    def test_lottery_deep_ensemble_committee_model(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}

        pre_lgb = LotteryLogitGradientBoostingModel(n_estimators=4, learning_rate=0.25, random_state=42).fit(synthetic_data, cfg)

        pre_mom = LotteryMomentumEnsembleModel(n_estimators=4, random_state=42).fit(synthetic_data, cfg)

        trained_dict = {'Logit GB': pre_lgb, 'Momentum Ensemble': pre_mom}

        d50 = LotteryDeepEnsembleCommitteeModel(n_committees=50, random_state=42).fit(
            synthetic_data, cfg, trained_ensembles=trained_dict
        )

        d100 = LotteryHighCommitteeBootstrappedMetaModel(n_committees=100, random_state=42).fit(
            synthetic_data, cfg, trained_ensembles=trained_dict
        )

        assert len(d50.estimators_) == 50

        assert len(d100.estimators_) == 100

        assert len(d50.convergence_curve_) == 50

        assert len(d100.convergence_curve_) == 100

        assert d50.oob_score_ is not None and 0.0 <= d50.oob_score_ <= 6.0

        assert d100.oob_score_ is not None and 0.0 <= d100.oob_score_ <= 6.0

        assert len(d50.family_oob_scores_) >= 2

        assert np.isfinite(d50.mean_se_) and np.isfinite(d100.mean_se_)

        assert d100.mean_se_ <= d50.mean_se_ * 1.15 + 1e-6

        for voting_mode in ('hybrid', 'soft', 'hard'):

            bet50 = d50.predict_top_k(6, voting=voting_mode)

            bet100 = d100.predict_top_k(6, voting=voting_mode)

            assert len(bet50) == 6 and len(set(bet50)) == 6

            assert len(bet100) == 6 and len(set(bet100)) == 6

            assert all(1 <= d <= 60 for d in bet50 + bet100)

        # Regression: top_k_ must respect configured self.voting across all modes

        for v_mode in ('hybrid', 'soft', 'hard'):

            dv = LotteryDeepEnsembleCommitteeModel(n_committees=24, voting=v_mode, random_state=42).fit(
                synthetic_data, cfg, trained_ensembles=trained_dict
            )

            assert dv.top_k_ == dv.predict_top_k(6)

        df_scores = d100.get_scores_df()

        assert len(df_scores) == 60

        for col in ['dezena', 'escore_comite_profundo', 'escore_comite_puro', 'escore_ancora', 'votos_hard', 'taxa_consenso_pct', 'incerteza_se']:

            assert col in df_scores.columns

        # Determinism check

        d50_dup = LotteryDeepEnsembleCommitteeModel(n_committees=50, random_state=42).fit(
            synthetic_data, cfg, trained_ensembles=trained_dict
        )

        assert d50.predict_top_k(6) == d50_dup.predict_top_k(6)

        # Standalone mode without anchor

        d50_no_anchor = LotteryDeepEnsembleCommitteeModel(
            n_committees=50, anchor_weight=0.0, oob_score=False, random_state=42
        ).fit(synthetic_data, cfg, trained_ensembles=None)

        assert len(d50_no_anchor.predict_top_k(6)) == 6

        # Multi-lottery shapes (Lotofacil 25/15 & Quina 80/5)

        rng = np.random.RandomState(101)

        mat_lf = np.zeros((40, 25), dtype=int)

        for i in range(40):

            mat_lf[i, rng.choice(25, size=15, replace=False)] = 1

        cfg_lf = {'k': 15, 'n_dezenas': 25, 'alpha_0': 1.0, 'beta_0': 1.0, 'gianella_group_size': 5}

        lf_deep = LotteryDeepEnsembleCommitteeModel(n_committees=50, random_state=7).fit(mat_lf, cfg_lf)

        assert len(lf_deep.predict_top_k(15)) == 15

        assert 0.0 <= lf_deep.oob_score_ <= 15.0

        mat_qn = np.zeros((40, 80), dtype=int)

        for i in range(40):

            mat_qn[i, rng.choice(80, size=5, replace=False)] = 1

        cfg_qn = {'k': 5, 'n_dezenas': 80, 'alpha_0': 1.0, 'beta_0': 15.0, 'gianella_group_size': 10}

        qn_deep = LotteryDeepEnsembleCommitteeModel(n_committees=50, random_state=7).fit(mat_qn, cfg_qn)

        assert len(qn_deep.predict_top_k(5)) == 5

        assert 0.0 <= qn_deep.oob_score_ <= 5.0

        # Edge cases: small sample, gd_bootstrap native vs n=1 fallback attribution, partial n_oob==0 & zero matrix

        assert len(LotteryDeepEnsembleCommitteeModel(n_committees=50, random_state=42).fit(synthetic_data[:5], cfg).predict_top_k(6)) == 6

        gd_n5 = LotteryDeepEnsembleCommitteeModel(
            n_committees=6, families=['gd_bootstrap'], random_state=42
        ).fit(synthetic_data[:5], cfg, trained_ensembles=trained_dict)

        assert all(e['family'] == 'gd_bootstrap' for e in gd_n5.estimators_)

        gd_n1 = LotteryDeepEnsembleCommitteeModel(
            n_committees=4, families=['gd_bootstrap'], random_state=42
        ).fit(synthetic_data[:1], cfg, trained_ensembles=trained_dict)

        assert all(e['family'] == 'bayes_gianella_bootstrap' for e in gd_n1.estimators_)

        d_oob_w = LotteryDeepEnsembleCommitteeModel(
            n_committees=30, max_samples=1.0, weight_by_oob=True, random_state=42
        ).fit(synthetic_data[:3], cfg, trained_ensembles=trained_dict)

        d_oob_u = LotteryDeepEnsembleCommitteeModel(
            n_committees=30, max_samples=1.0, weight_by_oob=False, random_state=42
        ).fit(synthetic_data[:3], cfg, trained_ensembles=trained_dict)

        assert float(np.max(np.abs(
            d_oob_w.get_scores_df().sort_values('dezena')['escore_comite_puro'].values -
            d_oob_u.get_scores_df().sort_values('dezena')['escore_comite_puro'].values
        ))) > 1e-4

        assert len(LotteryDeepEnsembleCommitteeModel(n_committees=50, random_state=42).fit(np.zeros((20, 60), dtype=int), cfg).predict_top_k(6)) == 6

        # Error checks

        with pytest.raises(RuntimeError):

            LotteryDeepEnsembleCommitteeModel(n_committees=50).predict_top_k(6)

        with pytest.raises(ValueError):

            LotteryDeepEnsembleCommitteeModel(n_committees=50).fit(np.zeros((0, 60)))

        with pytest.raises(ValueError):

            LotteryDeepEnsembleCommitteeModel(n_committees=0).fit(synthetic_data, cfg)

        with pytest.raises(ValueError):

            LotteryDeepEnsembleCommitteeModel(n_committees=50, max_samples=1.5).fit(synthetic_data, cfg)

        with pytest.raises(ValueError):

            LotteryDeepEnsembleCommitteeModel(n_committees=50, anchor_weight=1.5).fit(synthetic_data, cfg)

        with pytest.raises(ValueError):

            d50.predict_top_k(6, voting='invalid_mode')



    def test_compare_committee_sizes(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}

        pre_lgb = LotteryLogitGradientBoostingModel(n_estimators=4, learning_rate=0.25, random_state=42).fit(synthetic_data, cfg)

        pre_mom = LotteryMomentumEnsembleModel(n_estimators=4, random_state=42).fit(synthetic_data, cfg)

        df_cmp = compare_committee_sizes(
            synthetic_data,
            cfg,
            sizes=(10, 50, 100),
            trained_ensembles={'Logit GB': pre_lgb, 'Momentum Ensemble': pre_mom},
            random_state=42,
        )

        assert isinstance(df_cmp, pd.DataFrame)

        assert len(df_cmp) == 6

        assert set(df_cmp['n_committees'].tolist()) == {10, 50, 100}

        assert df_cmp['Arquitetura'].nunique() == 2

        for col in ['Arquitetura', 'n_committees', 'top_k', 'oob_score', 'mean_se', 'tail_jaccard_stability', 'fit_time_s']:

            assert col in df_cmp.columns

        assert all(len(tk) == 6 for tk in df_cmp['top_k'])

        meta_rows = df_cmp[df_cmp['Arquitetura'].str.contains('Meta')].sort_values('n_committees')

        assert meta_rows.iloc[-1]['mean_se'] <= meta_rows.iloc[0]['mean_se'] * 1.15 + 1e-6

        with pytest.raises(ValueError):

            compare_committee_sizes(synthetic_data, cfg, sizes=())

        with pytest.raises(ValueError):

            compare_committee_sizes(synthetic_data, cfg, sizes=(0, 50))

        with pytest.raises(ValueError):

            compare_committee_sizes(np.zeros((0, 60)), cfg, sizes=(50,))



    def test_ensemble_helper_functions(self, synthetic_data):

        cfg = {'k': 6, 'n_dezenas': 60, 'alpha_0': 1.0, 'beta_0': 9.0, 'gianella_group_size': 10}

        p_bag = predict_bagging(synthetic_data, cfg, k=6, n_estimators=3)

        p_paste = predict_pasting(synthetic_data, cfg, k=6, n_estimators=3)

        p_ada = predict_adaboost(synthetic_data, cfg, k=6, n_estimators=3)

        p_gb = predict_gradient_boosting(synthetic_data, cfg, k=6, n_estimators=3)

        p_sgb = predict_stochastic_gradient_boosting(synthetic_data, cfg, k=6, n_estimators=3)

        p_lgb = predict_logit_gradient_boosting(synthetic_data, cfg, k=6, n_estimators=3)

        p_bdgb = predict_binomial_deviance_gradient_boosting(synthetic_data, cfg, k=6, n_estimators=3)

        p_rcgb = predict_residual_cycling_gradient_boosting(synthetic_data, cfg, k=6, n_estimators=4)

        p_stack = predict_stacking(synthetic_data, cfg, k=6)

        p_mom_ens = predict_momentum_ensemble(synthetic_data, cfg, k=6, n_estimators=3)

        p_bal_ens = predict_balanced_ensemble(synthetic_data, cfg, k=6, n_estimators=3)

        p_batch_gd = predict_batch_gd_ensemble(synthetic_data, cfg, k=6, n_epochs=10)

        p_minibatch_gd = predict_minibatch_gd_ensemble(synthetic_data, cfg, k=6, n_epochs=10, batch_size=16)

        p_stoch_gd = predict_stochastic_gd_ensemble(synthetic_data, cfg, k=6, n_epochs=5)

        p_bagged_lgb = predict_bagged_logit_gb_ensemble(synthetic_data, cfg, k=6, n_estimators=3, base_n_iterations=3)

        p_dual_lgb = predict_dual_logit_gb_ensemble(synthetic_data, cfg, k=6, n_estimators_bagged=3, base_n_iterations=3)

        pre_dict = {

            'Logit GB': LotteryLogitGradientBoostingModel(n_estimators=3, random_state=42).fit(synthetic_data, cfg),

            'Momentum Ensemble': LotteryMomentumEnsembleModel(n_estimators=3, random_state=42).fit(synthetic_data, cfg),

        }

        p_meta_default = predict_pretrained_meta_committee(synthetic_data, cfg, k=6, trained_ensembles=pre_dict)

        p_meta_50 = predict_pretrained_meta_committee(synthetic_data, cfg, k=6, n_committees=50, trained_ensembles=pre_dict)

        p_meta_100 = predict_pretrained_meta_committee(synthetic_data, cfg, k=6, n_committees=100, trained_ensembles=pre_dict)

        p_deep_default = predict_deep_ensemble_committee(synthetic_data, cfg, k=6, trained_ensembles=pre_dict)

        p_deep_50 = predict_deep_ensemble_committee(synthetic_data, cfg, k=6, n_committees=50, trained_ensembles=pre_dict)

        p_deep_100 = predict_deep_ensemble_committee(synthetic_data, cfg, k=6, n_committees=100, trained_ensembles=pre_dict)

        assert p_meta_default == p_meta_100

        assert p_deep_default == p_deep_100

        assert LotteryPretrainedMetaCommitteeModel().n_committees == 100

        assert LotteryDeepEnsembleCommitteeModel().n_committees == 100



        for name, bet in [('bagging', p_bag), ('pasting', p_paste), ('adaboost', p_ada),

                          ('gradboost', p_gb), ('stoch_gb', p_sgb), ('logit_gb', p_lgb),

                          ('binomial_deviance_gb', p_bdgb), ('residual_cycling_gb', p_rcgb),

                          ('stacking', p_stack), ('momentum_ensemble', p_mom_ens), ('balanced_ensemble', p_bal_ens),

                          ('batch_gd', p_batch_gd), ('minibatch_gd', p_minibatch_gd), ('stochastic_gd', p_stoch_gd),

                          ('bagged_logit_gb', p_bagged_lgb), ('dual_logit_gb', p_dual_lgb),

                          ('pretrained_meta_50', p_meta_50), ('pretrained_meta_100', p_meta_100),

                          ('deep_committee_50', p_deep_50), ('deep_committee_100', p_deep_100)]:

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

                         'm_gradboost', 'm_stoch_gb', 'm_logit_gb', 'm_cycle_gb', 'm_stacking', 'm_mom_ens', 'm_bal_ens',

                         'm_batch_gd', 'm_minibatch_gd', 'm_stoch_gd', 'm_bagged_logit_gb', 'm_dual_logit_gb',

                         'm_meta_committee_100', 'm_deep_committee_100',

                         'predictions', 'predictions_ensemble', 'comparison_df']

        for k in required_keys:

            assert k in results, f"Chave {k} ausente em results"

            

        assert results['cfg']['k'] == expected_k

        assert results['cfg']['n_dezenas'] == expected_n

        assert len(results['predictions']) == 7

        assert len(results['comparison_df']) == 7

        assert len(results['predictions_ensemble']) == 17

        

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

                         'm_adaboost', 'm_gradboost', 'm_stoch_gb', 'm_logit_gb', 'm_cycle_gb', 'm_stacking', 'm_mom_ens', 'm_bal_ens',

                         'm_batch_gd', 'm_minibatch_gd', 'm_stoch_gd', 'm_bagged_logit_gb', 'm_dual_logit_gb',

                         'm_meta_committee_100', 'm_deep_committee_100',

                         'pred_consenso', 'consensus_scores', 'predictions', 'predictions_hybrid',

                         'predictions_ensemble', 'comparison_df']

        for k in required_keys:

            assert k in results, f"Chave {k} ausente em results"



        assert len(results['predictions']) == 40

        assert len(results['comparison_df']) == 40

        assert len(results['predictions_hybrid']) == 15

        assert len(results['predictions_ensemble']) == 17



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

