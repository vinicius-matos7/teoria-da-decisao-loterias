"""
tests/test_backtest_monitor.py
==============================
Testes automatizados para o sistema de monitoramento e comparação
de previsões de loterias (Walk-Forward Backtesting).
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd

# Adiciona o diretório raiz ao path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.monitor_comparativo_previsoes import (
    calculate_structural_metrics,
    get_theoretical_benchmarks,
    calculate_statistical_significance,
    get_prize_tier,
    RandomBaselineModel,
    LotteryBacktestMonitor,
    run_monitoring_comparison,
    PRIZE_TIERS
)


class TestStructuralMetrics:
    def test_metrics_calculation_deterministic(self):
        # Bilhete fixo de 6 números da Mega-Sena: [1, 2, 3, 10, 20, 30]
        numbers = [1, 2, 3, 10, 20, 30]
        n_dezenas = 60
        delays = np.zeros(60)
        delays[0] = 5    # dezena 1: atraso 5
        delays[1] = 10   # dezena 2: atraso 10
        delays[2] = 15   # dezena 3: atraso 15
        delays[9] = 20   # dezena 10: atraso 20
        delays[19] = 25  # dezena 20: atraso 25
        delays[29] = 30  # dezena 30: atraso 30

        metrics = calculate_structural_metrics(numbers, n_dezenas, delays)

        # 1. Paridade: 2, 10, 20, 30 são pares (4 pares, 2 ímpares)
        assert metrics['even_count'] == 4
        assert metrics['odd_count'] == 2

        # 2. Soma: 1 + 2 + 3 + 10 + 20 + 30 = 66
        assert metrics['sum_numbers'] == 66

        # 3. Amplitude: 30 - 1 = 29
        assert metrics['spread_amplitude'] == 29

        # 4. Consecutivos: (1, 2) e (2, 3) -> 2 pares adjacentes
        assert metrics['consecutive_count'] == 2

        # 5. Atraso médio: (5 + 10 + 15 + 20 + 25 + 30) / 6 = 105 / 6 = 17.5
        assert metrics['avg_delay'] == 17.5

        # 6. Gianella Cores:
        # Zona 1 (1 a 15): 1, 2, 3, 10 (4 dezenas)
        # Zona 2 (16 a 30): 20, 30 (2 dezenas)
        # Zona 3 (31 a 45): 0 dezenas
        # Zona 4 (46 a 60): 0 dezenas
        assert metrics['gianella_counts'] == (4, 2, 0, 0)
        assert metrics['gianella_diophantine'] == (4, 2, 0, 0)
        assert metrics['gianella_entropy'] > 0.0

    def test_metrics_empty(self):
        metrics = calculate_structural_metrics([], 60)
        assert metrics['even_count'] == 0
        assert metrics['sum_numbers'] == 0
        assert metrics['consecutive_count'] == 0
        assert metrics['avg_delay'] == 0.0


class TestTheoreticalBenchmarks:
    def test_megasena_benchmarks(self):
        b = get_theoretical_benchmarks('megasena', 60, 6)
        # E[Hits] = 6^2 / 60 = 0.60
        assert pytest.approx(b['expected_hits'], 0.001) == 0.60
        # E[Sum] = 6 * 61 / 2 = 183.0
        assert pytest.approx(b['expected_sum'], 0.001) == 183.0
        # E[Even] = 6 * 30 / 60 = 3.0
        assert pytest.approx(b['expected_even'], 0.001) == 3.0
        # E[Delay] pré-sorteio = (60 - 6) / 6 = 9.0; E[Gap] = 60 / 6 = 10.0
        assert pytest.approx(b['expected_delay'], 0.001) == 9.0
        assert pytest.approx(b['expected_gap'], 0.001) == 10.0

    def test_lotofacil_benchmarks(self):
        b = get_theoretical_benchmarks('lotofacil', 25, 15)
        # E[Hits] = 15^2 / 25 = 225 / 25 = 9.00
        assert pytest.approx(b['expected_hits'], 0.001) == 9.00
        # E[Sum] = 15 * 26 / 2 = 195.0
        assert pytest.approx(b['expected_sum'], 0.001) == 195.0
        # E[Delay] = (25 - 15) / 15 = 10/15; E[Gap] = 25 / 15
        assert pytest.approx(b['expected_delay'], 0.001) == 10.0 / 15.0
        assert pytest.approx(b['expected_gap'], 0.001) == 25.0 / 15.0

    def test_quina_benchmarks(self):
        b = get_theoretical_benchmarks('quina', 80, 5)
        # E[Hits] = 25 / 80 = 0.3125
        assert pytest.approx(b['expected_hits'], 0.0001) == 0.3125
        # E[Sum] = 5 * 81 / 2 = 202.5
        assert pytest.approx(b['expected_sum'], 0.001) == 202.5
        # E[Delay] = (80 - 5) / 5 = 15.0; E[Gap] = 80 / 5 = 16.0
        assert pytest.approx(b['expected_delay'], 0.001) == 15.0
        assert pytest.approx(b['expected_gap'], 0.001) == 16.0


class TestStatisticalSignificance:
    def test_identical_performance(self):
        # Desempenho exatamente igual ao baseline
        hits_m = np.array([1, 0, 1, 2, 0, 1, 0, 2])
        hits_b = np.array([1, 0, 1, 2, 0, 1, 0, 2])
        res = calculate_statistical_significance(hits_m, hits_b, theoretical_mean=0.6)
        assert res['diff_vs_baseline'] == 0.0
        assert res['paired_p_value'] == 1.0
        assert not res['significant_vs_baseline']

    def test_significantly_better_performance(self):
        # Desempenho estatisticamente muito superior
        hits_m = np.array([3, 4, 3, 5, 4, 3, 4, 5, 4, 4])
        hits_b = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
        res = calculate_statistical_significance(hits_m, hits_b, theoretical_mean=0.6)
        assert res['diff_vs_baseline'] > 0
        assert res['paired_p_value'] < 0.01
        assert res['significant_vs_baseline']
        assert res['cohens_d'] > 0

    def test_empty_sample(self):
        res = calculate_statistical_significance(np.array([]), np.array([]), 0.6)
        assert res['mean_hits'] == 0.0
        assert not res['significant_vs_baseline']


class TestPrizeTiers:
    def test_all_game_tiers(self):
        # Mega-Sena
        assert "Sena" in get_prize_tier('megasena', 6)
        assert "Quina" in get_prize_tier('megasena', 5)
        assert "Quadra" in get_prize_tier('megasena', 4)
        assert get_prize_tier('megasena', 3) == "Nenhum"

        # Lotofácil
        assert "15" in get_prize_tier('lotofacil', 15)
        assert "14" in get_prize_tier('lotofacil', 14)
        assert "11" in get_prize_tier('lotofacil', 11)
        assert get_prize_tier('lotofacil', 10) == "Nenhum"

        # Quina
        assert "Quina" in get_prize_tier('quina', 5)
        assert "Duque" in get_prize_tier('quina', 2)
        assert get_prize_tier('quina', 1) == "Nenhum"

    def test_casing_and_accents_normalization(self):
        # Validação de robustez a maiúsculas, hífens e acentos
        assert "15" in get_prize_tier('Lotofacil', 15)
        assert "14" in get_prize_tier('lotofácil', 14)
        assert "Duque" in get_prize_tier('Quina', 2)
        assert "Sena" in get_prize_tier('mega_sena', 6)
        assert "Sena" in get_prize_tier('Mega-Sena', 6)


class TestRandomBaselineModel:
    def test_random_generation(self):
        bm = RandomBaselineModel(n_dezenas=60, k=6, seed=123)
        bet1 = bm.predict()
        assert len(bet1) == 6
        assert len(set(bet1)) == 6
        assert all(1 <= x <= 60 for x in bet1)

        # Reprodutibilidade com mesma semente
        bm2 = RandomBaselineModel(n_dezenas=60, k=6, seed=123)
        bet2 = bm2.predict()
        assert bet1 == bet2


class TestLotteryBacktestMonitorExecution:
    @pytest.fixture
    def synthetic_monitor(self):
        """Cria um monitor com base sintética controlada (125 concursos)."""
        np.random.seed(42)
        n_concursos = 125
        n_dezenas = 60
        k = 6

        matrix = np.zeros((n_concursos, n_dezenas), dtype=int)
        for i in range(n_concursos):
            draw = np.random.choice(n_dezenas, size=k, replace=False)
            matrix[i, draw] = 1

        df = pd.DataFrame({'Concurso': np.arange(1, n_concursos + 1)})
        for col_idx in range(k):
            df[f'bola {col_idx + 1}'] = 0
        for i in range(n_concursos):
            drawn = np.where(matrix[i] == 1)[0] + 1
            for col_idx, num in enumerate(drawn):
                df.at[i, f'bola {col_idx + 1}'] = num

        monitor = LotteryBacktestMonitor(game='megasena')
        monitor.df = df
        monitor.binary_matrix = matrix
        monitor.n_concursos = n_concursos
        return monitor

    def test_walk_forward_synthetic(self, synthetic_monitor):
        # Roda walk-forward para os últimos 5 concursos
        n_test = 5
        models = [
            'Freq/Atraso (Balanceado)',
            'Freq/Atraso (Mais Atrasadas)',
            'Bayesiano Global (Beta-Binomial)',
            'Bayesiano Dinâmico Adaptativo',
            'Aposta Aleatória Uniforme'
        ]
        synthetic_monitor.run_walk_forward(n_contests=n_test, models_to_run=models, seed=42, verbose=False)

        assert len(synthetic_monitor.tested_contests_) == n_test
        assert synthetic_monitor.raw_results_df is not None
        assert len(synthetic_monitor.raw_results_df) == n_test * len(models)

        # Validação do sumário de acertos
        hits_summary = synthetic_monitor.get_hits_summary()
        assert len(hits_summary) == len(models)
        required_hits_cols = ['Modelo', 'Acertos Médios', 'Desvio Padrão', 'IC 95% Inf', 'IC 95% Sup', 'Acertos Acumulados']
        for col in required_hits_cols:
            assert col in hits_summary.columns

        # Validação do sumário estrutural
        struct_summary = synthetic_monitor.get_characteristics_summary()
        assert 'Sorteio Real' in struct_summary['Origem / Modelo'].values
        assert 'Benchmark Teórico Combinatório' in struct_summary['Origem / Modelo'].values
        required_struct_cols = ['Origem / Modelo', 'Pares Médios', 'Soma Média', 'Amplitude Média', 'Consecutivos Médios', 'Atraso Médio']
        for col in required_struct_cols:
            assert col in struct_summary.columns

        # Validação de testes estatísticos
        stats_summary = synthetic_monitor.get_statistical_tests()
        assert len(stats_summary) == len(models)
        assert 'p-valor (t-pareado)' in stats_summary.columns
        assert 'Cohen d' in stats_summary.columns

        # Validação da geração de relatório Markdown
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            report_str = synthetic_monitor.generate_markdown_report(output_file=tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 500
            assert "Relatório de Monitoramento" in report_str
            assert "Testes Formais de Significância" in report_str
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Validação da geração do gráfico consolidado
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            tmp_img_path = tmp_img.name

        try:
            fig = synthetic_monitor.plot_dashboard(save_path=tmp_img_path, show=False)
            assert os.path.exists(tmp_img_path)
            assert os.path.getsize(tmp_img_path) > 20000  # Imagem PNG válida e preenchida
            import matplotlib.pyplot as plt
            plt.close(fig)
        finally:
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)


class TestLotteryBacktestMonitorIntegration:
    @pytest.mark.parametrize("game", ["megasena", "lotofacil", "quina"])
    def test_walk_forward_real_subset(self, game):
        """Valida a integração real para cada uma das 3 loterias em um walk-forward curto de 3 concursos."""
        monitor = LotteryBacktestMonitor(game=game)
        # Roda modelos analíticos rápidos para validação ponta a ponta
        fast_models = [
            'Freq/Atraso (Balanceado)',
            'Bayesiano Global (Beta-Binomial)',
            'Aposta Aleatória Uniforme'
        ]
        monitor.run_walk_forward(n_contests=3, models_to_run=fast_models, seed=99, verbose=False)

        hits_summary = monitor.get_hits_summary()
        assert len(hits_summary) == len(fast_models)
        assert all(hits_summary['Acertos Médios'] >= 0)

        struct_summary = monitor.get_characteristics_summary()
        assert len(struct_summary) >= len(fast_models) + 2  # Sorteio Real + Modelos + Benchmark Teórico


class TestGeneratedMonitorGraphics:
    @pytest.mark.parametrize("game", ["megasena", "lotofacil", "quina"])
    def test_monitor_graphics_exist(self, game):
        graficos_dir = os.path.join(BASE_DIR, 'graficos')
        png_file = os.path.join(graficos_dir, f"monitor_comparativo_{game}.png")
        md_file = os.path.join(graficos_dir, f"relatorio_monitor_{game}.md")

        assert os.path.exists(png_file), f"Arquivo de gráfico ausente: {png_file}"
        assert os.path.getsize(png_file) > 100000, f"Arquivo corrompido ou muito pequeno: {png_file}"

        assert os.path.exists(md_file), f"Arquivo de relatório ausente: {md_file}"
        assert os.path.getsize(md_file) > 1000, f"Relatório corrompido ou vazio: {md_file}"


class TestZeroLookaheadBias:
    def test_strict_temporal_isolation(self):
        """
        Verifica matematicamente que o treino no concurso t não tem acesso a dados de t ou posteriores.
        """
        np.random.seed(42)
        n_concursos = 120
        n_dezenas = 60
        k = 6

        matrix = np.zeros((n_concursos, n_dezenas), dtype=int)
        for i in range(n_concursos):
            draw = np.random.choice(n_dezenas, size=k, replace=False)
            matrix[i, draw] = 1

        df = pd.DataFrame({'Concurso': np.arange(1, n_concursos + 1)})
        for col_idx in range(k):
            df[f'bola {col_idx + 1}'] = 0
        for i in range(n_concursos):
            drawn = np.where(matrix[i] == 1)[0] + 1
            for col_idx, num in enumerate(drawn):
                df.at[i, f'bola {col_idx + 1}'] = num

        monitor = LotteryBacktestMonitor(game='megasena')
        monitor.df = df.copy()
        monitor.binary_matrix = matrix.copy()
        monitor.n_concursos = n_concursos

        # Execução 1 original
        monitor.run_walk_forward(n_contests=3, models_to_run=['Bayesiano Global (Beta-Binomial)'], verbose=False)
        preds_1 = monitor.raw_results_df['dezenas'].tolist()

        # Modifica drasticamente o último concurso no dataset original (t = n_concursos - 1)
        matrix_tampered = matrix.copy()
        matrix_tampered[-1, :] = 0
        matrix_tampered[-1, :k] = 1  # Força dezenas 1..6 no último sorteio

        monitor2 = LotteryBacktestMonitor(game='megasena')
        monitor2.df = df.copy()
        monitor2.binary_matrix = matrix_tampered
        monitor2.n_concursos = n_concursos

        monitor2.run_walk_forward(n_contests=3, models_to_run=['Bayesiano Global (Beta-Binomial)'], verbose=False)
        preds_2 = monitor2.raw_results_df['dezenas'].tolist()

        # As predições para os concursos ANTERIORES ao último não podem ter mudado!
        assert preds_1[0] == preds_2[0], "Lookahead bias detectado no concurso t-2!"
        assert preds_1[1] == preds_2[1], "Lookahead bias detectado no concurso t-1!"
        # E mesmo a predição para o concurso t NÃO pode ser afetada pelo sorteio real do concurso t
        assert preds_1[2] == preds_2[2], "Lookahead bias detectado na predição do concurso t!"


class TestMLWalkForwardExecution:
    def test_ml_in_walk_forward(self):
        """Valida que o modelo de ML treina e prevê sem erros no walk-forward."""
        np.random.seed(42)
        n_concursos = 115
        n_dezenas = 60
        k = 6

        matrix = np.zeros((n_concursos, n_dezenas), dtype=int)
        for i in range(n_concursos):
            draw = np.random.choice(n_dezenas, size=k, replace=False)
            matrix[i, draw] = 1

        df = pd.DataFrame({'Concurso': np.arange(1, n_concursos + 1)})
        for col_idx in range(k):
            df[f'bola {col_idx + 1}'] = 0
        for i in range(n_concursos):
            drawn = np.where(matrix[i] == 1)[0] + 1
            for col_idx, num in enumerate(drawn):
                df.at[i, f'bola {col_idx + 1}'] = num

        monitor = LotteryBacktestMonitor(game='megasena')
        monitor.df = df
        monitor.binary_matrix = matrix
        monitor.n_concursos = n_concursos

        monitor.run_walk_forward(n_contests=2, models_to_run=['ML Regressão Logística'], fast_ml=True, verbose=False)
        assert len(monitor.raw_results_df) == 2
        assert all(len(d) == 6 for d in monitor.raw_results_df['dezenas'])


class TestEdgeCasesAndRobustness:
    def test_walk_forward_n_equals_1_and_2(self):
        """Verifica que o walk-forward opera perfeitamente com 1 ou 2 concursos sem warnings de singularidade."""
        np.random.seed(99)
        n_concursos = 110
        n_dezenas = 60
        k = 6

        matrix = np.zeros((n_concursos, n_dezenas), dtype=int)
        for i in range(n_concursos):
            draw = np.random.choice(n_dezenas, size=k, replace=False)
            matrix[i, draw] = 1

        df = pd.DataFrame({'Concurso': np.arange(1, n_concursos + 1)})
        for col_idx in range(k):
            df[f'bola {col_idx + 1}'] = 0
        for i in range(n_concursos):
            drawn = np.where(matrix[i] == 1)[0] + 1
            for col_idx, num in enumerate(drawn):
                df.at[i, f'bola {col_idx + 1}'] = num

        monitor = LotteryBacktestMonitor(game='megasena')
        monitor.df = df
        monitor.binary_matrix = matrix
        monitor.n_concursos = n_concursos

        # N = 1
        monitor.run_walk_forward(n_contests=1, models_to_run=['Aposta Aleatória Uniforme'], verbose=False)
        assert len(monitor.tested_contests_) == 1
        fig1 = monitor.plot_dashboard(save_path=None, show=False)
        assert fig1 is not None

        # N = 2
        monitor.run_walk_forward(n_contests=2, models_to_run=['Aposta Aleatória Uniforme'], verbose=False)
        assert len(monitor.tested_contests_) == 2
        hits_sum = monitor.get_hits_summary()
        assert all(hits_sum['IC 95% Inf'] >= 0.0)

    def test_random_forest_in_walk_forward(self):
        """Verifica que o modelo Random Forest pode ser executado no walk-forward."""
        np.random.seed(42)
        n_concursos = 112
        n_dezenas = 60
        k = 6

        matrix = np.zeros((n_concursos, n_dezenas), dtype=int)
        for i in range(n_concursos):
            draw = np.random.choice(n_dezenas, size=k, replace=False)
            matrix[i, draw] = 1

        df = pd.DataFrame({'Concurso': np.arange(1, n_concursos + 1)})
        for col_idx in range(k):
            df[f'bola {col_idx + 1}'] = 0
        for i in range(n_concursos):
            drawn = np.where(matrix[i] == 1)[0] + 1
            for col_idx, num in enumerate(drawn):
                df.at[i, f'bola {col_idx + 1}'] = num

        monitor = LotteryBacktestMonitor(game='megasena')
        monitor.df = df
        monitor.binary_matrix = matrix
        monitor.n_concursos = n_concursos

        monitor.run_walk_forward(n_contests=1, models_to_run=['ML Random Forest'], fast_ml=True, verbose=False)
        assert len(monitor.raw_results_df) == 1
        assert monitor.raw_results_df.iloc[0]['model'] == 'ML Random Forest'
        assert len(monitor.raw_results_df.iloc[0]['dezenas']) == 6

    def test_radar_with_non_default_model_subset(self):
        """Verifica que o radar chart plota polígonos mesmo quando nenhum dos modelos padrão de destaque é executado."""
        np.random.seed(42)
        n_concursos = 110
        n_dezenas = 60
        k = 6

        matrix = np.zeros((n_concursos, n_dezenas), dtype=int)
        for i in range(n_concursos):
            draw = np.random.choice(n_dezenas, size=k, replace=False)
            matrix[i, draw] = 1

        df = pd.DataFrame({'Concurso': np.arange(1, n_concursos + 1)})
        for col_idx in range(k):
            df[f'bola {col_idx + 1}'] = 0
        for i in range(n_concursos):
            drawn = np.where(matrix[i] == 1)[0] + 1
            for col_idx, num in enumerate(drawn):
                df.at[i, f'bola {col_idx + 1}'] = num

        monitor = LotteryBacktestMonitor(game='megasena')
        monitor.df = df
        monitor.binary_matrix = matrix
        monitor.n_concursos = n_concursos

        # Roda apenas Freq/Atraso (Mais Atrasadas), que não estava na lista fixa original de destaques
        monitor.run_walk_forward(n_contests=2, models_to_run=['Freq/Atraso (Mais Atrasadas)'], verbose=False)
        fig = monitor.plot_dashboard(save_path=None, show=False)
        assert fig is not None

    def test_ensemble_models_in_walk_forward(self):
        """Verifica que os modelos Ensemble (Bagging, AdaBoost, Gradient Boosting) rodam perfeitamente no walk-forward."""
        np.random.seed(42)
        n_concursos = 115
        n_dezenas = 60
        k = 6

        matrix = np.zeros((n_concursos, n_dezenas), dtype=int)
        for i in range(n_concursos):
            draw = np.random.choice(n_dezenas, size=k, replace=False)
            matrix[i, draw] = 1

        df = pd.DataFrame({'Concurso': np.arange(1, n_concursos + 1)})
        for col_idx in range(k):
            df[f'bola {col_idx + 1}'] = 0
        for i in range(n_concursos):
            drawn = np.where(matrix[i] == 1)[0] + 1
            for col_idx, num in enumerate(drawn):
                df.at[i, f'bola {col_idx + 1}'] = num

        monitor = LotteryBacktestMonitor(game='megasena')
        monitor.df = df
        monitor.binary_matrix = matrix
        monitor.n_concursos = n_concursos

        ensemble_models = ['Ensemble Bagging', 'Ensemble Pasting', 'Ensemble AdaBoost', 'Ensemble Gradient Boosting', 'Ensemble Stacking']
        monitor.run_walk_forward(n_contests=2, models_to_run=ensemble_models, verbose=False)

        assert len(monitor.raw_results_df) == 2 * len(ensemble_models)
        for m_name in ensemble_models:
            sub = monitor.raw_results_df[monitor.raw_results_df['model'] == m_name]
            assert len(sub) == 2
            assert all(len(d) == 6 for d in sub['dezenas'])
            assert all(len(set(d)) == 6 for d in sub['dezenas'])

    def test_insufficient_history_raises_error(self):
        """Verifica que ultrapassar o histórico com concursos insuficientes de warmup aciona ValueError com mensagem clara."""
        monitor = LotteryBacktestMonitor(game='megasena')
        with pytest.raises(ValueError, match="Histórico insuficiente"):
            monitor.run_walk_forward(n_contests=monitor.n_concursos)

