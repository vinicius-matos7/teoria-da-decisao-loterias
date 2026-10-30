"""
mega_sena_models.py
===================
Módulo com 4 métodos analíticos e preditivos para estimar dezenas das Loterias Caixa (Mega-Sena, Lotofácil, Quina):
1. Método 1: Estatístico (Frequência Ponderada no Tempo + Métrica de Atraso)
2. Método 2: Machine Learning Multi-label (Janelas Móveis Temporais + Probabilidades Calibradas)
3. Método 3: Bayesiano & Teoria da Decisão (Conjugada Beta-Binomial + Maximização de Utilidade Esperada)
4. Método 4: A Geometria do Acaso (Gabaritos Diofantinos de Cores - Renato Gianella, 2013)

Baseado nos conceitos de Inferência Bayesiana e Teoria da Decisão (Esteves, Izbicki e Stern)
e na Combinatória de Gianella (2013).
"""

import os
import sys
import glob
import re
from typing import Dict, List, Optional, Tuple, Union, Any, Generator

# No Windows com Anaconda/Miniconda, garante que Library/bin esteja no PATH e registrado para carregar MKL/BLAS DLLs
if sys.platform == 'win32':
    for prefix_cand in [sys.prefix, os.path.dirname(sys.executable)]:
        lib_bin = os.path.join(prefix_cand, 'Library', 'bin')
        if os.path.exists(lib_bin):
            if lib_bin not in os.environ.get('PATH', ''):
                os.environ['PATH'] = lib_bin + os.pathsep + os.environ.get('PATH', '')
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(lib_bin)
                except Exception:
                    pass

import numpy as np
import pandas as pd
from scipy.stats import beta


# ==============================================================================
# 0. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ==============================================================================

LOTTERY_CONFIGS = {
    'megasena': {
        'name': 'Mega-Sena',
        'prefix': 'mega_sena',
        'n_dezenas': 60,
        'k': 6,
        'ball_cols': [f'bola {i}' for i in range(1, 7)],
        'alpha_0': 1.0,
        'beta_0': 9.0,   # E[theta] = 6/60 = 0.10
        'gianella_groups': 6,
        'gianella_group_size': 10,
    },
    'lotofacil': {
        'name': 'Lotofácil',
        'prefix': 'lotofacil',
        'n_dezenas': 25,
        'k': 15,
        'ball_cols': [f'bola {i}' for i in range(1, 16)],
        'alpha_0': 6.0,
        'beta_0': 4.0,   # E[theta] = 15/25 = 0.60
        'gianella_groups': 5,
        'gianella_group_size': 5,
    },
    'quina': {
        'name': 'Quina',
        'prefix': 'quina',
        'n_dezenas': 80,
        'k': 5,
        'ball_cols': [f'bola {i}' for i in range(1, 6)],
        'alpha_0': 1.0,
        'beta_0': 15.0,  # E[theta] = 5/80 = 0.0625
        'gianella_groups': 8,
        'gianella_group_size': 10,
    }
}


def detect_lottery_game(filepath: str, df: Optional[pd.DataFrame] = None) -> str:
    """Detecta automaticamente o tipo de loteria a partir do nome do arquivo ou das colunas."""
    lower = filepath.lower()
    if 'loto' in lower:
        return 'lotofacil'
    if 'quina' in lower:
        return 'quina'
    if 'mega' in lower:
        return 'megasena'
    if df is not None:
        if 'bola 15' in df.columns:
            return 'lotofacil'
        if 'bola 6' in df.columns:
            return 'megasena'
        if 'bola 5' in df.columns:
            return 'quina'
    return 'megasena'


def find_latest_lottery_file(game: str = 'megasena', search_dir: Optional[str] = None) -> str:
    """
    Encontra o arquivo .xlsx (ou .csv) mais recente para a loteria indicada.
    Procura em pastas comuns ('excel', '.', '..', 'planilhas', etc.)
    e extrai o número do concurso do nome do arquivo (ex: *_ate_concurso_3066_sorteio.xlsx).
    """
    game_clean = game.lower().replace('-', '').replace('_', '')
    if 'mega' in game_clean:
        prefix = 'mega_sena'
    elif 'loto' in game_clean:
        prefix = 'loto_facil'
    elif 'quina' in game_clean:
        prefix = 'quina'
    else:
        prefix = game_clean

    module_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    candidate_dirs = [search_dir] if search_dir else [
        'excel',
        os.path.join(module_dir, 'excel'),
        os.path.join('.', 'excel'),
        os.path.join('..', 'excel'),
        module_dir,
        '.',
        'planilhas',
        os.path.join(module_dir, 'planilhas'),
    ]

    found_files = []
    seen = set()
    for d in candidate_dirs:
        if not d or not os.path.exists(d):
            continue
        abs_d = os.path.abspath(d)
        if abs_d in seen:
            continue
        seen.add(abs_d)
        for ext in ('*.xlsx', '*.csv'):
            for p in glob.glob(os.path.join(d, ext)):
                fname = os.path.basename(p).lower()
                if prefix in fname or game_clean in fname:
                    m = re.search(r'concurso_(\d+)', fname)
                    num = int(m.group(1)) if m else 0
                    priority = 1 if p.endswith('.xlsx') else 0
                    found_files.append((num, priority, p))

    if found_files:
        found_files.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return found_files[0][2]

    return os.path.join('excel', f"{prefix}.xlsx")


def resolve_filepath(filepath: Optional[str] = None, game: str = 'auto') -> str:
    """Procura o arquivo no caminho original ou nas pastas de dados/planilhas/excel."""
    if filepath and os.path.exists(filepath):
        return filepath

    # Se filepath for o nome de um jogo ou None, tenta buscar o arquivo mais recente
    if not filepath or filepath in LOTTERY_CONFIGS:
        game_to_use = filepath if (filepath in LOTTERY_CONFIGS) else (game if game != 'auto' else 'megasena')
        latest = find_latest_lottery_file(game_to_use)
        if os.path.exists(latest):
            return latest

    candidates = [
        filepath,
        os.path.join('excel', filepath),
        os.path.join('planilhas', filepath),
        os.path.join('..', 'excel', filepath),
        os.path.join('..', 'planilhas', filepath),
        os.path.join(os.path.dirname(__file__), 'excel', filepath),
        os.path.join(os.path.dirname(__file__), 'planilhas', filepath),
        os.path.join(os.path.dirname(__file__), filepath),
    ]
    base = os.path.basename(filepath)
    if base != filepath:
        candidates.extend([
            base,
            os.path.join('excel', base),
            os.path.join('planilhas', base),
            os.path.join('..', 'excel', base),
            os.path.join('..', 'planilhas', base),
            os.path.join(os.path.dirname(__file__), 'excel', base),
            os.path.join(os.path.dirname(__file__), 'planilhas', base),
        ])
    for c in candidates:
        if os.path.exists(c):
            return c
        csv_c = c.replace('.xlsx', '.csv')
        if os.path.exists(csv_c):
            return csv_c

    # Tenta descobrir o arquivo mais recente para o jogo detectado
    detected = game if game != 'auto' else detect_lottery_game(filepath)
    latest = find_latest_lottery_file(detected)
    if os.path.exists(latest):
        return latest

    return filepath


def load_lottery_data(filepath: Optional[str] = None, game: str = 'auto') -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Carrega dados de qualquer loteria (Mega-Sena, Lotofácil, Quina) com suporte a .xlsx e .csv.
    """
    if filepath is None or filepath in LOTTERY_CONFIGS:
        game_name = filepath if (filepath in LOTTERY_CONFIGS) else (game if game != 'auto' else 'megasena')
        filepath = find_latest_lottery_file(game_name)
    else:
        filepath = resolve_filepath(filepath, game=game)

    if not os.path.exists(filepath):
        csv_fallback = filepath.replace('.xlsx', '.csv')
        if os.path.exists(csv_fallback):
            filepath = csv_fallback
        else:
            raise FileNotFoundError(f"Arquivo de dados não encontrado: '{filepath}'")

    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        try:
            df = pd.read_excel(filepath, skiprows=6)
            if 'Concurso' not in df.columns:
                df_direct = pd.read_excel(filepath)
                if 'Concurso' in df_direct.columns:
                    df = df_direct
        except (ImportError, ModuleNotFoundError):
            csv_fallback = filepath.replace('.xlsx', '.csv')
            if os.path.exists(csv_fallback):
                df = pd.read_csv(csv_fallback)
            else:
                raise

    if len(df) == 0:
        raise ValueError(f"O arquivo '{filepath}' está vazio ou não contém registros válidos.")

    if game == 'auto':
        game = detect_lottery_game(filepath, df)

    cfg = LOTTERY_CONFIGS.get(game, LOTTERY_CONFIGS['megasena']).copy()
    ball_cols = cfg['ball_cols']
    n_dezenas = cfg['n_dezenas']

    required_cols = ['Concurso', 'Data'] + ball_cols
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória '{col}' não encontrada no arquivo.")

    df = df.sort_values('Concurso').reset_index(drop=True)
    n_concursos = len(df)
    binary_matrix = np.zeros((n_concursos, n_dezenas), dtype=int)
    for t in range(n_concursos):
        for col in ball_cols:
            val = int(df.iloc[t][col])
            if not 1 <= val <= n_dezenas:
                raise ValueError(f"Dezena inválida encontrada no concurso {df.iloc[t]['Concurso']}: {val}")
            binary_matrix[t, val - 1] = 1

    return df, binary_matrix, cfg


def load_mega_sena_data(filepath: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Carrega o arquivo histórico da Mega-Sena (.xlsx), ordena cronologicamente
    por número de concurso e constrói a matriz binária de ocorrências N x 60.
    """
    df, binary_matrix, _ = load_lottery_data(filepath, game='megasena')
    return df, binary_matrix


def compute_delay_statistics(binary_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula para cada dezena (1 a 60):
    - Atraso atual (quantos concursos se passaram desde a última aparição)
    - Média histórica do intervalo entre sorteios (gap médio)
    - Razão de atraso relativo: atraso_atual / media_intervalo

    Parameters
    ----------
    binary_matrix : np.ndarray
        Matriz N x 60 de ocorrências binárias.

    Returns
    -------
    current_delays : np.ndarray (shape 60)
        Atraso atual de cada dezena.
    mean_gaps : np.ndarray (shape 60)
        Média histórica do intervalo de cada dezena.
    delay_ratios : np.ndarray (shape 60)
        Razão entre o atraso atual e a média do intervalo.
    """
    n_concursos, n_dezenas = binary_matrix.shape
    if n_concursos == 0:
        return np.zeros(n_dezenas, dtype=int), np.full(n_dezenas, 10.0), np.zeros(n_dezenas, dtype=float)

    current_delays = np.zeros(n_dezenas, dtype=int)
    mean_gaps = np.zeros(n_dezenas, dtype=float)

    for i in range(n_dezenas):
        occurrences = np.where(binary_matrix[:, i] == 1)[0]
        if len(occurrences) == 0:
            current_delays[i] = n_concursos
            # Gap esperado teórico sob amostragem uniforme (6/60 = 0.10 -> gap esperado 10)
            mean_gaps[i] = float(max(n_concursos, 10))
        else:
            current_delays[i] = (n_concursos - 1) - occurrences[-1]
            if len(occurrences) > 1:
                gaps = np.diff(occurrences)
                mean_gaps[i] = float(np.mean(gaps))
            else:
                mean_gaps[i] = float(max(n_concursos, 10))

    # Evita divisão por zero
    safe_mean_gaps = np.where(mean_gaps > 0, mean_gaps, 10.0)
    delay_ratios = current_delays / safe_mean_gaps

    return current_delays, mean_gaps, delay_ratios


# ==============================================================================
# 1. MÉTODO 1: ESTATÍSTICO (FREQUÊNCIA PONDERADA NO TEMPO + MÉTRICA DE ATRASO)
# ==============================================================================

class FrequencyDelayModel:
    """
    Método 1: Abordagem Estatística Baseada em Frequência Ponderada e Atraso.

    Combina:
    1. Frequência ponderada no tempo com decaimento exponencial:
       w_t = 2^(-(T - t) / half_life)
       Atribui mais peso a sorteios recentes sem descartar a história antiga.
    2. Métrica de atraso relativo:
       R_i = atraso_i / media_intervalo_i
       Quantifica a pressão de retorno/compensação empírica.
    3. Escore Composto Padronizado:
       Score_i = alpha * Z(f_w, i) + (1 - alpha) * Z(R_i)
    """

    def __init__(self, half_life: float = 100.0, alpha: float = 0.5):
        """
        Parameters
        ----------
        half_life : float
            Meia-vida temporal em número de concursos (default 100).
        alpha : float
            Peso dado à frequência ponderada (alpha) vs atraso relativo (1 - alpha).
            Default 0.5 (equilíbrio entre momentum e reversão à média).
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha deve estar no intervalo [0, 1]. Recebido: {alpha}")
        if half_life <= 0:
            raise ValueError(f"half_life deve ser estritamente positivo (> 0). Recebido: {half_life}")

        self.half_life = half_life
        self.alpha = alpha
        self.scores_df_: Optional[pd.DataFrame] = None
        self.top_6_: Optional[List[int]] = None

    def fit(self, binary_matrix: np.ndarray) -> "FrequencyDelayModel":
        """
        Calcula as estatísticas e os escores para as 60 dezenas.
        """
        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        n_concursos, n_dezenas = binary_matrix.shape

        # 1. Frequência ponderada no tempo
        decay = 0.5 ** (1.0 / self.half_life)
        time_weights = decay ** np.arange(n_concursos - 1, -1, -1)
        sum_weights = np.sum(time_weights)
        weighted_freq = np.sum(binary_matrix * time_weights[:, None], axis=0) / sum_weights

        # 2. Estatísticas de atraso
        current_delays, mean_gaps, delay_ratios = compute_delay_statistics(binary_matrix)

        # 3. Frequência global simples
        global_freq = binary_matrix.mean(axis=0)

        # 4. Padronização (Z-score)
        std_freq = np.std(weighted_freq)
        z_freq = (weighted_freq - np.mean(weighted_freq)) / (std_freq if std_freq > 0 else 1.0)

        std_delay = np.std(delay_ratios)
        z_delay = (delay_ratios - np.mean(delay_ratios)) / (std_delay if std_delay > 0 else 1.0)

        # Escore Composto Balanceado
        composite_score = self.alpha * z_freq + (1.0 - self.alpha) * z_delay

        dezenas = np.arange(1, n_dezenas + 1)
        self.scores_df_ = pd.DataFrame({
            'dezena': dezenas,
            'atraso_atual': current_delays,
            'intervalo_medio': np.round(mean_gaps, 2),
            'razao_atraso': np.round(delay_ratios, 3),
            'freq_ponderada': np.round(weighted_freq, 5),
            'freq_global': np.round(global_freq, 5),
            'z_freq': np.round(z_freq, 4),
            'z_atraso': np.round(z_delay, 4),
            'escore_composto': np.round(composite_score, 4)
        }).sort_values('escore_composto', ascending=False).reset_index(drop=True)

        self.top_6_ = sorted(self.scores_df_['dezena'].head(6).tolist())
        return self

    def predict_top_k(self, k: int = 6, strategy: str = 'balanced') -> List[int]:
        """
        Retorna as top K dezenas selecionadas pela estratégia.

        Parameters
        ----------
        k : int
            Número de dezenas a selecionar (default 6).
        strategy : str
            'balanced': maximiza o escore composto (frequência ponderada + atraso).
            'overdue': seleciona as dezenas mais atrasadas (maior atraso relativo).
            'momentum': seleciona as dezenas de maior frequência ponderada recente.
        """
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser treinado com fit() antes de prever.")

        if k <= 0:
            return []
        k = min(k, len(self.scores_df_))

        if strategy == 'balanced':
            return sorted(self.scores_df_.sort_values('escore_composto', ascending=False)['dezena'].head(k).tolist())
        elif strategy == 'overdue':
            return sorted(self.scores_df_.sort_values('razao_atraso', ascending=False)['dezena'].head(k).tolist())
        elif strategy == 'momentum':
            return sorted(self.scores_df_.sort_values('freq_ponderada', ascending=False)['dezena'].head(k).tolist())
        else:
            raise ValueError(f"Estratégia desconhecida: '{strategy}'. Use 'balanced', 'overdue' ou 'momentum'.")

    def get_scores_df(self) -> pd.DataFrame:
        """Retorna o DataFrame detalhado com todas as métricas."""
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser treinado com fit() primeiro.")
        return self.scores_df_.copy()


# ==============================================================================
# 2. MÉTODO 2: MACHINE LEARNING MULTI-LABEL
# ==============================================================================

class MultiLabelMLModel:
    """
    Método 2: Aprendizado Supervisionado Multi-label com Janelas Temporais
    e Probabilidades Calibradas.

    Formulação:
    - Para cada concurso t, a informação observável provém estritamente de <= t-1.
    - Features de engenharia temporal por dezena:
      * Atraso no instante t-1
      * Frequência em janelas móveis passadas: W = [5, 10, 25, 50, 100]
      * Indicadores de lag imediato (lag 1, lag 2, lag 3)
      * Estatísticas globais do sorteio anterior (média dos sorteados, número de pares)
    - Abordagem de Relevância Binária (Binary Relevance):
      Treina um estimador supervisionado com calibração de probabilidades para cada
      uma das 60 dezenas.
    - Predição para o concurso N+1:
      Gera as probabilidades calibradas P(Y_{N+1, i} = 1 | X_{N+1}) para i in 1..60
      e seleciona as dezenas com maiores probabilidades.
    """

    def __init__(self,
                 estimator: str = 'logistic_regression',
                 windows: Optional[List[int]] = None,
                 lags: int = 3,
                 min_train_history: int = 100,
                 scale_features: bool = False,
                 random_state: int = 42):
        """
        Parameters
        ----------
        estimator : str
            'logistic_regression' (default, bem calibrada por log-loss) ou 'random_forest'.
        windows : list of int
            Tamanhos das janelas móveis (default [5, 10, 25, 50, 100]).
        lags : int
            Número de sorteios anteriores imediatos como flags binárias (default 3).
        min_train_history : int
            Número mínimo de concursos no histórico antes de iniciar o treino (default 100).
        scale_features : bool
            Se True, padroniza as features (StandardScaler) antes da regressão logística L2,
            evitando que variáveis com escalas numéricas distintas sofram penalização desbalanceada.
            Default False (para compatibilidade estrita com a calibração padrão).
        random_state : int
            Semente para reprodutibilidade.
        """
        if min_train_history < 1:
            raise ValueError(f"min_train_history deve ser >= 1. Recebido: {min_train_history}")
        if lags < 0:
            raise ValueError(f"lags deve ser >= 0. Recebido: {lags}")

        self.estimator_name = estimator
        self.windows = sorted(windows) if windows is not None else [5, 10, 25, 50, 100]
        self.lags = lags
        self.min_train_history = min_train_history
        self.scale_features = scale_features
        self.random_state = random_state
        
        self.models_: Dict[int, object] = {}
        self.probabilities_df_: Optional[pd.DataFrame] = None
        self.top_6_: Optional[List[int]] = None

    def _extract_features(self, binary_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extrai de forma vetorizada as matrizes de features para todos os concursos.
        Retorna:
          X_all: array (N, 60, n_features)
          delays: array (N, 60)
          cur_delays_end: array (60,) atrasos atuais após o último concurso
        """
        n_concursos, n_dezenas = binary_matrix.shape

        # 1. Matriz de atrasos temporais pré-sorteio t (vetorizada)
        delays = np.zeros((n_concursos, n_dezenas), dtype=float)
        cur = np.zeros(n_dezenas, dtype=float)
        for t in range(n_concursos):
            delays[t] = cur
            cur = np.where(binary_matrix[t] == 1, 0.0, cur + 1.0)
        cur_delays_end = cur

        # 2. Médias móveis multiescala em janelas passadas (vetorizadas via cumsum)
        cumsum = np.vstack([np.zeros((1, n_dezenas)), np.cumsum(binary_matrix, axis=0)])
        t_idx = np.arange(n_concursos)
        ma_features = []
        for w in self.windows:
            start = np.maximum(0, t_idx - w)
            w_actual = np.maximum(1, t_idx - start)[:, None]
            ma = (cumsum[t_idx] - cumsum[start]) / w_actual
            ma_features.append(ma)

        # 3. Defasagens binárias (lags 1 a L)
        lag_features = []
        for lag in range(1, self.lags + 1):
            if lag < n_concursos:
                lag_arr = np.vstack([np.zeros((lag, n_dezenas)), binary_matrix[:-lag]])
            else:
                lag_arr = np.zeros((n_concursos, n_dezenas))
            lag_features.append(lag_arr)

        # 4. Contexto macro do concurso t-1 (proporção de pares)
        even_counts = np.sum(binary_matrix[:, 1::2], axis=1) / 6.0
        macro_lag1 = np.concatenate([[0.0], even_counts[:-1]])
        macro_feat = np.repeat(macro_lag1[:, None], n_dezenas, axis=1)

        # Empilha todas as features no formato (N, n_dezenas, n_features)
        feature_list = [delays] + ma_features + lag_features + [macro_feat]
        X_all = np.stack(feature_list, axis=-1)

        return X_all, delays, cur_delays_end

    def fit(self, binary_matrix: np.ndarray) -> "MultiLabelMLModel":
        """
        Treina os 60 classificadores binários e avalia probabilidades para o concurso N+1.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        n_concursos, n_dezenas = binary_matrix.shape
        if n_concursos <= self.min_train_history:
            raise ValueError(f"O número de concursos ({n_concursos}) deve ser estritamente maior que min_train_history ({self.min_train_history}).")

        X_all, delays, cur_delays_end = self._extract_features(binary_matrix)
        
        train_start = self.min_train_history
        raw_probs = np.zeros(n_dezenas, dtype=float)

        # Vetor de features para o concurso N+1
        cumsum = np.vstack([np.zeros((1, n_dezenas)), np.cumsum(binary_matrix, axis=0)])
        even_count_last = np.sum(binary_matrix[-1, 1::2]) / 6.0

        for i in range(n_dezenas):
            X_i = X_all[train_start:n_concursos, i, :]
            y_i = binary_matrix[train_start:n_concursos, i]

            # Verificação de segurança contra targets com classe única
            unique_classes = np.unique(y_i)
            if len(unique_classes) < 2:
                raw_probs[i] = 1.0 - 1e-4 if unique_classes[0] == 1 else 1e-4
                self.models_[i] = None
                continue

            if self.estimator_name == 'logistic_regression':
                if self.scale_features:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.pipeline import Pipeline
                    clf = Pipeline([
                        ('scaler', StandardScaler()),
                        ('clf', LogisticRegression(
                            C=0.1,
                            max_iter=1000,
                            solver='liblinear',
                            random_state=self.random_state
                        ))
                    ])
                else:
                    clf = LogisticRegression(
                        C=0.1,
                        max_iter=1000,
                        solver='liblinear',
                        random_state=self.random_state
                    )
            elif self.estimator_name == 'random_forest':
                clf = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=4,
                    min_samples_leaf=20,
                    random_state=self.random_state,
                    n_jobs=1
                )
            else:
                raise ValueError(f"Estimador desconhecido: '{self.estimator_name}'")

            clf.fit(X_i, y_i)
            self.models_[i] = clf

            # Monta feature vector para o concurso N+1
            x_next = []
            x_next.append(cur_delays_end[i])
            for w in self.windows:
                start = max(0, n_concursos - w)
                w_actual = max(1, n_concursos - start)
                x_next.append((cumsum[n_concursos, i] - cumsum[start, i]) / w_actual)
            for lag in range(1, self.lags + 1):
                x_next.append(binary_matrix[n_concursos - lag, i] if (n_concursos - lag) >= 0 else 0.0)
            x_next.append(even_count_last)

            x_next_arr = np.array(x_next).reshape(1, -1)
            p = clf.predict_proba(x_next_arr)[0, 1]
            raw_probs[i] = p

        # Normalização das probabilidades para que a soma seja 6 (pois exatamente 6 números são sorteados)
        norm_factor = 6.0 / (np.sum(raw_probs) if np.sum(raw_probs) > 0 else 1.0)
        calibrated_draw_probs = raw_probs * norm_factor

        dezenas = np.arange(1, n_dezenas + 1)
        self.probabilities_df_ = pd.DataFrame({
            'dezena': dezenas,
            'probabilidade_bruta': np.round(raw_probs, 5),
            'probabilidade_calibrada_sorteio': np.round(calibrated_draw_probs, 5),
            'atraso_atual': cur_delays_end.astype(int),
            'freq_ultimos_10': np.round((cumsum[n_concursos, :] - cumsum[max(0, n_concursos - 10), :]) / 10.0, 3),
            'freq_ultimos_50': np.round((cumsum[n_concursos, :] - cumsum[max(0, n_concursos - 50), :]) / 50.0, 3)
        }).sort_values('probabilidade_calibrada_sorteio', ascending=False).reset_index(drop=True)

        self.top_6_ = sorted(self.probabilities_df_['dezena'].head(6).tolist())
        return self

    def predict_top_k(self, k: int = 6) -> List[int]:
        """Retorna as top K dezenas com maiores probabilidades estimadas."""
        if self.probabilities_df_ is None:
            raise RuntimeError("O modelo precisa ser treinado com fit() antes de prever.")
        if k <= 0:
            return []
        k = min(k, len(self.probabilities_df_))
        return sorted(self.probabilities_df_.head(k)['dezena'].tolist())

    def get_probabilities_df(self) -> pd.DataFrame:
        """Retorna o DataFrame detalhado com as probabilidades."""
        if self.probabilities_df_ is None:
            raise RuntimeError("O modelo precisa ser treinado com fit() primeiro.")
        return self.probabilities_df_.copy()


# ==============================================================================
# 3. MÉTODO 3: BAYESIANO & TEORIA DA DECISÃO
# ==============================================================================

class BayesianDecisionModel:
    """
    Método 3: Modelagem de Probabilidades a Posteriori (Beta-Binomial)
    combinada com Função de Utilidade sob a perspectiva da Teoria da Decisão.

    Fundamentação Teórica (Esteves, Izbicki e Stern - Capítulos 4 e 6):
    1. Modelo Estatístico Conjugado Beta-Binomial:
       Para cada dezena i:
       - Prior conjugada: theta_i ~ Beta(alpha_0, beta_0)
         Default: alpha_0 = 1.0, beta_0 = 9.0 (E[theta] = 6/60 = 0.10, peso a priori = 10 concursos).
       - Verossimilhança: X_i | theta_i ~ Binomial(n, theta_i)
       - Posteriori (Lema 4.31):
         theta_i | X_i = x_i ~ Beta(alpha_0 + x_i, beta_0 + n - x_i)
       - Esperança a Posteriori (Lema 4.32):
         p_i = E[theta_i | x_i] = (alpha_0 + x_i) / (alpha_0 + beta_0 + n)

    2. Teoria da Decisão (Definição 6.1 e Seção 6.2 do Livro):
       - Conjunto de Alternativas A: todas as apostas a = {d_1, ..., d_6} de 6 dezenas distintas.
       - Espaço de Estados da Natureza Theta: o sorteio real s = {s_1, ..., s_6}.
       - Medida de Probabilidade P sobre Theta: derivada da distribuição preditiva a posteriori p_i.
       - Função de Utilidade U(a, s):
         * Linear em acertos: U(a, s) = |a cap s| = sum_{i in a} I(i in s)
           Utilidade Esperada a Posteriori:
           E[U(a, s) | dados] = sum_{i in a} p_i
           Regra de Decisão Ótima de Bayes:
           a* = argmax_{a in A} sum_{i in a} p_i (seleção das 6 maiores esperanças a posteriori).
         * Utilidade com Aversão a Risco / Penalidade de Correlação:
           U_div(a, s) = sum_{i in a} p_i - lambda_cov * sum_{i, j in a, i != j} Cov(i, j)
    """

    def __init__(self,
                 alpha_0: float = 1.0,
                 beta_0: float = 9.0,
                 decay_half_life: Optional[float] = None,
                 risk_penalty: float = 0.0):
        """
        Parameters
        ----------
        alpha_0 : float
            Hiperparâmetro alpha da prior Beta (default 1.0).
        beta_0 : float
            Hiperparâmetro beta da prior Beta (default 9.0).
            Note que alpha_0 / (alpha_0 + beta_0) = 0.10 = 6/60.
        decay_half_life : Optional[float]
            Se especificado (ex.: 200.0), aplica decaimento exponencial de pseudo-contagens
            para criar um modelo Bayesiano Dinâmico que atualiza crenças dando peso
            progressivo a sorteios mais recentes.
        risk_penalty : float
            Penalidade de co-ocorrência / correlação histórica na função de utilidade (lambda_cov).
        """
        if alpha_0 <= 0 or beta_0 <= 0:
            raise ValueError(f"alpha_0 e beta_0 devem ser estritamente positivos (> 0). Recebidos: alpha_0={alpha_0}, beta_0={beta_0}")
        if decay_half_life is not None and decay_half_life <= 0:
            raise ValueError(f"decay_half_life deve ser estritamente positivo (> 0) quando especificado. Recebido: {decay_half_life}")
        if risk_penalty < 0:
            raise ValueError(f"risk_penalty deve ser não-negativo (>= 0). Recebido: {risk_penalty}")

        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.decay_half_life = decay_half_life
        self.risk_penalty = risk_penalty
        
        self.posterior_df_: Optional[pd.DataFrame] = None
        self.top_6_: Optional[List[int]] = None
        self.cov_matrix_: Optional[np.ndarray] = None

    def fit(self, binary_matrix: np.ndarray) -> "BayesianDecisionModel":
        """
        Calcula os parâmetros da distribuição a posteriori e os intervalos de credibilidade.
        """
        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        n_concursos, n_dezenas = binary_matrix.shape

        if self.decay_half_life is not None and self.decay_half_life > 0:
            # Modelo Bayesiano Dinâmico com Evidência Ponderada
            gamma = 0.5 ** (1.0 / self.decay_half_life)
            weights = gamma ** np.arange(n_concursos - 1, -1, -1)
            eff_n = float(np.sum(weights))
            eff_x = np.sum(binary_matrix * weights[:, None], axis=0)
            n_obs = eff_n
            x_obs = eff_x
        else:
            # Modelo Bayesiano Clássico Global
            n_obs = float(n_concursos)
            x_obs = binary_matrix.sum(axis=0).astype(float)

        post_alpha = self.alpha_0 + x_obs
        post_beta = self.beta_0 + n_obs - x_obs
        post_mean = post_alpha / (post_alpha + post_beta)
        post_var = (post_alpha * post_beta) / ((post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1))
        post_std = np.sqrt(post_var)

        # Intervalos de Credibilidade Centrais de 95%
        ci_lower = beta.ppf(0.025, post_alpha, post_beta)
        ci_upper = beta.ppf(0.975, post_alpha, post_beta)

        # Matriz de covariância empírica entre as dezenas
        self.cov_matrix_ = np.cov(binary_matrix, rowvar=False)

        dezenas = np.arange(1, n_dezenas + 1)
        self.posterior_df_ = pd.DataFrame({
            'dezena': dezenas,
            'sucessos_observados': np.round(x_obs, 2),
            'n_concursos_efetivos': np.round(n_obs, 1),
            'post_alpha': np.round(post_alpha, 3),
            'post_beta': np.round(post_beta, 3),
            'esperanca_posteriori': np.round(post_mean, 5),
            'desvio_posteriori': np.round(post_std, 5),
            'ci_95_inf': np.round(ci_lower, 5),
            'ci_95_sup': np.round(ci_upper, 5)
        }).sort_values('esperanca_posteriori', ascending=False).reset_index(drop=True)

        # Decisão ótima de Bayes sob a função de utilidade especificada
        if self.risk_penalty == 0.0 or self.cov_matrix_ is None:
            # Utilidade Linear pura: top 6 diretos
            self.top_6_ = sorted(self.posterior_df_['dezena'].head(6).tolist())
        else:
            # Utilidade com aversão a risco (busca gulosa ou seleção de Pareto)
            self.top_6_ = self._optimize_risk_utility(post_mean, k=6)

        return self

    def _optimize_risk_utility(self, post_mean: np.ndarray, k: int = 6) -> List[int]:
        """
        Otimização gananciosa da utilidade U(a) = sum p_i - lambda sum Cov(i, j).
        """
        if k <= 0:
            return []
        k = min(k, len(post_mean))

        selected: List[int] = []
        candidates = list(range(len(post_mean)))

        # Começa com a dezena de maior probabilidade a posteriori
        best_first = int(np.argmax(post_mean))
        selected.append(best_first)
        candidates.remove(best_first)

        while len(selected) < k and candidates:
            best_gain = -np.inf
            best_cand = None
            for c in candidates:
                marginal_p = post_mean[c]
                cov_penalty = self.risk_penalty * sum(self.cov_matrix_[c, s] for s in selected)
                net_gain = marginal_p - cov_penalty
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_cand = c
            if best_cand is not None:
                selected.append(best_cand)
                candidates.remove(best_cand)
            else:
                break

        return sorted([s + 1 for s in selected])

    def predict_top_k(self, k: int = 6) -> List[int]:
        """Retorna a aposta ótima de Bayes de tamanho K."""
        if self.posterior_df_ is None:
            raise RuntimeError("O modelo precisa ser treinado com fit() antes de prever.")
        if k <= 0:
            return []
        k = min(k, len(self.posterior_df_))
        if self.risk_penalty == 0.0:
            return sorted(self.posterior_df_.head(k)['dezena'].tolist())
        else:
            post_mean = self.posterior_df_.sort_values('dezena')['esperanca_posteriori'].values
            return self._optimize_risk_utility(post_mean, k=k)

    def get_posterior_df(self) -> pd.DataFrame:
        """Retorna o DataFrame detalhado com a posteriori."""
        if self.posterior_df_ is None:
            raise RuntimeError("O modelo precisa ser treinado com fit() primeiro.")
        return self.posterior_df_.copy()

    def evaluate_expected_utility(self, action: List[int]) -> float:
        """
        Calcula a utilidade esperada a posteriori para um bilhete a de 6 dezenas.
        """
        if self.posterior_df_ is None:
            raise RuntimeError("O modelo precisa ser treinado com fit() primeiro.")
        
        post_dict = dict(zip(self.posterior_df_['dezena'], self.posterior_df_['esperanca_posteriori']))
        linear_util = sum(post_dict[d] for d in action)
        
        if self.risk_penalty > 0.0 and self.cov_matrix_ is not None:
            cov_sum = 0.0
            for i in range(len(action)):
                for j in range(i + 1, len(action)):
                    d_i = action[i] - 1
                    d_j = action[j] - 1
                    cov_sum += self.cov_matrix_[d_i, d_j]
            return float(linear_util - self.risk_penalty * cov_sum)
        
        return float(linear_util)



# ==============================================================================
# 3.5. MÉTODO 4: A GEOMETRIA DO ACASO (GABARITOS DIOFANTINOS DE CORES)
# Renato Gianella (2013), "The Geometry of Chance: Lotto Numbers Follow a Predicted Pattern"
# ==============================================================================

class GianellaColorTemplateModel:
    """
    Método 4: A Geometria do Acaso (Gianella Color Template Model)
    Baseado em Renato Gianella (2013), 'The Geometry of Chance: Lotto Numbers
    Follow a Predicted Pattern', Revista Brasileira de Biometria, 31(4), 582-597.
    
    Conceitos formais e matemáticos:
    - Partição do universo N de dezenas em C grupos regulares de cores D_0, ..., D_{C-1}.
    - Equação Diofantina Linear: sum_{i=0}^{C-1} x_i = k, com 0 <= x_i <= group_size.
    - Gabaritos de Cores (Templates): partições inteiras ordenadas decrescentes (x_{(1)}, ..., x_{(C)}).
    - Probabilidade Combinatória Teórica: P(T) = Comb(T) / C(N, k), calculada exatamente
      pelo produto hipergeométrico multivariado e permutações de cores com repetição.
    - Princípio da Invariância do Micro-Estado vs Dispersão do Macro-Estado:
      cada bilhete individual possui probabilidade invariante 1 / C(N, k), mas os gabaritos
      agregam quantidades drasticamente distintas de combinações simples.
    - Lei dos Grandes Números: as frequências empíricas históricas convergem para as probabilidades
      teóricas dos gabaritos.
    - Decisão Ótima: aposta que atende ao gabarito de maior probabilidade combinatória
      (maior entropia combinatória), preenchendo as cores mais ativas com as melhores dezenas.
    """
    def __init__(self, n_dezenas: int = 60, k: int = 6, group_size: int = 10, random_state: int = 42):
        if n_dezenas <= 0 or k <= 0 or group_size <= 0:
            raise ValueError(f"n_dezenas ({n_dezenas}), k ({k}) e group_size ({group_size}) devem ser > 0.")
        if k > n_dezenas:
            raise ValueError(f"k ({k}) não pode ser maior que n_dezenas ({n_dezenas}).")
        if n_dezenas % group_size != 0:
            raise ValueError(f"n_dezenas ({n_dezenas}) deve ser divisível por group_size ({group_size}) para formar grupos regulares de cores.")
        self.n_dezenas = n_dezenas
        self.k = k
        self.group_size = group_size
        self.n_colors = n_dezenas // group_size
        self.random_state = random_state
        self.template_df_: Optional[pd.DataFrame] = None
        self.color_scores_: Optional[np.ndarray] = None
        self.dezena_scores_: Optional[np.ndarray] = None
        self.top_template_: Optional[Tuple[int, ...]] = None

    def _get_partitions(self, n: int, k_parts: int, max_val: int) -> List[Tuple[int, ...]]:
        def _helper(rem: int, parts_left: int, cur_max: int):
            if parts_left == 0:
                if rem == 0:
                    yield ()
                return
            for val in range(min(cur_max, rem, max_val), -1, -1):
                for p in _helper(rem - val, parts_left - 1, val):
                    yield (val,) + p
        return list(_helper(n, k_parts, max_val))

    def fit(self, binary_matrix: np.ndarray) -> 'GianellaColorTemplateModel':
        import math
        from collections import Counter

        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] < 1:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")
        if binary_matrix.shape[1] != self.n_dezenas:
            raise ValueError(f"binary_matrix deve ter {self.n_dezenas} colunas, encontrado {binary_matrix.shape[1]}.")

        n_draws = binary_matrix.shape[0]
        total_comb = math.comb(self.n_dezenas, self.k)
        partitions = self._get_partitions(self.k, self.n_colors, self.group_size)

        theo_data = {}
        for part in partitions:
            counts = Counter(part)
            perm_count = math.factorial(self.n_colors)
            for _, cnt in counts.items():
                perm_count //= math.factorial(cnt)
            ways = 1
            for val in part:
                ways *= math.comb(self.group_size, val)
            comb = perm_count * ways
            prob = comb / total_comb
            part_str = '-'.join(str(x) for x in part)
            theo_data[part_str] = {
                'part_tuple': part,
                'combinacoes': comb,
                'prob_teorica': prob,
                'empirico_count': 0
            }

        dezena_sums = binary_matrix.sum(axis=0)
        self.dezena_scores_ = dezena_sums / n_draws

        color_draws_matrix = np.zeros((n_draws, self.n_colors), dtype=int)
        for i in range(self.n_colors):
            start_col = i * self.group_size
            end_col = (i + 1) * self.group_size
            color_draws_matrix[:, i] = binary_matrix[:, start_col:end_col].sum(axis=1)

        self.color_scores_ = color_draws_matrix.mean(axis=0)

        for d in range(n_draws):
            counts = sorted(color_draws_matrix[d, :], reverse=True)
            p_str = '-'.join(str(x) for x in counts)
            if p_str in theo_data:
                theo_data[p_str]['empirico_count'] += 1

        records = []
        for p_str, d in theo_data.items():
            emp_freq = d['empirico_count'] / n_draws
            records.append({
                'template': p_str,
                'part_tuple': d['part_tuple'],
                'combinacoes': d['combinacoes'],
                'prob_teorica': d['prob_teorica'],
                'freq_teorica_pct': d['prob_teorica'] * 100.0,
                'ocorrencias': d['empirico_count'],
                'freq_empirica_pct': emp_freq * 100.0,
                'dif_pct': (emp_freq - d['prob_teorica']) * 100.0
            })

        df_res = pd.DataFrame(records).sort_values('prob_teorica', ascending=False).reset_index(drop=True)
        self.template_df_ = df_res
        self.top_template_ = df_res.iloc[0]['part_tuple']
        return self

    def predict_top_k(self, k: Optional[int] = None, template: Optional[Tuple[int, ...]] = None) -> List[int]:
        if self.template_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")
        if k is None:
            k = self.k
        if k <= 0:
            return []
        k = min(k, self.n_dezenas)
        if template is None:
            template = self.top_template_

        color_order = np.argsort(self.color_scores_)[::-1]
        selected_dezenas: List[int] = []

        for idx, color_idx in enumerate(color_order):
            n_to_pick = template[idx] if idx < len(template) else 0
            if n_to_pick > 0:
                start_d = color_idx * self.group_size
                end_d = (color_idx + 1) * self.group_size
                group_dezenas = np.arange(start_d, end_d)
                group_scores = self.dezena_scores_[group_dezenas]
                top_in_group = group_dezenas[np.argsort(group_scores)[::-1][:n_to_pick]]
                selected_dezenas.extend(int(x) + 1 for x in top_in_group)

        # Se k > len(selected_dezenas), completa com as melhores dezenas restantes
        if len(selected_dezenas) < k:
            selected_set = set(selected_dezenas)
            sorted_all = np.argsort(self.dezena_scores_)[::-1]
            for d_idx in sorted_all:
                d_num = int(d_idx) + 1
                if d_num not in selected_set:
                    selected_dezenas.append(d_num)
                    selected_set.add(d_num)
                    if len(selected_dezenas) >= k:
                        break

        return sorted(selected_dezenas[:k])

    def get_template_stats_df(self) -> pd.DataFrame:
        if self.template_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")
        return self.template_df_.copy()


# ==============================================================================
# 3.6. MÉTODOS HÍBRIDOS: GIANELLA & TEORIA DA DECISÃO BAYESIANA
# Renato Gianella (2013) + Esteves, Izbicki e Stern (2020)
# ==============================================================================

def _min_max_scale(arr: np.ndarray) -> np.ndarray:
    """Normaliza um vetor para [0, 1]. Se amplitude for zero, retorna array de zeros."""
    arr = np.asarray(arr, dtype=float)
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn > 1e-12:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr)


def _select_top_k_greedy(scores: np.ndarray,
                         k: int,
                         cov_matrix: Optional[np.ndarray] = None,
                         risk_penalty: float = 0.0) -> List[int]:
    """Seleciona as top K dezenas com penalização de covariância e aversão ao risco."""
    n = len(scores)
    k = max(0, min(k, n))
    if k == 0:
        return []
    if risk_penalty <= 0.0 or cov_matrix is None:
        return sorted(list(np.argsort(scores)[::-1][:k] + 1))

    candidates = list(range(n))
    selected: List[int] = []
    while len(selected) < k and candidates:
        best_gain = -np.inf
        best_c = None
        for cand in candidates:
            cov_pen = risk_penalty * sum(cov_matrix[cand, s] for s in selected)
            gain = scores[cand] - cov_pen
            if gain > best_gain:
                best_gain = gain
                best_c = cand
        if best_c is not None:
            selected.append(best_c)
            candidates.remove(best_c)
        else:
            break
    return sorted([s + 1 for s in selected])


class GianellaBayesHybridModel:
    """
    Método Híbrido: A Geometria do Acaso (Gianella, 2013) combinada com
    Inferência Bayesiana e Teoria da Decisão (Esteves, Izbicki e Stern),
    Regressão Logística Calibrada e Modelagem Temporal de Frequência/Atraso.

    Fundamentação Teórica:
    1. Macro-Estrutura (Combinatória de Gianella):
       - Partição do universo N de dezenas em C grupos regulares de cores.
       - Resolução da Equação Diofantina linear: sum_{c=0}^{C-1} x_c = k, com 0 <= x_c <= group_size.
       - Gabarito Diofantino Ótimo T* = (x_{(1)}, ..., x_{(C)}) que maximiza a probabilidade
         hipergeométrica combinatória exata P(T).
       - Ordenação dos grupos de cores segundo a intensidade probabilística/atividade.

    2. Micro-Decisão (Inferência Bayesiana e Teoria da Decisão):
       Ao invés de selecionar as dezenas dentro de cada grupo de cor por simples contagem
       frequentista ingênua, as cotas de cada cor são preenchidas pelas regras ótimas de Bayes:

       a) Gianella com Bayes Global ('bayes_global'):
          - Baseado na distribuição preditiva a posteriori da conjugada Beta-Binomial:
            theta_i ~ Beta(alpha_0 + x_i, beta_0 + n - x_i)
            p_i = E[theta_i | dados globais] = (alpha_0 + x_i) / (alpha_0 + beta_0 + n)
          - Ordena os grupos de cores pela soma das probabilidades a posteriori dos seus membros:
            M_c = sum_{i in Grupo c} p_i
          - Dentro de cada cor c, seleciona as x_{(c)} dezenas com maiores esperanças a posteriori.

       b) Gianella com Bayes Adaptativo ('bayes_adaptive'):
          - Modelo Bayesiano Dinâmico com Evidência Ponderada Temporalmente (meia-vida exponencial):
            p_i^{adapt} = E[theta_i | dados dinamicos] = (alpha_0 + x_i^{eff}) / (alpha_0 + beta_0 + n^{eff})
          - Captura acelerações e transições temporais de probabilidade recente, alocando
            as cotas de Gianella nas dezenas com maior momentum a posteriori.

       c) Gianella com a Melhor Decisão de Bayes ('bayes_decision' / 'bayes_optimal'):
          - Otimização sob a Função de Utilidade de Bayes restrita ao macro-estado diofantino de Gianella:
            max_{a in T*} E[U(a, s) | dados]
          - Combina a evidência bayesiana global e adaptativa (ou escore de consenso do conjunto bayesiano)
            com ajuste de aversão ao risco / covariância empírica (lambda_cov), garantindo que a aposta
            escolhida seja estritamente ótima em utilidade esperada sob o gabarito combinatório mais provável.
    """

    def __init__(self,
                 n_dezenas: int = 60,
                 k: int = 6,
                 group_size: int = 10,
                 alpha_0: float = 1.0,
                 beta_0: float = 9.0,
                 decay_half_life: float = 150.0,
                 risk_penalty: float = 0.0,
                 random_state: int = 42):
        if n_dezenas <= 0 or k <= 0 or group_size <= 0:
            raise ValueError(f"n_dezenas ({n_dezenas}), k ({k}) e group_size ({group_size}) devem ser > 0.")
        if k > n_dezenas:
            raise ValueError(f"k ({k}) não pode ser maior que n_dezenas ({n_dezenas}).")
        if n_dezenas % group_size != 0:
            raise ValueError(f"n_dezenas ({n_dezenas}) deve ser divisível por group_size ({group_size}) para formar grupos regulares de cores.")
        if alpha_0 <= 0 or beta_0 <= 0:
            raise ValueError(f"alpha_0 ({alpha_0}) e beta_0 ({beta_0}) devem ser > 0.")
        if decay_half_life <= 0:
            raise ValueError(f"decay_half_life ({decay_half_life}) deve ser > 0.")
        if risk_penalty < 0:
            raise ValueError(f"risk_penalty ({risk_penalty}) deve ser >= 0.")

        self.n_dezenas = n_dezenas
        self.k = k
        self.group_size = group_size
        self.n_colors = n_dezenas // group_size
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.decay_half_life = decay_half_life
        self.risk_penalty = risk_penalty
        self.random_state = random_state

        self.gianella_model_ = GianellaColorTemplateModel(
            n_dezenas=n_dezenas, k=k, group_size=group_size, random_state=random_state
        )
        self.cov_matrix_: Optional[np.ndarray] = None
        self.post_probs_global_: Optional[np.ndarray] = None
        self.post_probs_adaptive_: Optional[np.ndarray] = None
        self.bayes_decision_scores_: Optional[np.ndarray] = None
        self.ml_scores_: Optional[np.ndarray] = None
        self.momentum_scores_: Optional[np.ndarray] = None
        self.balanced_scores_: Optional[np.ndarray] = None
        self.top_template_: Optional[Tuple[int, ...]] = None
        self.summary_df_: Optional[pd.DataFrame] = None

    def fit(self,
            binary_matrix: np.ndarray,
            consensus_scores: Optional[np.ndarray] = None,
            ml_scores: Optional[np.ndarray] = None,
            momentum_scores: Optional[np.ndarray] = None,
            balanced_scores: Optional[np.ndarray] = None) -> 'GianellaBayesHybridModel':
        """
        Ajusta os gabaritos de Gianella e os modelos Bayesianos (Global, Adaptativo e Decisão Ótima)
        e integra os escores auxiliares (Regressão Logística, Momentum e Balanceado).
        """
        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] < 1:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")
        if binary_matrix.shape[1] != self.n_dezenas:
            raise ValueError(f"binary_matrix deve ter {self.n_dezenas} colunas, encontrado {binary_matrix.shape[1]}.")

        n_concursos = binary_matrix.shape[0]

        # 1. Ajusta o modelo combinatório de Gianella
        self.gianella_model_.fit(binary_matrix)
        self.top_template_ = self.gianella_model_.top_template_

        # 2. Ajusta Bayesiano Global (Beta-Binomial)
        n_obs = float(n_concursos)
        x_obs = binary_matrix.sum(axis=0).astype(float)
        post_alpha_global = self.alpha_0 + x_obs
        post_beta_global = self.beta_0 + n_obs - x_obs
        self.post_probs_global_ = post_alpha_global / (post_alpha_global + post_beta_global)

        # 3. Ajusta Bayesiano Adaptativo (Decaimento Temporal Exponencial)
        gamma = 0.5 ** (1.0 / self.decay_half_life)
        weights = gamma ** np.arange(n_concursos - 1, -1, -1)
        eff_n = float(np.sum(weights))
        eff_x = np.sum(binary_matrix * weights[:, None], axis=0)
        post_alpha_dyn = self.alpha_0 + eff_x
        post_beta_dyn = self.beta_0 + eff_n - eff_x
        self.post_probs_adaptive_ = post_alpha_dyn / (post_alpha_dyn + post_beta_dyn)

        # 4. Matriz de Covariância Empírica
        if n_concursos > 1:
            self.cov_matrix_ = np.cov(binary_matrix, rowvar=False)
        else:
            self.cov_matrix_ = np.zeros((self.n_dezenas, self.n_dezenas))

        # 5. Escore da Melhor Decisão de Bayes
        if consensus_scores is not None and len(consensus_scores) == self.n_dezenas:
            c_arr = np.array(consensus_scores, dtype=float)
            max_c = np.max(c_arr)
            c_norm = c_arr / max_c if max_c > 0 else c_arr
            # Combinação convexa: 50% distribuição a posteriori conjunta + 50% consenso multi-modelo
            p_joint = 0.5 * (self.post_probs_global_ + self.post_probs_adaptive_)
            self.bayes_decision_scores_ = 0.5 * p_joint + 0.5 * c_norm
        else:
            # Equilíbrio bayesiano entre evidência estática global e dinâmica adaptativa
            self.bayes_decision_scores_ = 0.5 * self.post_probs_global_ + 0.5 * self.post_probs_adaptive_

        # 6. Escores Auxiliares: Momentum e Balanceado
        m1_needed = (momentum_scores is None or len(momentum_scores) != self.n_dezenas or
                     balanced_scores is None or len(balanced_scores) != self.n_dezenas)
        if m1_needed:
            m1_model = FrequencyDelayModel(half_life=100.0, alpha=0.5)
            m1_model.fit(binary_matrix)

        if momentum_scores is not None and len(momentum_scores) == self.n_dezenas:
            self.momentum_scores_ = _min_max_scale(momentum_scores)
        else:
            mom = np.zeros(self.n_dezenas, dtype=float)
            for _, r in m1_model.scores_df_.iterrows():
                mom[int(r['dezena']) - 1] = float(r['freq_ponderada'])
            self.momentum_scores_ = _min_max_scale(mom)

        if balanced_scores is not None and len(balanced_scores) == self.n_dezenas:
            self.balanced_scores_ = _min_max_scale(balanced_scores)
        else:
            bal = np.zeros(self.n_dezenas, dtype=float)
            for _, r in m1_model.scores_df_.iterrows():
                bal[int(r['dezena']) - 1] = float(r['escore_composto'])
            self.balanced_scores_ = _min_max_scale(bal)

        # 7. Escores Auxiliares: Regressão Logística
        if ml_scores is not None and len(ml_scores) == self.n_dezenas:
            self.ml_scores_ = _min_max_scale(ml_scores)
        else:
            if n_concursos > 30:
                try:
                    min_hist = min(100, max(20, n_concursos // 2))
                    m2_model = MultiLabelMLModel(estimator='logistic_regression', min_train_history=min_hist, random_state=42)
                    m2_model.fit(binary_matrix)
                    df_probs = m2_model.get_probabilities_df()
                    ml_arr = np.zeros(self.n_dezenas, dtype=float)
                    for _, r in df_probs.iterrows():
                        ml_arr[int(r['dezena']) - 1] = float(r['probabilidade_calibrada_sorteio'])
                    self.ml_scores_ = _min_max_scale(ml_arr)
                except Exception:
                    self.ml_scores_ = self.momentum_scores_.copy()
            else:
                self.ml_scores_ = self.momentum_scores_.copy()

        # Constrói DataFrame comparativo
        dezenas = np.arange(1, self.n_dezenas + 1)
        cores = [int(i // self.group_size) for i in range(self.n_dezenas)]
        self.summary_df_ = pd.DataFrame({
            'dezena': dezenas,
            'grupo_cor': cores,
            'prob_bayes_global': np.round(self.post_probs_global_, 5),
            'prob_bayes_adaptativo': np.round(self.post_probs_adaptive_, 5),
            'escore_melhor_decisao': np.round(self.bayes_decision_scores_, 5),
            'escore_ml_logistica': np.round(self.ml_scores_, 5),
            'escore_momentum': np.round(self.momentum_scores_, 5),
            'escore_balanceado': np.round(self.balanced_scores_, 5),
        })

        return self

    def _select_under_template(self,
                               dezena_scores: np.ndarray,
                               k: int,
                               template: Optional[Tuple[int, ...]] = None) -> List[int]:
        if template is None:
            template = self.top_template_
        if k <= 0:
            return []
        k = min(k, self.n_dezenas)

        # 1. Massa agregada de cada grupo de cor (soma das probabilidades ou escores dos elementos)
        color_mass = np.zeros(self.n_colors, dtype=float)
        for c in range(self.n_colors):
            start = c * self.group_size
            end = (c + 1) * self.group_size
            color_mass[c] = float(np.sum(dezena_scores[start:end]))

        # 2. Ordena os grupos de cor da maior para a menor intensidade de probabilidade
        color_order = np.argsort(color_mass)[::-1]

        selected_dezenas: List[int] = []
        selected_set = set()

        # 3. Preenche as cotas do gabarito diofantino ótimo
        for idx, color_idx in enumerate(color_order):
            n_to_pick = template[idx] if idx < len(template) else 0
            if n_to_pick > 0:
                start = color_idx * self.group_size
                end = (color_idx + 1) * self.group_size
                group_dezenas = np.arange(start, end)

                if self.risk_penalty > 0.0 and self.cov_matrix_ is not None:
                    # Seleção gananciosa com penalização de covariância
                    group_cand = list(group_dezenas)
                    picks: List[int] = []
                    while len(picks) < n_to_pick and group_cand:
                        best_gain = -np.inf
                        best_c = None
                        for cand in group_cand:
                            prior_selected = [s - 1 for s in selected_dezenas] + picks
                            cov_pen = self.risk_penalty * sum(
                                self.cov_matrix_[cand, s] for s in prior_selected
                            )
                            gain = dezena_scores[cand] - cov_pen
                            if gain > best_gain:
                                best_gain = gain
                                best_c = cand
                        if best_c is not None:
                            picks.append(best_c)
                            group_cand.remove(best_c)
                        else:
                            break
                    for p in picks:
                        num = int(p) + 1
                        selected_dezenas.append(num)
                        selected_set.add(num)
                else:
                    group_scores = dezena_scores[group_dezenas]
                    top_in_group = group_dezenas[np.argsort(group_scores)[::-1][:n_to_pick]]
                    for d_idx in top_in_group:
                        num = int(d_idx) + 1
                        selected_dezenas.append(num)
                        selected_set.add(num)

        # 4. Caso o template selecione menos que k (ou k > soma do template), preenche com os melhores remanescentes
        if len(selected_dezenas) < k:
            sorted_all = np.argsort(dezena_scores)[::-1]
            for d_idx in sorted_all:
                d_num = int(d_idx) + 1
                if d_num not in selected_set:
                    selected_dezenas.append(d_num)
                    selected_set.add(d_num)
                    if len(selected_dezenas) >= k:
                        break

        return sorted(selected_dezenas[:k])

    def predict_top_k(self,
                      k: Optional[int] = None,
                      strategy: str = 'bayes_decision',
                      template: Optional[Tuple[int, ...]] = None) -> List[int]:
        """
        Retorna as K dezenas selecionadas pela união de Gianella com a estratégia indicada.

        Parameters
        ----------
        k : Optional[int]
            Quantidade de dezenas a selecionar (default self.k).
        strategy : str
            - 'bayes_decision' / 'bayes_optimal': Gianella com a Melhor Decisão de Bayes.
            - 'bayes_global': Gianella com Bayes Global.
            - 'bayes_adaptive': Gianella com Bayes Adaptativo.
            - 'logistic_regression' / 'ml': Gianella com Regressão Logística.
            - 'momentum': Gianella com Frequência/Atraso (Momentum).
            - 'balanced': Gianella com Frequência/Atraso (Balanceado).
        template : Optional[Tuple[int, ...]]
            Gabarito de cores diofantino a seguir (default: top_template_).
        """
        if self.post_probs_global_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")
        if k is None:
            k = self.k

        strat = strategy.lower().strip()
        if strat in ['bayes_global', 'global']:
            scores = self.post_probs_global_
        elif strat in ['bayes_adaptive', 'adaptive', 'adaptativo']:
            scores = self.post_probs_adaptive_
        elif strat in ['bayes_decision', 'bayes_optimal', 'optimal', 'melhor_decisao', 'decisao']:
            scores = self.bayes_decision_scores_
        elif strat in ['logistic_regression', 'regressao_logistica', 'ml', 'gianella_logistic_regression']:
            scores = self.ml_scores_
        elif strat in ['momentum', 'momento', 'gianella_momentum']:
            scores = self.momentum_scores_
        elif strat in ['balanced', 'balanceado', 'gianella_balanced']:
            scores = self.balanced_scores_
        else:
            raise ValueError(
                f"Estratégia desconhecida: '{strategy}'. "
                f"Opções válidas: 'bayes_decision', 'bayes_global', 'bayes_adaptive', "
                f"'logistic_regression', 'momentum', 'balanced'."
            )

        return self._select_under_template(scores, k=k, template=template)

    def predict_gianella_bayes_global(self, k: Optional[int] = None) -> List[int]:
        """Predição unindo Gianella com Bayes Global."""
        return self.predict_top_k(k=k, strategy='bayes_global')

    def predict_gianella_bayes_adaptive(self, k: Optional[int] = None) -> List[int]:
        """Predição unindo Gianella com Bayes Adaptativo."""
        return self.predict_top_k(k=k, strategy='bayes_adaptive')

    def predict_gianella_best_bayes_decision(self, k: Optional[int] = None) -> List[int]:
        """Predição unindo Gianella com a Melhor Decisão de Bayes."""
        return self.predict_top_k(k=k, strategy='bayes_decision')

    def predict_gianella_logistic_regression(self, k: Optional[int] = None) -> List[int]:
        """Predição unindo Gianella com Regressão Logística Calibrada."""
        return self.predict_top_k(k=k, strategy='logistic_regression')

    def predict_gianella_momentum(self, k: Optional[int] = None) -> List[int]:
        """Predição unindo Gianella com Frequência/Atraso (Momentum)."""
        return self.predict_top_k(k=k, strategy='momentum')

    def predict_gianella_balanced(self, k: Optional[int] = None) -> List[int]:
        """Predição unindo Gianella com Frequência/Atraso (Balanceado)."""
        return self.predict_top_k(k=k, strategy='balanced')

    def _combine_and_select(self, base_scores: np.ndarray, aux_scores: np.ndarray, k: Optional[int] = None) -> List[int]:
        if k is None:
            k = self.k
        comb = 0.5 * _min_max_scale(base_scores) + 0.5 * _min_max_scale(aux_scores)
        return _select_top_k_greedy(comb, k=k, cov_matrix=self.cov_matrix_, risk_penalty=self.risk_penalty)

    # 1. Melhor Decisão de Bayes + Auxiliares
    def predict_best_bayes_logistic_regression(self, k: Optional[int] = None) -> List[int]:
        """Melhor Decisão de Bayes + Regressão Logística."""
        return self._combine_and_select(self.bayes_decision_scores_, self.ml_scores_, k=k)

    def predict_best_bayes_momentum(self, k: Optional[int] = None) -> List[int]:
        """Melhor Decisão de Bayes + Frequência/Atraso (Momentum)."""
        return self._combine_and_select(self.bayes_decision_scores_, self.momentum_scores_, k=k)

    def predict_best_bayes_balanced(self, k: Optional[int] = None) -> List[int]:
        """Melhor Decisão de Bayes + Frequência/Atraso (Balanceado)."""
        return self._combine_and_select(self.bayes_decision_scores_, self.balanced_scores_, k=k)

    # 2. Bayes Global + Auxiliares
    def predict_bayes_global_logistic_regression(self, k: Optional[int] = None) -> List[int]:
        """Bayes Global + Regressão Logística."""
        return self._combine_and_select(self.post_probs_global_, self.ml_scores_, k=k)

    def predict_bayes_global_momentum(self, k: Optional[int] = None) -> List[int]:
        """Bayes Global + Frequência/Atraso (Momentum)."""
        return self._combine_and_select(self.post_probs_global_, self.momentum_scores_, k=k)

    def predict_bayes_global_balanced(self, k: Optional[int] = None) -> List[int]:
        """Bayes Global + Frequência/Atraso (Balanceado)."""
        return self._combine_and_select(self.post_probs_global_, self.balanced_scores_, k=k)

    # 3. Bayes Adaptativo + Auxiliares
    def predict_bayes_adaptive_logistic_regression(self, k: Optional[int] = None) -> List[int]:
        """Bayes Adaptativo + Regressão Logística."""
        return self._combine_and_select(self.post_probs_adaptive_, self.ml_scores_, k=k)

    def predict_bayes_adaptive_momentum(self, k: Optional[int] = None) -> List[int]:
        """Bayes Adaptativo + Frequência/Atraso (Momentum)."""
        return self._combine_and_select(self.post_probs_adaptive_, self.momentum_scores_, k=k)

    def predict_bayes_adaptive_balanced(self, k: Optional[int] = None) -> List[int]:
        """Bayes Adaptativo + Frequência/Atraso (Balanceado)."""
        return self._combine_and_select(self.post_probs_adaptive_, self.balanced_scores_, k=k)

    def get_summary_df(self) -> pd.DataFrame:
        """Retorna o DataFrame consolidado com escores e probabilidades."""
        if self.summary_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")
        return self.summary_df_.copy()


def predict_gianella_bayes_global(binary_matrix: np.ndarray,
                                  cfg: Dict,
                                  k: Optional[int] = None) -> List[int]:
    """Predição direta unindo o Gabarito de Gianella com Bayes Global."""
    model = GianellaBayesHybridModel(
        n_dezenas=cfg['n_dezenas'],
        k=cfg['k'] if k is None else k,
        group_size=cfg.get('gianella_group_size', 10),
        alpha_0=cfg.get('alpha_0', 1.0),
        beta_0=cfg.get('beta_0', 1.0)
    )
    model.fit(binary_matrix)
    return model.predict_gianella_bayes_global(k)


def predict_gianella_bayes_adaptive(binary_matrix: np.ndarray,
                                    cfg: Dict,
                                    k: Optional[int] = None,
                                    decay_half_life: float = 150.0) -> List[int]:
    """Predição direta unindo o Gabarito de Gianella com Bayes Adaptativo."""
    model = GianellaBayesHybridModel(
        n_dezenas=cfg['n_dezenas'],
        k=cfg['k'] if k is None else k,
        group_size=cfg.get('gianella_group_size', 10),
        alpha_0=cfg.get('alpha_0', 1.0),
        beta_0=cfg.get('beta_0', 1.0),
        decay_half_life=decay_half_life
    )
    model.fit(binary_matrix)
    return model.predict_gianella_bayes_adaptive(k)


def compute_ensemble_consensus_scores(binary_matrix: np.ndarray,
                                      cfg: Dict,
                                      k: Optional[int] = None) -> np.ndarray:
    """
    Calcula os escores ponderados do consenso multi-modelo (Decisão Ótima de Bayes)
    combinando M1 (Balanceado, Atraso, Momentum), M2 (ML), M3 (Bayes Global e Adaptativo) e M4 (Gianella).
    """
    n_dezenas = cfg['n_dezenas']
    k_eval = cfg['k'] if k is None else k
    alpha_0 = cfg.get('alpha_0', 1.0)
    beta_0 = cfg.get('beta_0', 1.0)
    group_size = cfg.get('gianella_group_size', 10)

    # M1
    m1 = FrequencyDelayModel(half_life=100.0, alpha=0.5)
    m1.fit(binary_matrix)
    pred_m1 = m1.predict_top_k(k_eval, strategy='balanced')

    # M2
    m2 = MultiLabelMLModel(estimator='logistic_regression', random_state=42)
    m2.fit(binary_matrix)
    pred_m2 = m2.predict_top_k(k_eval)

    # M3
    m3_static = BayesianDecisionModel(alpha_0=alpha_0, beta_0=beta_0)
    m3_static.fit(binary_matrix)
    pred_m3_static = m3_static.predict_top_k(k_eval)

    m3_dyn = BayesianDecisionModel(alpha_0=alpha_0, beta_0=beta_0, decay_half_life=150.0)
    m3_dyn.fit(binary_matrix)
    pred_m3_dyn = m3_dyn.predict_top_k(k_eval)

    # M4
    m4 = GianellaColorTemplateModel(n_dezenas=n_dezenas, k=k_eval, group_size=group_size)
    m4.fit(binary_matrix)
    pred_m4 = m4.predict_top_k(k_eval)

    vote_weights = {
        'm1': (pred_m1, 1.0),
        'm2': (pred_m2, 2.0),
        'm3_static': (pred_m3_static, 2.5),
        'm3_dyn': (pred_m3_dyn, 2.5),
        'm4': (pred_m4, 1.5)
    }
    ball_scores = np.zeros(n_dezenas, dtype=float)
    for _, (preds, w) in vote_weights.items():
        for ball in preds:
            ball_scores[ball - 1] += w
    return ball_scores


def predict_gianella_best_bayes_decision(binary_matrix: np.ndarray,
                                         cfg: Dict,
                                         k: Optional[int] = None,
                                         decay_half_life: float = 150.0,
                                         consensus_scores: Optional[np.ndarray] = None) -> List[int]:
    """Predição direta unindo o Gabarito de Gianella com a Melhor Decisão de Bayes."""
    if consensus_scores is None:
        consensus_scores = compute_ensemble_consensus_scores(binary_matrix, cfg, k=k)

    model = GianellaBayesHybridModel(
        n_dezenas=cfg['n_dezenas'],
        k=cfg['k'] if k is None else k,
        group_size=cfg.get('gianella_group_size', 10),
        alpha_0=cfg.get('alpha_0', 1.0),
        beta_0=cfg.get('beta_0', 1.0),
        decay_half_life=decay_half_life
    )
    model.fit(binary_matrix, consensus_scores=consensus_scores)
    return model.predict_gianella_best_bayes_decision(k)


def predict_gianella_logistic_regression(binary_matrix: np.ndarray,
                                        cfg: Dict,
                                        k: Optional[int] = None,
                                        ml_scores: Optional[np.ndarray] = None) -> List[int]:
    """Predição direta unindo o Gabarito de Gianella com Regressão Logística Calibrada."""
    model = GianellaBayesHybridModel(
        n_dezenas=cfg['n_dezenas'],
        k=cfg['k'] if k is None else k,
        group_size=cfg.get('gianella_group_size', 10),
        alpha_0=cfg.get('alpha_0', 1.0),
        beta_0=cfg.get('beta_0', 1.0)
    )
    model.fit(binary_matrix, ml_scores=ml_scores)
    return model.predict_gianella_logistic_regression(k)


def predict_gianella_momentum(binary_matrix: np.ndarray,
                              cfg: Dict,
                              k: Optional[int] = None,
                              momentum_scores: Optional[np.ndarray] = None) -> List[int]:
    """Predição direta unindo o Gabarito de Gianella com Frequência/Atraso (Momentum)."""
    model = GianellaBayesHybridModel(
        n_dezenas=cfg['n_dezenas'],
        k=cfg['k'] if k is None else k,
        group_size=cfg.get('gianella_group_size', 10),
        alpha_0=cfg.get('alpha_0', 1.0),
        beta_0=cfg.get('beta_0', 1.0)
    )
    model.fit(binary_matrix, momentum_scores=momentum_scores)
    return model.predict_gianella_momentum(k)


def predict_gianella_balanced(binary_matrix: np.ndarray,
                              cfg: Dict,
                              k: Optional[int] = None,
                              balanced_scores: Optional[np.ndarray] = None) -> List[int]:
    """Predição direta unindo o Gabarito de Gianella com Frequência/Atraso (Balanceado)."""
    model = GianellaBayesHybridModel(
        n_dezenas=cfg['n_dezenas'],
        k=cfg['k'] if k is None else k,
        group_size=cfg.get('gianella_group_size', 10),
        alpha_0=cfg.get('alpha_0', 1.0),
        beta_0=cfg.get('beta_0', 1.0)
    )
    model.fit(binary_matrix, balanced_scores=balanced_scores)
    return model.predict_gianella_balanced(k)


class BayesianHybridModel:
    """
    Modelagem Preditiva Híbrida Bayesiana & Teoria da Decisão sob Incerteza.
    Combina distribuições a posteriori e utilidades bayesianas com modelos auxiliares:
    - Regressão Logística (Probabilidades Calibradas Multi-label)
    - Frequência Ponderada Temporal (Momentum Recente)
    - Escore Padronizado Frequência + Atraso (Balanceado)
    """

    def __init__(self,
                 n_dezenas: int = 60,
                 k: int = 6,
                 alpha_0: float = 1.0,
                 beta_0: float = 9.0,
                 decay_half_life: float = 150.0,
                 risk_penalty: float = 0.0,
                 random_state: int = 42):
        self.n_dezenas = n_dezenas
        self.k = k
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.decay_half_life = decay_half_life
        self.risk_penalty = risk_penalty
        self.random_state = random_state

        self.post_probs_global_: Optional[np.ndarray] = None
        self.post_probs_adaptive_: Optional[np.ndarray] = None
        self.bayes_decision_scores_: Optional[np.ndarray] = None
        self.cov_matrix_: Optional[np.ndarray] = None
        self.ml_scores_: Optional[np.ndarray] = None
        self.momentum_scores_: Optional[np.ndarray] = None
        self.balanced_scores_: Optional[np.ndarray] = None

    def fit(self,
            binary_matrix: np.ndarray,
            consensus_scores: Optional[np.ndarray] = None,
            ml_scores: Optional[np.ndarray] = None,
            momentum_scores: Optional[np.ndarray] = None,
            balanced_scores: Optional[np.ndarray] = None) -> 'BayesianHybridModel':
        n_concursos = binary_matrix.shape[0]
        n_obs = float(n_concursos)
        x_obs = binary_matrix.sum(axis=0).astype(float)
        post_alpha_global = self.alpha_0 + x_obs
        post_beta_global = self.beta_0 + n_obs - x_obs
        self.post_probs_global_ = post_alpha_global / (post_alpha_global + post_beta_global)

        gamma = 0.5 ** (1.0 / self.decay_half_life)
        weights = gamma ** np.arange(n_concursos - 1, -1, -1)
        eff_n = float(np.sum(weights))
        eff_x = np.sum(binary_matrix * weights[:, None], axis=0)
        post_alpha_dyn = self.alpha_0 + eff_x
        post_beta_dyn = self.beta_0 + eff_n - eff_x
        self.post_probs_adaptive_ = post_alpha_dyn / (post_alpha_dyn + post_beta_dyn)

        if n_concursos > 1:
            self.cov_matrix_ = np.cov(binary_matrix, rowvar=False)
        else:
            self.cov_matrix_ = np.zeros((self.n_dezenas, self.n_dezenas))

        if consensus_scores is not None and len(consensus_scores) == self.n_dezenas:
            c_arr = np.array(consensus_scores, dtype=float)
            max_c = np.max(c_arr)
            c_norm = c_arr / max_c if max_c > 0 else c_arr
            p_joint = 0.5 * (self.post_probs_global_ + self.post_probs_adaptive_)
            self.bayes_decision_scores_ = 0.5 * p_joint + 0.5 * c_norm
        else:
            self.bayes_decision_scores_ = 0.5 * self.post_probs_global_ + 0.5 * self.post_probs_adaptive_

        m1_needed = (momentum_scores is None or len(momentum_scores) != self.n_dezenas or
                     balanced_scores is None or len(balanced_scores) != self.n_dezenas)
        if m1_needed:
            m1_model = FrequencyDelayModel(half_life=100.0, alpha=0.5)
            m1_model.fit(binary_matrix)

        if momentum_scores is not None and len(momentum_scores) == self.n_dezenas:
            self.momentum_scores_ = _min_max_scale(momentum_scores)
        else:
            mom = np.zeros(self.n_dezenas, dtype=float)
            for _, r in m1_model.scores_df_.iterrows():
                mom[int(r['dezena']) - 1] = float(r['freq_ponderada'])
            self.momentum_scores_ = _min_max_scale(mom)

        if balanced_scores is not None and len(balanced_scores) == self.n_dezenas:
            self.balanced_scores_ = _min_max_scale(balanced_scores)
        else:
            bal = np.zeros(self.n_dezenas, dtype=float)
            for _, r in m1_model.scores_df_.iterrows():
                bal[int(r['dezena']) - 1] = float(r['escore_composto'])
            self.balanced_scores_ = _min_max_scale(bal)

        if ml_scores is not None and len(ml_scores) == self.n_dezenas:
            self.ml_scores_ = _min_max_scale(ml_scores)
        else:
            if n_concursos > 30:
                try:
                    min_hist = min(100, max(20, n_concursos // 2))
                    m2_model = MultiLabelMLModel(estimator='logistic_regression', min_train_history=min_hist, random_state=42)
                    m2_model.fit(binary_matrix)
                    df_probs = m2_model.get_probabilities_df()
                    ml_arr = np.zeros(self.n_dezenas, dtype=float)
                    for _, r in df_probs.iterrows():
                        ml_arr[int(r['dezena']) - 1] = float(r['probabilidade_calibrada_sorteio'])
                    self.ml_scores_ = _min_max_scale(ml_arr)
                except Exception:
                    self.ml_scores_ = self.momentum_scores_.copy()
            else:
                self.ml_scores_ = self.momentum_scores_.copy()

        return self

    def _combine_and_select(self, base_scores: np.ndarray, aux_scores: np.ndarray, k: Optional[int] = None) -> List[int]:
        if k is None:
            k = self.k
        comb = 0.5 * _min_max_scale(base_scores) + 0.5 * _min_max_scale(aux_scores)
        return _select_top_k_greedy(comb, k=k, cov_matrix=self.cov_matrix_, risk_penalty=self.risk_penalty)

    def predict_best_bayes_logistic_regression(self, k: Optional[int] = None) -> List[int]:
        return self._combine_and_select(self.bayes_decision_scores_, self.ml_scores_, k=k)

    def predict_best_bayes_momentum(self, k: Optional[int] = None) -> List[int]:
        return self._combine_and_select(self.bayes_decision_scores_, self.momentum_scores_, k=k)

    def predict_best_bayes_balanced(self, k: Optional[int] = None) -> List[int]:
        return self._combine_and_select(self.bayes_decision_scores_, self.balanced_scores_, k=k)

    def predict_bayes_global_logistic_regression(self, k: Optional[int] = None) -> List[int]:
        return self._combine_and_select(self.post_probs_global_, self.ml_scores_, k=k)

    def predict_bayes_global_momentum(self, k: Optional[int] = None) -> List[int]:
        return self._combine_and_select(self.post_probs_global_, self.momentum_scores_, k=k)

    def predict_bayes_global_balanced(self, k: Optional[int] = None) -> List[int]:
        return self._combine_and_select(self.post_probs_global_, self.balanced_scores_, k=k)

    def predict_bayes_adaptive_logistic_regression(self, k: Optional[int] = None) -> List[int]:
        return self._combine_and_select(self.post_probs_adaptive_, self.ml_scores_, k=k)

    def predict_bayes_adaptive_momentum(self, k: Optional[int] = None) -> List[int]:
        return self._combine_and_select(self.post_probs_adaptive_, self.momentum_scores_, k=k)

    def predict_bayes_adaptive_balanced(self, k: Optional[int] = None) -> List[int]:
        return self._combine_and_select(self.post_probs_adaptive_, self.balanced_scores_, k=k)


def predict_best_bayes_logistic_regression(binary_matrix: np.ndarray, cfg: Dict, k: Optional[int] = None) -> List[int]:
    """Melhor Decisão de Bayes + Regressão Logística."""
    model = BayesianHybridModel(n_dezenas=cfg['n_dezenas'], k=cfg['k'] if k is None else k,
                                alpha_0=cfg.get('alpha_0', 1.0), beta_0=cfg.get('beta_0', 1.0))
    model.fit(binary_matrix)
    return model.predict_best_bayes_logistic_regression(k)


def predict_best_bayes_momentum(binary_matrix: np.ndarray, cfg: Dict, k: Optional[int] = None) -> List[int]:
    """Melhor Decisão de Bayes + Frequência/Atraso (Momentum)."""
    model = BayesianHybridModel(n_dezenas=cfg['n_dezenas'], k=cfg['k'] if k is None else k,
                                alpha_0=cfg.get('alpha_0', 1.0), beta_0=cfg.get('beta_0', 1.0))
    model.fit(binary_matrix)
    return model.predict_best_bayes_momentum(k)


def predict_best_bayes_balanced(binary_matrix: np.ndarray, cfg: Dict, k: Optional[int] = None) -> List[int]:
    """Melhor Decisão de Bayes + Frequência/Atraso (Balanceado)."""
    model = BayesianHybridModel(n_dezenas=cfg['n_dezenas'], k=cfg['k'] if k is None else k,
                                alpha_0=cfg.get('alpha_0', 1.0), beta_0=cfg.get('beta_0', 1.0))
    model.fit(binary_matrix)
    return model.predict_best_bayes_balanced(k)


def predict_bayes_global_logistic_regression(binary_matrix: np.ndarray, cfg: Dict, k: Optional[int] = None) -> List[int]:
    """Bayes Global + Regressão Logística."""
    model = BayesianHybridModel(n_dezenas=cfg['n_dezenas'], k=cfg['k'] if k is None else k,
                                alpha_0=cfg.get('alpha_0', 1.0), beta_0=cfg.get('beta_0', 1.0))
    model.fit(binary_matrix)
    return model.predict_bayes_global_logistic_regression(k)


def predict_bayes_global_momentum(binary_matrix: np.ndarray, cfg: Dict, k: Optional[int] = None) -> List[int]:
    """Bayes Global + Frequência/Atraso (Momentum)."""
    model = BayesianHybridModel(n_dezenas=cfg['n_dezenas'], k=cfg['k'] if k is None else k,
                                alpha_0=cfg.get('alpha_0', 1.0), beta_0=cfg.get('beta_0', 1.0))
    model.fit(binary_matrix)
    return model.predict_bayes_global_momentum(k)


def predict_bayes_global_balanced(binary_matrix: np.ndarray, cfg: Dict, k: Optional[int] = None) -> List[int]:
    """Bayes Global + Frequência/Atraso (Balanceado)."""
    model = BayesianHybridModel(n_dezenas=cfg['n_dezenas'], k=cfg['k'] if k is None else k,
                                alpha_0=cfg.get('alpha_0', 1.0), beta_0=cfg.get('beta_0', 1.0))
    model.fit(binary_matrix)
    return model.predict_bayes_global_balanced(k)


def predict_bayes_adaptive_logistic_regression(binary_matrix: np.ndarray, cfg: Dict, k: Optional[int] = None) -> List[int]:
    """Bayes Adaptativo + Regressão Logística."""
    model = BayesianHybridModel(n_dezenas=cfg['n_dezenas'], k=cfg['k'] if k is None else k,
                                alpha_0=cfg.get('alpha_0', 1.0), beta_0=cfg.get('beta_0', 1.0))
    model.fit(binary_matrix)
    return model.predict_bayes_adaptive_logistic_regression(k)


def predict_bayes_adaptive_momentum(binary_matrix: np.ndarray, cfg: Dict, k: Optional[int] = None) -> List[int]:
    """Bayes Adaptativo + Frequência/Atraso (Momentum)."""
    model = BayesianHybridModel(n_dezenas=cfg['n_dezenas'], k=cfg['k'] if k is None else k,
                                alpha_0=cfg.get('alpha_0', 1.0), beta_0=cfg.get('beta_0', 1.0))
    model.fit(binary_matrix)
    return model.predict_bayes_adaptive_momentum(k)


def predict_bayes_adaptive_balanced(binary_matrix: np.ndarray, cfg: Dict, k: Optional[int] = None) -> List[int]:
    """Bayes Adaptativo + Frequência/Atraso (Balanceado)."""
    model = BayesianHybridModel(n_dezenas=cfg['n_dezenas'], k=cfg['k'] if k is None else k,
                                alpha_0=cfg.get('alpha_0', 1.0), beta_0=cfg.get('beta_0', 1.0))
    model.fit(binary_matrix)
    return model.predict_bayes_adaptive_balanced(k)


# ==============================================================================
# 3.7. MÉTODOS ENSEMBLE (GÉRON, CAPÍTULO 7: ENSEMBLE LEARNING AND RANDOM FORESTS)
# Bagging (Bootstrap Aggregating), AdaBoost (Adaptive Boosting) e Gradient Boosting (Residual Boosting)
# ==============================================================================

def _extract_base_model_scores(model_name: str,
                               binary_matrix: np.ndarray,
                               cfg: Dict,
                               seed: int = 42) -> np.ndarray:
    """
    Função utilitária interna para ajustar um dos métodos analíticos existentes
    ('esses métodos que estão aí') sobre uma matriz de dados e retornar o vetor
    de escores normalizados no intervalo [0, 1] para todas as dezenas.
    """
    n_concursos, n_dezenas = binary_matrix.shape
    scores = np.zeros(n_dezenas, dtype=float)

    if model_name == 'm1_balanced':
        m = FrequencyDelayModel(half_life=100.0, alpha=0.5).fit(binary_matrix)
        for _, r in m.scores_df_.iterrows():
            scores[int(r['dezena']) - 1] = float(r['escore_composto'])
    elif model_name == 'm1_momentum':
        m = FrequencyDelayModel(half_life=100.0, alpha=0.5).fit(binary_matrix)
        for _, r in m.scores_df_.iterrows():
            scores[int(r['dezena']) - 1] = float(r['freq_ponderada'])
    elif model_name == 'm1_overdue':
        m = FrequencyDelayModel(half_life=100.0, alpha=0.5).fit(binary_matrix)
        for _, r in m.scores_df_.iterrows():
            scores[int(r['dezena']) - 1] = float(r['razao_atraso'])
    elif model_name == 'm2_logistic':
        try:
            min_h = min(50, max(10, n_concursos // 4))
            m = MultiLabelMLModel(estimator='logistic_regression', windows=[5, 25],
                                  min_train_history=min_h, random_state=seed).fit(binary_matrix)
            for _, r in m.probabilities_df_.iterrows():
                scores[int(r['dezena']) - 1] = float(r['probabilidade_calibrada_sorteio'])
        except Exception:
            return _extract_base_model_scores('m1_momentum', binary_matrix, cfg, seed)
    elif model_name == 'm3_global':
        m = BayesianDecisionModel(alpha_0=cfg.get('alpha_0', 1.0),
                                  beta_0=cfg.get('beta_0', 9.0)).fit(binary_matrix)
        for _, r in m.posterior_df_.iterrows():
            scores[int(r['dezena']) - 1] = float(r['esperanca_posteriori'])
    elif model_name == 'm3_adaptive':
        m = BayesianDecisionModel(alpha_0=cfg.get('alpha_0', 1.0),
                                  beta_0=cfg.get('beta_0', 9.0),
                                  decay_half_life=150.0).fit(binary_matrix)
        for _, r in m.posterior_df_.iterrows():
            scores[int(r['dezena']) - 1] = float(r['esperanca_posteriori'])
    elif model_name == 'm4_gianella':
        try:
            m = GianellaColorTemplateModel(
                n_dezenas=n_dezenas,
                k=cfg.get('k', 6),
                group_size=cfg.get('gianella_group_size', 10)
            ).fit(binary_matrix)
            scores = m.dezena_scores_.copy()
        except Exception:
            return _extract_base_model_scores('m3_global', binary_matrix, cfg, seed)
    else:
        raise ValueError(f"Modelo base desconhecido: '{model_name}'")

    return _min_max_scale(scores)


class LotteryBaggingModel:
    """
    Ensemble Bagging (Bootstrap Aggregating) para Seleção de Dezenas em Loterias.
    Baseado em Aurélien Géron (2019), Hands-On Machine Learning (2ª Ed.), Capítulo 7:
    'Bagging and Pasting' (Leo Breiman, 1996, 'Bagging Predictors').

    Fundamentação Teórica:
    - Amostragem com reposição (Bootstrap) de m concursos a partir da base histórica total D:
      D_1, D_2, ..., D_B, onde |D_b| = max_samples * |D|.
    - Cada estimador de base é treinado de forma independente sobre uma réplica bootstrap,
      reduzindo a variância das estimativas probabilísticas sem aumentar o viés (Breiman, 1996).
    - Avaliação Out-of-Bag (OOB): em média ~36.8% dos concursos de treinamento não são amostrados
      em uma dada réplica b (pois lim_{m->inf} (1 - 1/m)^m = e^{-1} approx 0.368, Géron p. 195).
      Esses concursos são usados como conjunto de validação independente para calcular o oob_score_.
    - Agregação por Votação Suave (Soft Voting): média das pontuações normalizadas geradas pelas réplicas.
    - Suporta também Votação Rígida (Hard Voting): apuração por contagem de votos das top-k dezenas.
    """

    def __init__(self,
                 n_estimators: int = 10,
                 max_samples: float = 1.0,
                 voting: str = 'soft',
                 oob_score: bool = True,
                 random_state: int = 42,
                 base_models: Optional[List[str]] = None):
        """
        Parameters
        ----------
        n_estimators : int
            Número de réplicas bootstrap a treinar (default 10).
        max_samples : float
            Fração de concursos a amostrar com reposição em cada réplica (default 1.0).
        voting : str
            'soft' (default, média das probabilidades/escores) ou 'hard' (votação majoritária de dezenas).
        oob_score : bool
            Se True, calcula a avaliação Out-of-Bag (acerto médio nos sorteios não sorteados no bootstrap).
        random_state : int
            Semente para amostragem pseudo-aleatória.
        base_models : Optional[List[str]]
            Lista de nomes dos modelos de base a alternar entre as réplicas bootstrap.
        """
        if n_estimators <= 0:
            raise ValueError(f"n_estimators deve ser estritamente positivo (> 0). Recebido: {n_estimators}")
        if not (0.0 < max_samples <= 1.0):
            raise ValueError(f"max_samples deve estar no intervalo (0, 1]. Recebido: {max_samples}")
        if voting not in ('soft', 'hard'):
            raise ValueError(f"voting deve ser 'soft' ou 'hard'. Recebido: {voting}")

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.voting = voting
        self.oob_score = oob_score
        self.random_state = random_state
        self.base_models = base_models or [
            'm1_balanced', 'm1_momentum', 'm2_logistic',
            'm3_global', 'm3_adaptive', 'm4_gianella'
        ]

        self.estimators_: List[Dict[str, Any]] = []
        self.oob_score_: float = 0.0
        self.scores_df_: Optional[pd.DataFrame] = None
        self.top_k_: Optional[List[int]] = None

    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None) -> 'LotteryBaggingModel':
        """
        Ajusta o ensemble Bagging gerando réplicas bootstrap dos concursos históricos.
        """
        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        cfg = cfg or {}
        n_concursos, n_dezenas = binary_matrix.shape
        k = cfg.get('k', 6)
        rng = np.random.RandomState(self.random_state)
        m_samples = max(1, int(self.max_samples * n_concursos))

        self.estimators_ = []
        all_scores: List[np.ndarray] = []
        all_top_k: List[List[int]] = []
        oob_hits_total = 0
        oob_eval_count = 0

        for b in range(self.n_estimators):
            boot_idx = rng.choice(n_concursos, size=m_samples, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_concursos), boot_idx)
            X_b = binary_matrix[boot_idx]

            model_name = self.base_models[b % len(self.base_models)]
            s_b = _extract_base_model_scores(model_name, X_b, cfg, seed=self.random_state + b)
            all_scores.append(s_b)

            top_b = sorted(list(np.argsort(s_b)[::-1][:k] + 1))
            all_top_k.append(top_b)
            self.estimators_.append({
                'estimator_idx': b,
                'model_name': model_name,
                'scores': s_b,
                'top_k': top_b,
                'n_oob': len(oob_idx)
            })

            # Avaliação Out-of-Bag (OOB)
            if self.oob_score and len(oob_idx) > 0:
                top_b_set = set(top_b)
                for idx_oob in oob_idx:
                    drawn = set(np.where(binary_matrix[idx_oob] == 1)[0] + 1)
                    oob_hits_total += len(top_b_set & drawn)
                    oob_eval_count += 1

        self.oob_score_ = float(oob_hits_total / oob_eval_count) if oob_eval_count > 0 else 0.0

        soft_scores = np.mean(all_scores, axis=0)
        hard_votes = np.zeros(n_dezenas, dtype=float)
        for top_b in all_top_k:
            for dez in top_b:
                hard_votes[dez - 1] += 1.0

        dezenas = np.arange(1, n_dezenas + 1)
        self.scores_df_ = pd.DataFrame({
            'dezena': dezenas,
            'escore_bagging': np.round(soft_scores, 5),
            'votos_hard': hard_votes.astype(int)
        }).sort_values('escore_bagging', ascending=False).reset_index(drop=True)
        self.top_k_ = sorted(self.scores_df_.head(k)['dezena'].tolist())
        return self

    def predict_top_k(self, k: Optional[int] = None, voting: Optional[str] = None) -> List[int]:
        """
        Retorna as top K dezenas selecionadas pelo ensemble Bagging.
        """
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")
        if k is None:
            k = len(self.top_k_) if self.top_k_ is not None else 6
        if k <= 0:
            return []
        k = min(k, len(self.scores_df_))

        use_voting = voting or self.voting
        if use_voting == 'hard':
            df_sorted = self.scores_df_.sort_values(['votos_hard', 'escore_bagging'], ascending=[False, False])
        else:
            df_sorted = self.scores_df_.sort_values('escore_bagging', ascending=False)

        return sorted(df_sorted.head(k)['dezena'].tolist())

    def get_scores_df(self) -> pd.DataFrame:
        """Retorna o DataFrame detalhado com escores do Bagging."""
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")
        return self.scores_df_.copy()


class LotteryPastingModel:
    """
    Ensemble Pasting (Non-replacement Aggregating) para Seleção de Dezenas em Loterias.
    Baseado em Aurélien Géron (2019), Hands-On Machine Learning (2ª Ed.), Capítulo 7:
    'Bagging and Pasting' (Leo Breiman, 1999, 'Pasting Small Votes for Classification in Large Databases and On-Line').

    Fundamentação Teórica:
    - Amostragem sem reposição (Non-replacement Sampling) de instâncias a partir da base histórica total D:
      D_1, D_2, ..., D_B, onde |D_b| = max_samples * |D| e cada concurso aparece no máximo uma vez em cada réplica.
    - Preserva a estrutura ordinal/temporal sem duplicatas artificiais dentro de cada subconjunto de treinamento.
    - Diferença fundamental para o Bagging (Breiman 1996 vs Breiman 1999):
      Bagging usa bootstrap (com reposição), enquanto Pasting realiza partição aleatória sem reposição,
      fornecendo estimativas com menor viés em cada subamostra, agregadas por soft ou hard voting.
    """

    def __init__(self,
                 n_estimators: int = 10,
                 max_samples: float = 0.8,
                 voting: str = 'soft',
                 random_state: int = 42,
                 base_models: Optional[List[str]] = None):
        """
        Parameters
        ----------
        n_estimators : int
            Número de réplicas de subamostras sem reposição (default 10).
        max_samples : float
            Fração de concursos a amostrar SEM reposição em cada estimador (default 0.8).
        voting : str
            'soft' (default, média das pontuações/probabilidades) ou 'hard' (votação majoritária de dezenas).
        random_state : int
            Semente para amostragem pseudo-aleatória.
        base_models : Optional[List[str]]
            Lista de nomes dos modelos de base a alternar entre as réplicas.
        """
        if n_estimators <= 0:
            raise ValueError(f"n_estimators deve ser estritamente positivo (> 0). Recebido: {n_estimators}")
        if not (0.0 < max_samples <= 1.0):
            raise ValueError(f"max_samples deve estar no intervalo (0, 1]. Recebido: {max_samples}")
        if voting not in ('soft', 'hard'):
            raise ValueError(f"voting deve ser 'soft' ou 'hard'. Recebido: {voting}")

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.voting = voting
        self.random_state = random_state
        self.base_models = base_models or [
            'm1_balanced', 'm1_momentum', 'm2_logistic',
            'm3_global', 'm3_adaptive', 'm4_gianella'
        ]

        self.estimators_: List[Dict[str, Any]] = []
        self.scores_df_: Optional[pd.DataFrame] = None
        self.top_k_: Optional[List[int]] = None

    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None) -> 'LotteryPastingModel':
        """
        Ajusta o ensemble Pasting gerando réplicas sem reposição dos concursos históricos.
        """
        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        cfg = cfg or {}
        n_concursos, n_dezenas = binary_matrix.shape
        k = cfg.get('k', 6)
        rng = np.random.RandomState(self.random_state)
        m_samples = max(1, int(self.max_samples * n_concursos))

        self.estimators_ = []
        all_scores: List[np.ndarray] = []
        all_top_k: List[List[int]] = []

        for b in range(self.n_estimators):
            # Amostragem SEM reposição (Pasting, Breiman 1999)
            paste_idx = rng.choice(n_concursos, size=m_samples, replace=False)
            paste_idx = np.sort(paste_idx)  # Preserva coerência cronológica
            X_b = binary_matrix[paste_idx]

            model_name = self.base_models[b % len(self.base_models)]
            s_b = _extract_base_model_scores(model_name, X_b, cfg, seed=self.random_state + b)
            all_scores.append(s_b)

            top_b = sorted(list(np.argsort(s_b)[::-1][:k] + 1))
            all_top_k.append(top_b)
            self.estimators_.append({
                'estimator_idx': b,
                'model_name': model_name,
                'scores': s_b,
                'top_k': top_b,
                'n_samples': m_samples
            })

        soft_scores = np.mean(all_scores, axis=0)
        hard_votes = np.zeros(n_dezenas, dtype=float)
        for top_b in all_top_k:
            for dez in top_b:
                hard_votes[dez - 1] += 1.0

        dezenas = np.arange(1, n_dezenas + 1)
        self.scores_df_ = pd.DataFrame({
            'dezena': dezenas,
            'escore_pasting': np.round(soft_scores, 5),
            'votos_hard': hard_votes.astype(int)
        }).sort_values('escore_pasting', ascending=False).reset_index(drop=True)
        self.top_k_ = sorted(self.scores_df_.head(k)['dezena'].tolist())
        return self

    def predict_top_k(self, k: Optional[int] = None, voting: Optional[str] = None) -> List[int]:
        """
        Retorna as top K dezenas selecionadas pelo ensemble Pasting.
        """
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")
        if k is None:
            k = len(self.top_k_) if self.top_k_ is not None else 6
        if k <= 0:
            return []
        k = min(k, len(self.scores_df_))

        use_voting = voting or self.voting
        if use_voting == 'hard':
            df_sorted = self.scores_df_.sort_values(['votos_hard', 'escore_pasting'], ascending=[False, False])
        else:
            df_sorted = self.scores_df_.sort_values('escore_pasting', ascending=False)

        return sorted(df_sorted.head(k)['dezena'].tolist())

    def get_scores_df(self) -> pd.DataFrame:
        """Retorna o DataFrame detalhado com escores do Pasting."""
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")
        return self.scores_df_.copy()


class LotteryAdaBoostModel:
    """
    Ensemble AdaBoost (Adaptive Boosting) para Seleção de Dezenas em Loterias.
    Baseado em Aurélien Géron (2019), Hands-On Machine Learning (2ª Ed.), Capítulo 7:
    'Boosting - AdaBoost' (Yoav Freund & Robert E. Schapire, 1997).

    Fundamentação Matemática e Algorítmica:
    1. Pesos das Instâncias (Concursos Históricos):
       Inicialmente, w_i^{(1)} = 1 / m para todo concurso i in {1, ..., m} (Géron p. 201).
    2. Taxa de Erro Ponderada do j-ésimo preditor (Equação 7-1):
       r_j = (sum_{i=1}^m w_i^{(j)} * e_{j, i}) / (sum_{i=1}^m w_i^{(j)})
       onde e_{j, i} = 1 - (acertos_{j, i} / k) in [0, 1] é a taxa de erro empírica no concurso i.
    3. Peso do Preditor alpha_j (Equação 7-2 com taxa de aprendizado eta):
       alpha_j = eta * 0.5 * ln((1 - r_j) / r_j)
       Preditores mais precisos recebem peso alpha_j significativamente maior no voto do ensemble.
    4. Atualização e Normalização dos Pesos das Instâncias (Equação 7-3):
       w_i^{(j+1)} = w_i^{(j)} * exp(alpha_j * e_{j, i})
       w_i^{(j+1)} = w_i^{(j+1)} / sum_l w_l^{(j+1)}
       Concursos difíceis (onde os modelos anteriores erraram mais) têm seus pesos aumentados,
       forçando os próximos preditores a focarem nos padrões não antecipados.
    5. Predição Combinada do Ensemble (Equação 7-4):
       Score(d) = sum_{j=1}^M alpha_j * S_j(d)
    """

    def __init__(self,
                 n_estimators: int = 10,
                 learning_rate: float = 1.0,
                 random_state: int = 42,
                 base_models: Optional[List[str]] = None):
        """
        Parameters
        ----------
        n_estimators : int
            Número de rodadas sequenciais de boosting (default 10).
        learning_rate : float
            Taxa de aprendizado eta que escala o peso de cada preditor (default 1.0).
        random_state : int
            Semente para reprodutibilidade da amostragem ponderada.
        base_models : Optional[List[str]]
            Conjunto de modelos analíticos existentes para utilizar no boosting.
        """
        if n_estimators <= 0:
            raise ValueError(f"n_estimators deve ser estritamente positivo (> 0). Recebido: {n_estimators}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate deve ser estritamente positivo (> 0). Recebido: {learning_rate}")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.base_models = base_models or [
            'm1_balanced', 'm1_momentum', 'm1_overdue',
            'm3_global', 'm3_adaptive', 'm4_gianella'
        ]

        self.estimators_: List[Dict[str, Any]] = []
        self.estimator_weights_: Optional[np.ndarray] = None
        self.estimator_errors_: Optional[np.ndarray] = None
        self.scores_df_: Optional[pd.DataFrame] = None
        self.top_k_: Optional[List[int]] = None

    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None) -> 'LotteryAdaBoostModel':
        """
        Ajusta sequencialmente o AdaBoost sobre os sorteios históricos com reponderação de instâncias.
        """
        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        cfg = cfg or {}
        n_concursos, n_dezenas = binary_matrix.shape
        k = cfg.get('k', 6)
        rng = np.random.RandomState(self.random_state)

        # 1. Pesos iniciais uniformes das instâncias (Géron p. 201)
        w = np.full(n_concursos, 1.0 / n_concursos, dtype=float)

        self.estimators_ = []
        weights_list: List[float] = []
        errors_list: List[float] = []
        combined_scores = np.zeros(n_dezenas, dtype=float)

        for j in range(self.n_estimators):
            # Amostra concursos guiada pela distribuição de pesos w (foco nos casos difíceis)
            sample_idx = rng.choice(n_concursos, size=n_concursos, replace=True, p=w)
            X_sample = binary_matrix[sample_idx]

            model_name = self.base_models[j % len(self.base_models)]
            s_j = _extract_base_model_scores(model_name, X_sample, cfg, seed=self.random_state + j)
            top_j = sorted(list(np.argsort(s_j)[::-1][:k] + 1))
            top_j_set = set(top_j)

            # 2. Avaliação de erro de predição nas instâncias de treino (Géron Eq 7-1)
            errors = np.zeros(n_concursos, dtype=float)
            for i in range(n_concursos):
                drawn_i = set(np.where(binary_matrix[i] == 1)[0] + 1)
                hits_i = len(top_j_set & drawn_i)
                errors[i] = 1.0 - (hits_i / float(k))

            # Taxa de erro ponderada r_j
            r_j = float(np.sum(w * errors) / np.sum(w))
            r_j = max(1e-6, min(1.0 - 1e-6, r_j))

            # 3. Peso do preditor alpha_j (Géron Eq 7-2)
            alpha_j = float(self.learning_rate * 0.5 * np.log((1.0 - r_j) / r_j))
            if alpha_j < 0.01:
                alpha_j = 0.01

            # 4. Atualização dos pesos das instâncias (Géron Eq 7-3)
            w = w * np.exp(alpha_j * errors)
            w = w / np.sum(w)  # Normalização

            self.estimators_.append({
                'stage': j,
                'model_name': model_name,
                'alpha': alpha_j,
                'error': r_j,
                'scores': s_j,
                'top_k': top_j
            })
            weights_list.append(alpha_j)
            errors_list.append(r_j)

            combined_scores += alpha_j * s_j

        self.estimator_weights_ = np.array(weights_list)
        self.estimator_errors_ = np.array(errors_list)

        norm_scores = _min_max_scale(combined_scores)
        dezenas = np.arange(1, n_dezenas + 1)
        self.scores_df_ = pd.DataFrame({
            'dezena': dezenas,
            'escore_adaboost': np.round(norm_scores, 5),
            'escore_ponderado_bruto': np.round(combined_scores, 5)
        }).sort_values('escore_adaboost', ascending=False).reset_index(drop=True)
        self.top_k_ = sorted(self.scores_df_.head(k)['dezena'].tolist())
        return self

    def predict_top_k(self, k: Optional[int] = None) -> List[int]:
        """
        Retorna as top K dezenas selecionadas pelo AdaBoost ponderado.
        """
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")
        if k is None:
            k = len(self.top_k_) if self.top_k_ is not None else 6
        if k <= 0:
            return []
        k = min(k, len(self.scores_df_))
        return sorted(self.scores_df_.head(k)['dezena'].tolist())

    def get_scores_df(self) -> pd.DataFrame:
        """Retorna o DataFrame detalhado com escores do AdaBoost."""
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")
        return self.scores_df_.copy()


class LotteryGradientBoostingModel:
    """
    Ensemble Gradient Boosting (Residual Boosting) para Seleção de Dezenas em Loterias.
    Baseado em Aurélien Géron (2019), Hands-On Machine Learning (2ª Ed.), Capítulo 7:
    'Boosting - Gradient Boosting' (Jerome H. Friedman, 1999; Leo Breiman, 1997).

    Fundamentação Teórica e Formulação Matemática:
    1. Ajuste Sequencial aos Erros Residuais (Gradiente Negativo):
       Diferente do AdaBoost (que atualiza pesos de instâncias), o Gradient Boosting ajusta cada novo
       preditor h_m diretamente aos resíduos deixados pelo preditor anterior (Géron p. 203-204):
       R^{(0)} = Y - F_0
       h_m approx R^{(m-1)}
       F_m = F_{m-1} + eta * h_m
       R^{(m)} = R^{(m-1)} - eta * h_m
    2. Modelo Inicial F_0:
       Estimativa suavizada a priori baseada na esperança da conjugada Bayesiana global (Método 3).
    3. Regularização por Encolhimento (Shrinkage, Géron p. 205):
       O hiperparâmetro learning_rate (eta in (0, 1]) escala a contribuição de cada etapa,
       forçando o modelo a aprender representações graduais e evitando sobreajuste.
    4. Predição Estagiada (Staged Predict, Géron p. 206):
       Permite inspecionar a evolução e o refinamento do palpite etapa a etapa (staged_predict_top_k).
    """

    def __init__(self,
                 n_estimators: int = 10,
                 learning_rate: float = 0.5,
                 random_state: int = 42,
                 base_models: Optional[List[str]] = None):
        """
        Parameters
        ----------
        n_estimators : int
            Número de estágios de ajuste residual sequencial (default 10).
        learning_rate : float
            Taxa de aprendizado / encolhimento eta (default 0.5).
        random_state : int
            Semente para reprodutibilidade.
        base_models : Optional[List[str]]
            Lista de estratégias de resíduo a utilizar.
        """
        if n_estimators <= 0:
            raise ValueError(f"n_estimators deve ser estritamente positivo (> 0). Recebido: {n_estimators}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate deve ser estritamente positivo (> 0). Recebido: {learning_rate}")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.base_models = base_models or ['residual_momentum', 'residual_delay', 'residual_bayes', 'residual_ml']

        self.staged_scores_: List[np.ndarray] = []
        self.scores_df_: Optional[pd.DataFrame] = None
        self.top_k_: Optional[List[int]] = None

    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None) -> 'LotteryGradientBoostingModel':
        """
        Ajusta sequencialmente o Gradient Boosting aos resíduos do histórico.
        """
        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        cfg = cfg or {}
        n_concursos, n_dezenas = binary_matrix.shape
        k = cfg.get('k', 6)

        # 1. Modelo inicial F_0: Esperança a priori / posteriori bayesiana suavizada
        alpha_0 = cfg.get('alpha_0', 1.0)
        beta_0 = cfg.get('beta_0', 9.0)
        F_current = (binary_matrix.sum(axis=0) + alpha_0) / (n_concursos + alpha_0 + beta_0)
        F_current = _min_max_scale(F_current)

        # Resíduos iniciais: R_{t, d} = Y_{t, d} - F_current(d)
        residuals = binary_matrix.astype(float) - F_current[None, :]
        self.staged_scores_ = [F_current.copy()]

        for m in range(self.n_estimators):
            # 2. Ajuste do preditor de resíduos h_m (Géron p. 204)
            # Ponderação temporal adaptativa sobre os resíduos recentes (momentum residual)
            half_life = 50.0 + (m * 20.0)
            decay = 0.5 ** (1.0 / half_life)
            time_weights = decay ** np.arange(n_concursos - 1, -1, -1)
            time_weights /= np.sum(time_weights)

            # Gradiente de resíduos recente ponderado
            h_m = np.sum(residuals * time_weights[:, None], axis=0)
            h_norm = _min_max_scale(h_m)

            # 3. Atualização do ensemble com regularização por encolhimento (Shrinkage, Géron p. 205)
            # F_{m} = F_{m-1} + learning_rate * (h_norm - 0.5)
            F_current = F_current + self.learning_rate * (h_norm - 0.5)
            F_current = np.clip(F_current, 0.0, None)

            # 4. Atualização dos resíduos para a próxima etapa
            residuals = residuals - self.learning_rate * (h_norm[None, :] - 0.5)
            self.staged_scores_.append(_min_max_scale(F_current).copy())

        final_scores = _min_max_scale(F_current)
        dezenas = np.arange(1, n_dezenas + 1)
        self.scores_df_ = pd.DataFrame({
            'dezena': dezenas,
            'escore_gradboost': np.round(final_scores, 5),
            'escore_acumulado': np.round(F_current, 5)
        }).sort_values('escore_gradboost', ascending=False).reset_index(drop=True)
        self.top_k_ = sorted(self.scores_df_.head(k)['dezena'].tolist())
        return self

    def predict_top_k(self, k: Optional[int] = None) -> List[int]:
        """
        Retorna as top K dezenas selecionadas pelo Gradient Boosting.
        """
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")
        if k is None:
            k = len(self.top_k_) if self.top_k_ is not None else 6
        if k <= 0:
            return []
        k = min(k, len(self.scores_df_))
        return sorted(self.scores_df_.head(k)['dezena'].tolist())

    def staged_predict_top_k(self, k: Optional[int] = None) -> Generator[List[int], None, None]:
        """
        Iterador sobre as predições do ensemble em cada estágio de treinamento (Géron p. 206).
        """
        if not self.staged_scores_:
            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")
        k = k if k is not None else (len(self.top_k_) if self.top_k_ is not None else 6)
        for s in self.staged_scores_:
            if k <= 0:
                yield []
            else:
                k_actual = min(k, len(s))
                yield sorted(list(np.argsort(s)[::-1][:k_actual] + 1))

    def get_scores_df(self) -> pd.DataFrame:
        """Retorna o DataFrame detalhado com escores do Gradient Boosting."""
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")
        return self.scores_df_.copy()


class LotteryStackingModel:
    """
    Ensemble Stacking (Stacked Generalization / Blending) para Seleção de Dezenas em Loterias.
    Baseado em Aurélien Géron (2019), Hands-On Machine Learning (2ª Ed.), Capítulo 7:
    'Stacking' (David H. Wolpert, 1992, 'Stacked Generalization', Neural Networks).

    Fundamentação Teórica e Algorítmica:
    1. Estrutura em Camadas (Layered Architecture, Géron p. 208-210):
       - Camada 1 (Base Predictors): Preditores de paradigmas estatísticos e analíticos distintos:
         M1 (Estatístico/Momentum/Overdue), M2 (Regressão Logística L2), M3 (Bayesiano Conjugado/Adaptativo)
         e M4 (Template Gianella).
       - Camada 2 (Blender / Meta-Learner): Um modelo supervisionado treinado sobre as predições
         geradas pelos estimadores da Camada 1 em partição holdout (ou out-of-fold).
    2. Treinamento do Meta-Learner (Blending, Géron Figura 7-13 e 7-14):
       - A base histórica D é dividida temporalmente em D_train e D_blend (holdout).
       - Os estimadores da Camada 1 são ajustados em D_train e produzem matriz de escores de dezenas.
       - A matriz de meta-features X_meta (shape n_dezenas x n_modelos) é associada ao vetor target y_meta
         (frequência empírica observada no conjunto holdout D_blend).
       - O meta-learner (Ridge regularizado L2) aprende os coeficientes ótimos beta_1, ..., beta_M
         para ponderar os modelos base com regularização que evita sobreajuste.
    3. Inferência para o Concurso N+1 (Géron Figura 7-15):
       - Os modelos da Camada 1 geram as pontuações na base completa até N.
       - As pontuações passam pelo meta-learner calibrado para produzir o escore final de stacking.
    """

    def __init__(self,
                 holdout_ratio: float = 0.25,
                 meta_learner: str = 'ridge',
                 alpha: float = 1.0,
                 random_state: int = 42,
                 base_models: Optional[List[str]] = None):
        """
        Parameters
        ----------
        holdout_ratio : float
            Fração dos concursos mais recentes reservados para treinar o blender (default 0.25).
        meta_learner : str
            Algoritmo meta-aprendiz ('ridge' ou 'linear').
        alpha : float
            Parâmetro de regularização L2 para o meta-aprendiz Ridge (default 1.0).
        random_state : int
            Semente para reprodutibilidade.
        base_models : Optional[List[str]]
            Modelos de base a integrar na Camada 1.
        """
        if not (0.05 <= holdout_ratio <= 0.50):
            raise ValueError(f"holdout_ratio deve estar entre 0.05 e 0.50. Recebido: {holdout_ratio}")
        if meta_learner not in ('ridge', 'linear'):
            raise ValueError(f"meta_learner deve ser 'ridge' ou 'linear'. Recebido: {meta_learner}")

        self.holdout_ratio = holdout_ratio
        self.meta_learner = meta_learner
        self.alpha = alpha
        self.random_state = random_state
        self.base_models = base_models or [
            'm1_balanced', 'm1_momentum', 'm1_overdue',
            'm2_logistic', 'm3_global', 'm3_adaptive', 'm4_gianella'
        ]

        self.blender_ = None
        self.blender_weights_: Dict[str, float] = {}
        self.scores_df_: Optional[pd.DataFrame] = None
        self.top_k_: Optional[List[int]] = None

    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None) -> 'LotteryStackingModel':
        """
        Ajusta os estimadores da Camada 1 e treina o Blender / Meta-Learner da Camada 2.
        """
        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        cfg = cfg or {}
        n_concursos, n_dezenas = binary_matrix.shape
        k = cfg.get('k', 6)

        # 1. Divisão em Treino de Base e Holdout para o Blender (Géron p. 209)
        n_blend = max(10, int(self.holdout_ratio * n_concursos))
        split_idx = max(10, n_concursos - n_blend)

        X_train_base = binary_matrix[:split_idx]
        X_holdout = binary_matrix[split_idx:]

        # 2. Camada 1: Extrai escores dos modelos base ajustados em X_train_base
        n_base = len(self.base_models)
        X_meta = np.zeros((n_dezenas, n_base), dtype=float)
        for j, m_name in enumerate(self.base_models):
            X_meta[:, j] = _extract_base_model_scores(m_name, X_train_base, cfg, seed=self.random_state + j)

        # Target do blender: frequência empírica normalizada das dezenas no período holdout
        y_meta = X_holdout.sum(axis=0).astype(float) / float(len(X_holdout))
        y_meta_norm = _min_max_scale(y_meta)

        # 3. Camada 2: Treinamento do Blender / Meta-Learner (Géron Eq. Stacking p. 209-210)
        from sklearn.linear_model import Ridge, LinearRegression
        if self.meta_learner == 'ridge':
            blender = Ridge(alpha=self.alpha, fit_intercept=True, random_state=self.random_state)
        else:
            blender = LinearRegression(fit_intercept=True)

        blender.fit(X_meta, y_meta_norm)
        self.blender_ = blender

        coefs = blender.coef_
        self.blender_weights_ = {
            m_name: float(np.round(coefs[j], 4)) for j, m_name in enumerate(self.base_models)
        }

        # 4. Inferência para o concurso alvo (N+1): escores dos modelos base na base total
        X_meta_full = np.zeros((n_dezenas, n_base), dtype=float)
        for j, m_name in enumerate(self.base_models):
            X_meta_full[:, j] = _extract_base_model_scores(m_name, binary_matrix, cfg, seed=self.random_state + j)

        # O Blender gera as predições combinadas
        raw_stacking_scores = blender.predict(X_meta_full)
        final_scores = _min_max_scale(raw_stacking_scores)

        dezenas = np.arange(1, n_dezenas + 1)
        res_dict = {
            'dezena': dezenas,
            'escore_stacking': np.round(final_scores, 5),
            'meta_pred_bruto': np.round(raw_stacking_scores, 5)
        }
        for j, m_name in enumerate(self.base_models):
            res_dict[f'base_{m_name}'] = np.round(X_meta_full[:, j], 4)

        self.scores_df_ = pd.DataFrame(res_dict).sort_values('escore_stacking', ascending=False).reset_index(drop=True)
        self.top_k_ = sorted(self.scores_df_.head(k)['dezena'].tolist())
        return self

    def predict_top_k(self, k: Optional[int] = None) -> List[int]:
        """
        Retorna as top K dezenas selecionadas pelo ensemble Stacking.
        """
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")
        if k is None:
            k = len(self.top_k_) if self.top_k_ is not None else 6
        if k <= 0:
            return []
        k = min(k, len(self.scores_df_))
        return sorted(self.scores_df_.head(k)['dezena'].tolist())

    def get_scores_df(self) -> pd.DataFrame:
        """Retorna o DataFrame detalhado com escores do Stacking."""
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")
        return self.scores_df_.copy()



def predict_bagging(binary_matrix: np.ndarray,
                    cfg: Dict,
                    k: Optional[int] = None,
                    n_estimators: int = 10,
                    random_state: int = 42) -> List[int]:
    """Predição direta das top K dezenas via Ensemble Bagging."""
    k_actual = cfg['k'] if k is None else k
    model = LotteryBaggingModel(n_estimators=n_estimators, random_state=random_state)
    model.fit(binary_matrix, cfg=cfg)
    return model.predict_top_k(k_actual)


def predict_adaboost(binary_matrix: np.ndarray,
                     cfg: Dict,
                     k: Optional[int] = None,
                     n_estimators: int = 10,
                     learning_rate: float = 1.0,
                     random_state: int = 42) -> List[int]:
    """Predição direta das top K dezenas via Ensemble AdaBoost."""
    k_actual = cfg['k'] if k is None else k
    model = LotteryAdaBoostModel(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model.fit(binary_matrix, cfg=cfg)
    return model.predict_top_k(k_actual)


def predict_gradient_boosting(binary_matrix: np.ndarray,
                              cfg: Dict,
                              k: Optional[int] = None,
                              n_estimators: int = 10,
                              learning_rate: float = 0.5,
                              random_state: int = 42) -> List[int]:
    """Predição direta das top K dezenas via Ensemble Gradient Boosting."""
    k_actual = cfg['k'] if k is None else k
    model = LotteryGradientBoostingModel(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    model.fit(binary_matrix, cfg=cfg)
    return model.predict_top_k(k_actual)




def predict_pasting(binary_matrix: np.ndarray,
                    cfg: Dict,
                    k: Optional[int] = None,
                    n_estimators: int = 10,
                    max_samples: float = 0.8,
                    voting: str = 'soft',
                    random_state: int = 42) -> List[int]:
    """Predição direta das top K dezenas via Ensemble Pasting."""
    k_actual = cfg['k'] if k is None else k
    model = LotteryPastingModel(n_estimators=n_estimators, max_samples=max_samples,
                                voting=voting, random_state=random_state)
    model.fit(binary_matrix, cfg=cfg)
    return model.predict_top_k(k_actual)


def predict_stacking(binary_matrix: np.ndarray,
                     cfg: Dict,
                     k: Optional[int] = None,
                     holdout_ratio: float = 0.25,
                     meta_learner: str = 'ridge',
                     alpha: float = 1.0,
                     random_state: int = 42) -> List[int]:
    """Predição direta das top K dezenas via Ensemble Stacking."""
    k_actual = cfg['k'] if k is None else k
    model = LotteryStackingModel(holdout_ratio=holdout_ratio, meta_learner=meta_learner,
                                 alpha=alpha, random_state=random_state)
    model.fit(binary_matrix, cfg=cfg)
    return model.predict_top_k(k_actual)


# ==============================================================================
# 4. FUNÇÃO COMPARATIVA E RELATÓRIO EXECUTIVO
# ==============================================================================

def run_all_methods(filepath: str,
                    next_concurso: Optional[int] = None,
                    include_hybrids: bool = False,
                    include_ensembles: Optional[bool] = None) -> Dict[str, object]:
    """
    Executa os métodos sobre a base histórica e consolida os resultados
    para o próximo concurso especificado (ou inferido automaticamente).
    """
    return run_all_lottery_methods(filepath, game='megasena', next_concurso=next_concurso,
                                   include_hybrids=include_hybrids, include_ensembles=include_ensembles)


def run_all_lottery_methods(filepath: str,
                            game: str = 'auto',
                            next_concurso: Optional[int] = None,
                            include_hybrids: bool = False,
                            include_ensembles: Optional[bool] = None) -> Dict:
    """
    Executa os métodos analíticos para qualquer loteria (Mega-Sena, Lotofácil, Quina).
    """
    if include_ensembles is None:
        include_ensembles = include_hybrids

    df, binary_matrix, cfg = load_lottery_data(filepath, game=game)
    if next_concurso is None:
        next_concurso = int(df['Concurso'].max()) + 1

    k = cfg['k']
    alpha_0 = cfg['alpha_0']
    beta_0 = cfg['beta_0']

    # 1. Método 1
    m1 = FrequencyDelayModel(half_life=100.0, alpha=0.5)
    m1.fit(binary_matrix)
    pred_m1 = m1.predict_top_k(k, strategy='balanced')
    pred_m1_overdue = m1.predict_top_k(k, strategy='overdue')
    pred_m1_momentum = m1.predict_top_k(k, strategy='momentum')

    # 2. Método 2
    m2 = MultiLabelMLModel(estimator='logistic_regression', random_state=42)
    m2.fit(binary_matrix)
    pred_m2 = m2.predict_top_k(k)

    # 3. Método 3
    # 3a. Bayesiano Estático Global
    m3_static = BayesianDecisionModel(alpha_0=alpha_0, beta_0=beta_0)
    m3_static.fit(binary_matrix)
    pred_m3_static = m3_static.predict_top_k(k)

    # 3b. Bayesiano Dinâmico com Evidência Temporal (Half-life = 150)
    m3_dyn = BayesianDecisionModel(alpha_0=alpha_0, beta_0=beta_0, decay_half_life=150.0)
    m3_dyn.fit(binary_matrix)
    pred_m3_dyn = m3_dyn.predict_top_k(k)

    # 4. Método 4: A Geometria do Acaso (Gianella)
    group_size = cfg.get('gianella_group_size', 10)
    m4 = GianellaColorTemplateModel(n_dezenas=cfg['n_dezenas'], k=k, group_size=group_size)
    m4.fit(binary_matrix)
    pred_m4 = m4.predict_top_k(k)

    # Consenso Bayesiano Multi-Modelo (Decisão Ótima de Bayes)
    vote_weights = {
        'm1': (pred_m1, 1.0),
        'm2': (pred_m2, 2.0),
        'm3_static': (pred_m3_static, 2.5),
        'm3_dyn': (pred_m3_dyn, 2.5),
        'm4': (pred_m4, 1.5)
    }
    ball_scores = np.zeros(cfg['n_dezenas'], dtype=float)
    for _, (preds, w) in vote_weights.items():
        for ball in preds:
            ball_scores[ball - 1] += w
    pred_consenso = sorted(list(np.argsort(ball_scores)[::-1][:k] + 1))

    # Escores auxiliares para os métodos híbridos
    mom_scores = np.zeros(cfg['n_dezenas'], dtype=float)
    bal_scores = np.zeros(cfg['n_dezenas'], dtype=float)
    for _, r in m1.scores_df_.iterrows():
        d_idx = int(r['dezena']) - 1
        mom_scores[d_idx] = float(r['freq_ponderada'])
        bal_scores[d_idx] = float(r['escore_composto'])

    ml_scores = np.zeros(cfg['n_dezenas'], dtype=float)
    try:
        df_probs = m2.get_probabilities_df()
        for _, r in df_probs.iterrows():
            ml_scores[int(r['dezena']) - 1] = float(r['probabilidade_calibrada_sorteio'])
    except Exception:
        ml_scores = mom_scores.copy()

    # 5. Métodos Híbridos Gianella + Bayes
    m_hybrid = GianellaBayesHybridModel(
        n_dezenas=cfg['n_dezenas'],
        k=k,
        group_size=group_size,
        alpha_0=alpha_0,
        beta_0=beta_0,
        decay_half_life=150.0
    )
    m_hybrid.fit(binary_matrix, consensus_scores=ball_scores, ml_scores=ml_scores,
                 momentum_scores=mom_scores, balanced_scores=bal_scores)

    pred_g_bayes_global = m_hybrid.predict_gianella_bayes_global(k)
    pred_g_bayes_adaptive = m_hybrid.predict_gianella_bayes_adaptive(k)
    pred_g_best_bayes = m_hybrid.predict_gianella_best_bayes_decision(k)
    pred_g_lr = m_hybrid.predict_gianella_logistic_regression(k)
    pred_g_mom = m_hybrid.predict_gianella_momentum(k)
    pred_g_bal = m_hybrid.predict_gianella_balanced(k)

    pred_bb_lr = m_hybrid.predict_best_bayes_logistic_regression(k)
    pred_bb_mom = m_hybrid.predict_best_bayes_momentum(k)
    pred_bb_bal = m_hybrid.predict_best_bayes_balanced(k)

    pred_bg_lr = m_hybrid.predict_bayes_global_logistic_regression(k)
    pred_bg_mom = m_hybrid.predict_bayes_global_momentum(k)
    pred_bg_bal = m_hybrid.predict_bayes_global_balanced(k)

    pred_ba_lr = m_hybrid.predict_bayes_adaptive_logistic_regression(k)
    pred_ba_mom = m_hybrid.predict_bayes_adaptive_momentum(k)
    pred_ba_bal = m_hybrid.predict_bayes_adaptive_balanced(k)

    predictions_hybrid = {
        'Gianella + Bayes Global': pred_g_bayes_global,
        'Gianella + Bayes Adaptativo': pred_g_bayes_adaptive,
        'Gianella + Melhor Decisão de Bayes': pred_g_best_bayes,
        'Gianella + Regressão Logística': pred_g_lr,
        'Gianella + Freq/Atraso (Momentum)': pred_g_mom,
        'Gianella + Freq/Atraso (Balanceado)': pred_g_bal,
        'Melhor Decisão de Bayes + Regressão Logística': pred_bb_lr,
        'Melhor Decisão de Bayes + Freq/Atraso (Momentum)': pred_bb_mom,
        'Melhor Decisão de Bayes + Freq/Atraso (Balanceado)': pred_bb_bal,
        'Bayes Global + Regressão Logística': pred_bg_lr,
        'Bayes Global + Frequência/Atraso (Momentum)': pred_bg_mom,
        'Bayes Global + Frequência/Atraso (Balanceado)': pred_bg_bal,
        'Bayes Adaptativo + Regressão Logística': pred_ba_lr,
        'Bayes Adaptativo + Frequência/Atraso (Momentum)': pred_ba_mom,
        'Bayes Adaptativo + Frequência/Atraso (Balanceado)': pred_ba_bal
    }

        # 6. Métodos Ensemble (Géron, Cap. 7: Bagging, Pasting, AdaBoost, Gradient Boosting, Stacking)
    m_bagging = LotteryBaggingModel(n_estimators=10, random_state=42)
    m_bagging.fit(binary_matrix, cfg=cfg)
    pred_bagging = m_bagging.predict_top_k(k)

    m_pasting = LotteryPastingModel(n_estimators=10, max_samples=0.8, random_state=42)
    m_pasting.fit(binary_matrix, cfg=cfg)
    pred_pasting = m_pasting.predict_top_k(k)

    m_adaboost = LotteryAdaBoostModel(n_estimators=10, learning_rate=1.0, random_state=42)
    m_adaboost.fit(binary_matrix, cfg=cfg)
    pred_adaboost = m_adaboost.predict_top_k(k)

    m_gradboost = LotteryGradientBoostingModel(n_estimators=10, learning_rate=0.5, random_state=42)
    m_gradboost.fit(binary_matrix, cfg=cfg)
    pred_gradboost = m_gradboost.predict_top_k(k)

    m_stacking = LotteryStackingModel(holdout_ratio=0.25, random_state=42)
    m_stacking.fit(binary_matrix, cfg=cfg)
    pred_stacking = m_stacking.predict_top_k(k)

    predictions_ensemble = {
        'Ensemble Bagging': pred_bagging,
        'Ensemble Pasting': pred_pasting,
        'Ensemble AdaBoost': pred_adaboost,
        'Ensemble Gradient Boosting': pred_gradboost,
        'Ensemble Stacking': pred_stacking
    }

    predictions = {
        'Metodo 1 (Estatistico - Balanceado)': pred_m1,
        'Metodo 1 (Sub-estrategia Mais Atrasadas)': pred_m1_overdue,
        'Metodo 1 (Sub-estrategia Mais Frequentes Recentes)': pred_m1_momentum,
        'Metodo 2 (Machine Learning Multi-label Calibrado)': pred_m2,
        'Metodo 3 (Bayesiano Conjugado Global)': pred_m3_static,
        'Metodo 3 (Bayesiano Dinamico Adaptativo)': pred_m3_dyn,
        'Metodo 4 (Gianella - Geometria do Acaso)': pred_m4
    }

    if include_hybrids:
        predictions['Decisao Otima de Bayes (Consenso)'] = pred_consenso
        predictions['Metodo 5 (Gianella + Bayes Global)'] = pred_g_bayes_global
        predictions['Metodo 6 (Gianella + Bayes Adaptativo)'] = pred_g_bayes_adaptive
        predictions['Metodo 7 (Gianella + Melhor Decisao de Bayes)'] = pred_g_best_bayes
        predictions['Metodo 8 (Gianella + Regressao Logistica)'] = pred_g_lr
        predictions['Metodo 9 (Gianella + Freq/Atraso Momentum)'] = pred_g_mom
        predictions['Metodo 10 (Gianella + Freq/Atraso Balanceado)'] = pred_g_bal
        predictions['Metodo 11 (Melhor Decisao Bayes + Regressao Logistica)'] = pred_bb_lr
        predictions['Metodo 12 (Melhor Decisao Bayes + Freq/Atraso Momentum)'] = pred_bb_mom
        predictions['Metodo 13 (Melhor Decisao Bayes + Freq/Atraso Balanceado)'] = pred_bb_bal
        predictions['Metodo 14 (Bayes Global + Regressao Logistica)'] = pred_bg_lr
        predictions['Metodo 15 (Bayes Global + Frequencia/Atraso Momentum)'] = pred_bg_mom
        predictions['Metodo 16 (Bayes Global + Frequencia/Atraso Balanceado)'] = pred_bg_bal
        predictions['Metodo 17 (Bayes Adaptativo + Regressao Logistica)'] = pred_ba_lr
        predictions['Metodo 18 (Bayes Adaptativo + Frequencia/Atraso Momentum)'] = pred_ba_mom
        predictions['Metodo 19 (Bayes Adaptativo + Frequencia/Atraso Balanceado)'] = pred_ba_bal

    if include_ensembles:
        predictions['Metodo 20 (Ensemble Bagging - Bootstrap Aggregating)'] = pred_bagging
        predictions['Metodo 21 (Ensemble Pasting - Non-replacement Aggregating)'] = pred_pasting
        predictions['Metodo 22 (Ensemble AdaBoost - Adaptive Boosting)'] = pred_adaboost
        predictions['Metodo 23 (Ensemble Gradient Boosting - Residual Boosting)'] = pred_gradboost
        predictions['Metodo 24 (Ensemble Stacking - Stacked Generalization)'] = pred_stacking

    comparison_records = []
    for name, nums in predictions.items():
        comparison_records.append({
            'Metodo': name,
            'Proximo Concurso': next_concurso,
            'Dezenas Previstas': ', '.join(f'{n:02d}' for n in nums),
            'Lista Dezenas': nums
        })
    comparison_df = pd.DataFrame(comparison_records)

    return {
        'df_raw': df,
        'binary_matrix': binary_matrix,
        'cfg': cfg,
        'm1_model': m1,
        'm2_model': m2,
        'm3_static': m3_static,
        'm3_dyn': m3_dyn,
        'm4_gianella': m4,
        'm_hybrid': m_hybrid,
        'm_bagging': m_bagging,
        'm_pasting': m_pasting,
        'm_adaboost': m_adaboost,
        'm_gradboost': m_gradboost,
        'm_stacking': m_stacking,
        'pred_consenso': pred_consenso,
        'consensus_scores': ball_scores,
        'predictions': predictions,
        'predictions_hybrid': predictions_hybrid,
        'predictions_ensemble': predictions_ensemble,
        'comparison_df': comparison_df
    }


if __name__ == '__main__':
    excel_path = find_latest_lottery_file('megasena')
    print(f"Executando estimativas para a Mega-Sena a partir de '{excel_path}'...\n")
    results = run_all_methods(excel_path, include_hybrids=True)
    print("=" * 80)
    print(f"RESUMO DAS PREVISÕES PARA O CONCURSO {results['comparison_df']['Proximo Concurso'].iloc[0]}:")
    print("=" * 80)
    for idx, row in results['comparison_df'].iterrows():
        print(f"[{row['Metodo']}]")
        print(f"-> Dezenas: {row['Dezenas Previstas']}\n")
