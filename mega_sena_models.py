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
import glob
import re
from typing import Dict, List, Optional, Tuple, Union
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
# 4. FUNÇÃO COMPARATIVA E RELATÓRIO EXECUTIVO
# ==============================================================================

def run_all_methods(filepath: str, next_concurso: Optional[int] = None) -> Dict[str, object]:
    """
    Executa os 4 métodos sobre a base histórica e consolida os resultados
    para o próximo concurso especificado (ou inferido automaticamente).
    """
    return run_all_lottery_methods(filepath, game='megasena', next_concurso=next_concurso)


def run_all_lottery_methods(filepath: str,
                            game: str = 'auto',
                            next_concurso: Optional[int] = None) -> Dict:
    """
    Executa os 4 métodos analíticos para qualquer loteria (Mega-Sena, Lotofácil, Quina).
    """
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

    predictions = {
        'Metodo 1 (Estatistico - Balanceado)': pred_m1,
        'Metodo 1 (Sub-estrategia Mais Atrasadas)': pred_m1_overdue,
        'Metodo 1 (Sub-estrategia Mais Frequentes Recentes)': pred_m1_momentum,
        'Metodo 2 (Machine Learning Multi-label Calibrado)': pred_m2,
        'Metodo 3 (Bayesiano Conjugado Global)': pred_m3_static,
        'Metodo 3 (Bayesiano Dinamico Adaptativo)': pred_m3_dyn,
        'Metodo 4 (Gianella - Geometria do Acaso)': pred_m4
    }

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
        'predictions': predictions,
        'comparison_df': comparison_df
    }


if __name__ == '__main__':
    excel_path = find_latest_lottery_file('megasena')
    print(f"Executando estimativas para a Mega-Sena a partir de '{excel_path}'...\n")
    results = run_all_methods(excel_path)
    print("=" * 80)
    print(f"RESUMO DAS PREVISÕES PARA O CONCURSO {results['comparison_df']['Proximo Concurso'].iloc[0]}:")
    print("=" * 80)
    for idx, row in results['comparison_df'].iterrows():
        print(f"[{row['Metodo']}]")
        print(f"-> Dezenas: {row['Dezenas Previstas']}\n")
