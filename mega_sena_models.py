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

        if n_concursos > 1:
            self.cov_matrix_ = np.cov(binary_matrix, rowvar=False)
        else:
            self.cov_matrix_ = np.zeros((n_dezenas, n_dezenas))



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





class LotteryStochasticGradientBoostingModel:

    """

    Ensemble Stochastic Gradient Boosting para Seleção de Dezenas em Loterias.

    Baseado em Aurélien Géron (2019), Hands-On Machine Learning (2ª Ed.), Capítulo 7:

    'Stochastic Gradient Boosting' (Jerome H. Friedman, 2002, Computational Statistics & Data Analysis).



    Fundamentação Teórica e Algorítmica (Géron p. 207):

    1. Amostragem Aleatória de Subconjuntos (Subsample):

       A cada iteração de boosting m, uma fração aleatória 'subsample' (ex.: 0.7 ou 70%) dos concursos

       históricos é sorteada sem reposição. O preditor residual h_m é ajustado exclusivamente sobre essa

       amostra estocástica de concursos.

    2. Redução Drástica de Variância e Combate ao Sobreajuste:

       A injeção de estocasticidade desacopla os preditores residuais sequenciais, mitigando a correlação

       entre erros sucessivos e proporcionando um efeito de regularização similar ao Bagging dentro do Boosting.

    3. Atualização Regularizada por Encolhimento (Shrinkage, Géron p. 205):

       F_m = F_{m-1} + learning_rate * (h_norm - 0.5)

    4. Predição Estagiada (Staged Predict, Géron p. 206):

       Permite inspecionar a convergência das estimativas concurso a concurso.

    """



    def __init__(self,

                 n_estimators: int = 10,

                 learning_rate: float = 0.5,

                 subsample: float = 0.7,

                 random_state: int = 42,

                 base_models: Optional[List[str]] = None):

        """

        Parameters

        ----------

        n_estimators : int

            Número de estágios de boosting (default 10).

        learning_rate : float

            Taxa de aprendizado / encolhimento eta (default 0.5).

        subsample : float

            Fração de concursos históricos amostrados a cada estágio (0 < subsample <= 1.0, default 0.7).

        random_state : int

            Semente para amostragem estocástica reprodutível.

        base_models : Optional[List[str]]

            Estratégias de resíduo a utilizar.

        """

        if n_estimators <= 0:

            raise ValueError(f"n_estimators deve ser estritamente positivo (> 0). Recebido: {n_estimators}")

        if learning_rate <= 0:

            raise ValueError(f"learning_rate deve ser estritamente positivo (> 0). Recebido: {learning_rate}")

        if not (0.0 < subsample <= 1.0):

            raise ValueError(f"subsample deve estar no intervalo (0.0, 1.0]. Recebido: {subsample}")



        self.n_estimators = n_estimators

        self.learning_rate = learning_rate

        self.subsample = subsample

        self.random_state = random_state

        self.base_models = base_models



        self.staged_scores_: List[np.ndarray] = []

        self.scores_df_: Optional[pd.DataFrame] = None

        self.top_k_: Optional[List[int]] = None



    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None) -> 'LotteryStochasticGradientBoostingModel':

        """

        Ajusta sequencialmente o Stochastic Gradient Boosting com amostragem aleatória a cada iteração.

        """

        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:

            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")



        cfg = cfg or {}

        n_concursos, n_dezenas = binary_matrix.shape

        k = cfg.get('k', 6)



        # 1. Estimativa inicial F_0 (Esperança a priori bayesiana suavizada)

        alpha_0 = cfg.get('alpha_0', 1.0)

        beta_0 = cfg.get('beta_0', 9.0)

        F_current = (binary_matrix.sum(axis=0) + alpha_0) / (n_concursos + alpha_0 + beta_0)

        F_current = _min_max_scale(F_current)



        # Resíduos iniciais completos

        residuals = binary_matrix.astype(float) - F_current[None, :]

        self.staged_scores_ = [F_current.copy()]



        rng = np.random.RandomState(self.random_state)

        n_sample = max(2, int(np.floor(self.subsample * n_concursos)))

        n_sample = min(n_sample, n_concursos)



        for m in range(self.n_estimators):

            # 2. Sorteio Estocástico dos concursos históricos (Géron p. 207; Friedman 2002)

            if n_sample < n_concursos:

                sub_indices = np.sort(rng.choice(n_concursos, size=n_sample, replace=False))

            else:

                sub_indices = np.arange(n_concursos)



            sub_residuals = residuals[sub_indices, :]



            # Decaimento temporal adaptativo no subconjunto amostrado

            half_life = 50.0 + (m * 20.0)

            decay = 0.5 ** (1.0 / half_life)

            time_weights = decay ** np.arange(len(sub_indices) - 1, -1, -1)

            time_weights /= np.sum(time_weights)



            # Gradiente residual estocástico ponderado

            h_m = np.sum(sub_residuals * time_weights[:, None], axis=0)

            h_norm = _min_max_scale(h_m)



            # 3. Atualização do ensemble com regularização por encolhimento (Shrinkage)

            F_current = F_current + self.learning_rate * (h_norm - 0.5)

            F_current = np.clip(F_current, 0.0, None)



            # 4. Atualização dos resíduos em toda a base histórica

            residuals = residuals - self.learning_rate * (h_norm[None, :] - 0.5)

            self.staged_scores_.append(_min_max_scale(F_current).copy())



        final_scores = _min_max_scale(F_current)

        dezenas = np.arange(1, n_dezenas + 1)

        self.scores_df_ = pd.DataFrame({

            'dezena': dezenas,

            'escore_stoch_gradboost': np.round(final_scores, 5),

            'escore_acumulado': np.round(F_current, 5)

        }).sort_values('escore_stoch_gradboost', ascending=False).reset_index(drop=True)

        self.top_k_ = sorted(self.scores_df_.head(k)['dezena'].tolist())

        return self



    def predict_top_k(self, k: Optional[int] = None) -> List[int]:

        """

        Retorna as top K dezenas selecionadas pelo Stochastic Gradient Boosting.

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

        """Retorna o DataFrame detalhado com escores do Stochastic Gradient Boosting."""

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")

        return self.scores_df_.copy()





class LotteryLogitGradientBoostingModel:

    """

    Ensemble Log-Loss / Binomial Deviance Gradient Boosting para Seleção de Dezenas em Loterias.

    Baseado em Aurélien Géron (2019), Hands-On Machine Learning (2ª Ed.), Capítulo 7 ('Gradient Boosting')

    e Jerome H. Friedman (2001), 'Greedy Function Approximation: A Gradient Boosting Machine'

    (Section 4.2: Binomial Deviance / Two-Class Classification).



    Fundamentação Matemática:

    1. Espaço de Log-Odds (Logits) e Função Sigmoide:

       A previsão do modelo reside no espaço não-delimitado dos log-odds F(d) in (-inf, +inf).

       A probabilidade calibrada de ocorrência da dezena d é dada pela função sigmoide:

       p(d) = sigma(F(d)) = 1 / (1 + exp(-F(d)))

    2. Função de Perda de Entropia Cruzada Binária (Binomial Deviance Loss):

       L(y, F) = - [ y * ln(sigma(F)) + (1 - y) * ln(1 - sigma(F)) ] = ln(1 + exp(F)) - y * F

    3. Resíduo de Gradiente Negativo (Pseudo-Resíduo):

       - dL / dF = y - sigma(F) = y - p

       O gradiente negativo aponta exatamente na direção do erro de probabilidade (y - p).

    4. Atualização de Newton-Raphson com Regularização por Encolhimento (Shrinkage):

       gamma_{m}(d) = sum_t w_t * (y_{t, d} - p_{t, d}) / (sum_t w_t * p_{t, d} * (1 - p_{t, d}) + epsilon)

       F_m(d) = F_{m-1}(d) + learning_rate * gamma_{step}

    """



    def __init__(self,

                 n_estimators: int = 10,

                 learning_rate: float = 0.3,

                 random_state: int = 42,

                 base_models: Optional[List[str]] = None):

        """

        Parameters

        ----------

        n_estimators : int

            Número de iterações de boosting no espaço de logit (default 10).

        learning_rate : float

            Taxa de aprendizado / encolhimento no espaço log-odds (default 0.3).

        random_state : int

            Semente para reprodutibilidade.

        base_models : Optional[List[str]]

            Lista de modelos auxiliares (opcional).

        """

        if n_estimators <= 0:

            raise ValueError(f"n_estimators deve ser estritamente positivo (> 0). Recebido: {n_estimators}")

        if learning_rate <= 0:

            raise ValueError(f"learning_rate deve ser estritamente positivo (> 0). Recebido: {learning_rate}")



        self.n_estimators = n_estimators

        self.learning_rate = learning_rate

        self.random_state = random_state

        self.base_models = base_models



        self.staged_scores_: List[np.ndarray] = []

        self.scores_df_: Optional[pd.DataFrame] = None

        self.top_k_: Optional[List[int]] = None



    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None) -> 'LotteryLogitGradientBoostingModel':

        """

        Ajusta o Gradient Boosting de Deviância Binomial no espaço de logits.

        """

        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:

            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")



        cfg = cfg or {}

        n_concursos, n_dezenas = binary_matrix.shape

        k = cfg.get('k', 6)



        # 1. Inicialização F_0(d) = log-odds da priori suavizada (Friedman 2001, Eq. 25)

        alpha_0 = cfg.get('alpha_0', 1.0)

        beta_0 = cfg.get('beta_0', 9.0)

        p_0 = (binary_matrix.sum(axis=0) + alpha_0) / (n_concursos + alpha_0 + beta_0)

        p_0 = np.clip(p_0, 1e-4, 1.0 - 1e-4)

        F_current = np.log(p_0 / (1.0 - p_0))

        prob_current = 1.0 / (1.0 + np.exp(-np.clip(F_current, -20.0, 20.0)))

        self.staged_scores_ = [_min_max_scale(prob_current).copy()]



        for m in range(self.n_estimators):

            # Pesos temporais adaptativos (prioriza concursos mais recentes)

            half_life = 50.0 + (m * 20.0)

            decay = 0.5 ** (1.0 / half_life)

            time_weights = decay ** np.arange(n_concursos - 1, -1, -1)

            time_weights /= np.sum(time_weights)



            # Probabilidades atuais para cada concurso e dezena

            p_mat = 1.0 / (1.0 + np.exp(-np.clip(F_current[None, :], -20.0, 20.0)))



            # Gradiente negativo da Binomial Deviance (y - p)

            dev_residuals = binary_matrix.astype(float) - p_mat



            # 2. Passo ótimo de Newton-Raphson por dezena (Friedman 2001 Eq. 27)

            grad_w = np.sum(dev_residuals * time_weights[:, None], axis=0)

            hessian_w = np.sum((p_mat * (1.0 - p_mat)) * time_weights[:, None], axis=0) + 1e-5

            gamma_m = grad_w / hessian_w



            # Normalização de escala para estabilidade numérica

            gamma_std = np.std(gamma_m)

            if gamma_std > 1e-8:

                gamma_step = (gamma_m - np.mean(gamma_m)) / gamma_std

            else:

                gamma_step = gamma_m - np.mean(gamma_m)



            # 3. Atualização no espaço logit com regularização por encolhimento (Shrinkage)

            F_current = F_current + self.learning_rate * gamma_step

            F_current = np.clip(F_current, -20.0, 20.0)



            # Probabilidade calibrada atualizada

            prob_updated = 1.0 / (1.0 + np.exp(-F_current))

            self.staged_scores_.append(_min_max_scale(prob_updated).copy())



        final_probs = 1.0 / (1.0 + np.exp(-F_current))

        final_scores = _min_max_scale(final_probs)

        dezenas = np.arange(1, n_dezenas + 1)

        self.scores_df_ = pd.DataFrame({

            'dezena': dezenas,

            'probabilidade_calibrada': np.round(final_probs, 5),

            'logit_escore': np.round(F_current, 5),

            'escore_logit_gradboost': np.round(final_scores, 5)

        }).sort_values('escore_logit_gradboost', ascending=False).reset_index(drop=True)

        self.top_k_ = sorted(self.scores_df_.head(k)['dezena'].tolist())

        return self



    def predict_top_k(self, k: Optional[int] = None) -> List[int]:

        """

        Retorna as top K dezenas selecionadas pelo Logit Gradient Boosting.

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

        Iterador sobre as predições do ensemble em cada estágio de treinamento no espaço de probabilidade.

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

        """Retorna o DataFrame detalhado com escores e probabilidades do Logit Gradient Boosting."""

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")

        return self.scores_df_.copy()





# Alias conveniente e formal

# Alias conveniente e formal

LotteryBinomialDevianceGradientBoostingModel = LotteryLogitGradientBoostingModel





class LotteryBaggedLogitGBEnsembleModel:

    """

    Meta-Ensemble Bagged Logit Gradient Boosting (Ensemble do Ensemble Logit GB).

    Baseado em Leo Breiman (1996), 'Bagging Predictors' e Jerome H. Friedman (2001, 2002),

    'Stochastic Gradient Boosting'.



    Fundamentação Matemática e Redução de Variância:

    O modelo base LotteryLogitGradientBoostingModel é eficiente na redução de viés através

    da minimização da Binomial Deviance no espaço de logits. No entanto, em sistemas estocásticos

    como loterias, estimadores sequenciais isolados podem sofrer de variância estocástica

    não-desprezível.



    Este Meta-Ensemble treina um comitê de M estimadores Logit Gradient Boosting independentes,

    cada um submetido a:

    1. Bootstrap Subsampling Temporal: Amostragem estocástica com reposição de concursos.

    2. Perturbação Estocástica de Taxa de Aprendizado: Variabilidade em eta ~ Uniform(0.20, 0.45).

    3. Soft-Voting / Média de Probabilidades Calibradas:

       p_meta(d) = (1 / M) * sum_{m=1}^M p_m(d)

       Var(bar{p}) = rho * sigma^2 + ((1 - rho) / M) * sigma^2 -> Redução assintótica da variância!

    """



    def __init__(self,

                 n_estimators: int = 10,

                 base_n_iterations: int = 10,

                 subsample: float = 0.85,

                 learning_rate_range: Tuple[float, float] = (0.20, 0.45),

                 random_state: int = 42):

        if n_estimators <= 0:

            raise ValueError(f"n_estimators deve ser > 0. Recebido: {n_estimators}")

        self.n_estimators = n_estimators

        self.base_n_iterations = base_n_iterations

        self.subsample = subsample

        self.learning_rate_range = learning_rate_range

        self.random_state = random_state



        self.sub_models_: List[LotteryLogitGradientBoostingModel] = []

        self.scores_df_: Optional[pd.DataFrame] = None

        self.top_k_: Optional[List[int]] = None



    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None) -> 'LotteryBaggedLogitGBEnsembleModel':

        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:

            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")



        cfg = cfg or {}

        n_concursos, n_dezenas = binary_matrix.shape

        k = cfg.get('k', 6)

        rng = np.random.RandomState(self.random_state)

        sample_size = max(10, int(n_concursos * self.subsample))



        self.sub_models_ = []

        all_probs = []



        lr_min, lr_max = self.learning_rate_range



        for i in range(self.n_estimators):

            boot_idx = rng.choice(n_concursos, size=sample_size, replace=True)

            boot_idx = np.sort(boot_idx)

            sub_X = binary_matrix[boot_idx, :]



            sub_lr = float(rng.uniform(lr_min, lr_max))

            sub_seed = int(rng.randint(0, 100000))



            sub_model = LotteryLogitGradientBoostingModel(

                n_estimators=self.base_n_iterations,

                learning_rate=sub_lr,

                random_state=sub_seed

            )

            sub_model.fit(sub_X, cfg=cfg)

            self.sub_models_.append(sub_model)



            probs = sub_model.scores_df_.sort_values('dezena')['probabilidade_calibrada'].values

            all_probs.append(probs)



        avg_probs = np.mean(all_probs, axis=0)

        final_scores = _min_max_scale(avg_probs)

        dezenas = np.arange(1, n_dezenas + 1)



        self.scores_df_ = pd.DataFrame({

            'dezena': dezenas,

            'probabilidade_calibrada': np.round(avg_probs, 5),

            'escore_meta_logit_gb': np.round(final_scores, 5)

        }).sort_values('escore_meta_logit_gb', ascending=False).reset_index(drop=True)



        self.top_k_ = sorted(self.scores_df_.head(k)['dezena'].tolist())

        return self



    def predict_top_k(self, k: Optional[int] = None) -> List[int]:

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")

        if k is None:

            k = len(self.top_k_) if self.top_k_ is not None else 6

        if k <= 0:

            return []

        k = min(k, len(self.scores_df_))

        return sorted(self.scores_df_.head(k)['dezena'].tolist())



    def get_scores_df(self) -> pd.DataFrame:

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")

        return self.scores_df_.copy()





LotteryMetaLogitGradientBoostingModel = LotteryBaggedLogitGBEnsembleModel





class LotteryDualLogitGBEnsembleModel:

    """

    Meta-Ensemble Dual Logit Gradient Boosting (Método 34).

    Fusão Hierárquica do Ensemble Logit Gradient Boosting (Método 28) com o

    Meta-Ensemble Bagged Logit GB (Método 33).



    Fundamentação Matemática e Teoria da Decisão:

    1. Componente de Baixo Viés (Sinal Analítico Direto):

       O modelo LotteryLogitGradientBoostingModel (Friedman, 2001) atua como estimador âncora,

       calibrando o gradiente analítico exato da perda Binomial Deviance no espaço contínuo dos logits

       sobre a totalidade do histórico de sorteios.

    2. Componente de Baixa Variância (Filtro de Ruído Estocástico):

       O modelo LotteryBaggedLogitGBEnsembleModel (Breiman, 1996) atua como estabilizador de variância,

       treinando um comitê ótimo de M=10 estimadores sob amostragem bootstrap temporal com perturbações

       estocásticas na taxa de aprendizado eta ~ Uniform(0.20, 0.45).

    3. Fusão Convexa Bayesiana:

       Combina as evidências dos dois estimadores via ponderação convexa w in [0, 1]:

       p_dual(d) = w * p_base(d) + (1 - w) * p_bagged(d)

       onde o default w = 0.45 atribui 45% ao estimador analítico direto e 55% ao comitê suavizador

       de variância, proporcionando a máxima robustez contra sobreajustes e ruídos de curto prazo.

    """



    def __init__(self,

                 n_estimators_bagged: int = 10,

                 base_n_iterations: int = 10,

                 weight_base: float = 0.45,

                 learning_rate_base: float = 0.3,

                 subsample: float = 0.85,

                 random_state: int = 42):

        if n_estimators_bagged <= 0:

            raise ValueError(f"n_estimators_bagged deve ser > 0. Recebido: {n_estimators_bagged}")

        if not (0.0 <= weight_base <= 1.0):

            raise ValueError(f"weight_base deve estar entre 0.0 e 1.0. Recebido: {weight_base}")

        self.n_estimators_bagged = n_estimators_bagged

        self.base_n_iterations = base_n_iterations

        self.weight_base = weight_base

        self.learning_rate_base = learning_rate_base

        self.subsample = subsample

        self.random_state = random_state



        self.model_base_: Optional[LotteryLogitGradientBoostingModel] = None

        self.model_bagged_: Optional[LotteryBaggedLogitGBEnsembleModel] = None

        self.scores_df_: Optional[pd.DataFrame] = None

        self.top_k_: Optional[List[int]] = None



    def fit(self,

            binary_matrix: np.ndarray,

            cfg: Optional[Dict] = None,

            base_model: Optional[LotteryLogitGradientBoostingModel] = None,

            bagged_model: Optional[LotteryBaggedLogitGBEnsembleModel] = None) -> 'LotteryDualLogitGBEnsembleModel':

        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:

            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")



        cfg = cfg or {}

        n_concursos, n_dezenas = binary_matrix.shape

        k = cfg.get('k', 6)



        # 1. Componente 1: Logit GB Padrão (ou reutilizar se fornecido)

        if base_model is not None and base_model.scores_df_ is not None:

            self.model_base_ = base_model

        else:

            self.model_base_ = LotteryLogitGradientBoostingModel(

                n_estimators=self.base_n_iterations,

                learning_rate=self.learning_rate_base,

                random_state=self.random_state

            )

            self.model_base_.fit(binary_matrix, cfg=cfg)



        probs_base = self.model_base_.scores_df_.sort_values('dezena')['probabilidade_calibrada'].values



        # 2. Componente 2: Bagged Logit GB (ou reutilizar se fornecido)

        if bagged_model is not None and bagged_model.scores_df_ is not None:

            self.model_bagged_ = bagged_model

        else:

            self.model_bagged_ = LotteryBaggedLogitGBEnsembleModel(

                n_estimators=self.n_estimators_bagged,

                base_n_iterations=self.base_n_iterations,

                subsample=self.subsample,

                random_state=self.random_state

            )

            self.model_bagged_.fit(binary_matrix, cfg=cfg)



        probs_bagged = self.model_bagged_.scores_df_.sort_values('dezena')['probabilidade_calibrada'].values



        # 3. Fusão Convexa Bayesiana das Probabilidades

        w = float(self.weight_base)

        fused_probs = w * probs_base + (1.0 - w) * probs_bagged

        final_scores = _min_max_scale(fused_probs)

        dezenas = np.arange(1, n_dezenas + 1)



        self.scores_df_ = pd.DataFrame({

            'dezena': dezenas,

            'probabilidade_calibrada': np.round(fused_probs, 5),

            'probabilidade_base_logit': np.round(probs_base, 5),

            'probabilidade_bagged_logit': np.round(probs_bagged, 5),

            'escore_dual_logit_gb': np.round(final_scores, 5)

        }).sort_values('escore_dual_logit_gb', ascending=False).reset_index(drop=True)



        self.top_k_ = sorted(self.scores_df_.head(k)['dezena'].tolist())

        return self



    def predict_top_k(self, k: Optional[int] = None) -> List[int]:

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")

        if k is None:

            k = len(self.top_k_) if self.top_k_ is not None else 6

        if k <= 0:

            return []

        k = min(k, len(self.scores_df_))

        return sorted(self.scores_df_.head(k)['dezena'].tolist())



    def get_scores_df(self) -> pd.DataFrame:

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")

        return self.scores_df_.copy()





LotteryHierarchicalLogitGBEnsembleModel = LotteryDualLogitGBEnsembleModel





def _extract_trained_ensemble_signals(trained_ensembles: Dict[str, Any],
                                      n_dezenas: int,
                                      include_sub_estimators: bool = True) -> Tuple[List[str], np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Extrai os vetores de escores normalizados em [0, 1] e priors Out-of-Bag (OOB)
    a partir de um dicionário de ensembles já treinados.

    Retorna:
      - main_names: nomes dos ensembles principais válidos
      - main_matrix: matriz shape (K_main, n_dezenas)
      - main_oob_priors: vetor shape (K_main,) com bônus OOB não-negativos (Zero Lookahead)
      - sub_names: nomes dos sub-estimadores internos extraídos
      - sub_matrix: matriz shape (K_sub, n_dezenas) (ou shape (0, n_dezenas))
    """
    preferred_score_cols = [
        'probabilidade_calibrada',
        'escore_dual_logit_gb',
        'escore_meta_logit_gb',
        'escore_bagging',
        'escore_pasting',
        'escore_adaboost',
        'escore_gradboost',
        'escore_stoch_gradboost',
        'escore_logit_gradboost',
        'escore_cycling_gradboost',
        'escore_stacking',
        'escore_momentum_ensemble',
        'escore_balanceado_ensemble',
        'escore_batch_gd',
        'escore_minibatch_gd',
        'escore_stochastic_gd',
        'escore_meta_comite',
        'escore_comite_profundo',
        'escore_gianella_comite',
    ]

    def _df_to_score_vec(df_scores: pd.DataFrame) -> Optional[np.ndarray]:
        if df_scores is None or len(df_scores) == 0 or 'dezena' not in df_scores.columns:
            return None
        col_sel = None
        for c in preferred_score_cols:
            if c in df_scores.columns:
                col_sel = c
                break
        if col_sel is None:
            escore_cols = [c for c in df_scores.columns if str(c).startswith('escore_') or str(c).startswith('prob')]
            if not escore_cols:
                return None
            col_sel = escore_cols[0]

        vec = np.zeros(n_dezenas, dtype=float)
        for _, row in df_scores.iterrows():
            d_idx = int(row['dezena']) - 1
            if 0 <= d_idx < n_dezenas:
                val = float(row[col_sel])
                vec[d_idx] = 0.0 if not np.isfinite(val) else val
        return _min_max_scale(vec)

    main_names: List[str] = []
    main_vecs: List[np.ndarray] = []
    main_oobs: List[Optional[float]] = []
    sub_names: List[str] = []
    sub_vecs: List[np.ndarray] = []

    for ens_name, obj in trained_ensembles.items():
        if obj is None:
            continue
        vec_main: Optional[np.ndarray] = None
        oob_val: Optional[float] = None

        if isinstance(obj, np.ndarray) and obj.ndim == 1 and len(obj) == n_dezenas:
            vec_main = _min_max_scale(np.nan_to_num(obj.astype(float), nan=0.0, posinf=1.0, neginf=0.0))
        elif isinstance(obj, dict) and 'scores' in obj:
            arr = np.asarray(obj['scores'], dtype=float)
            if arr.ndim == 1 and len(arr) == n_dezenas:
                vec_main = _min_max_scale(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0))
            raw_oob = obj.get('oob_score')
            if isinstance(raw_oob, (int, float, np.floating, np.integer)) and np.isfinite(raw_oob) and float(raw_oob) > 0:
                oob_val = float(raw_oob)
        elif hasattr(obj, 'scores_df_') and obj.scores_df_ is not None:
            vec_main = _df_to_score_vec(obj.scores_df_)
            raw_oob = getattr(obj, 'oob_score_', None)
            if isinstance(raw_oob, (int, float, np.floating, np.integer)) and np.isfinite(raw_oob) and float(raw_oob) > 0:
                oob_val = float(raw_oob)

        if vec_main is None:
            continue

        main_names.append(str(ens_name))
        main_vecs.append(vec_main)
        main_oobs.append(oob_val)

        if include_sub_estimators:
            est_list = getattr(obj, 'estimators_', None)
            if isinstance(est_list, list):
                for idx_s, est_item in enumerate(est_list):
                    if isinstance(est_item, dict) and 'scores' in est_item:
                        s_arr = np.asarray(est_item['scores'], dtype=float)
                        if s_arr.ndim == 1 and len(s_arr) == n_dezenas:
                            sub_names.append(f"{ens_name}::est_{idx_s}")
                            sub_vecs.append(_min_max_scale(np.nan_to_num(s_arr, nan=0.0)))

            sub_mods = getattr(obj, 'sub_models_', None)
            if isinstance(sub_mods, list):
                for idx_s, sub_m in enumerate(sub_mods):
                    sub_df = getattr(sub_m, 'scores_df_', None)
                    if sub_df is not None:
                        sv = _df_to_score_vec(sub_df)
                        if sv is not None:
                            sub_names.append(f"{ens_name}::sub_{idx_s}")
                            sub_vecs.append(sv)

            staged_list = getattr(obj, 'staged_scores_', None)
            if isinstance(staged_list, list) and len(staged_list) > 1:
                tail_stages = staged_list[1:][-4:]
                for idx_s, st_arr in enumerate(tail_stages):
                    s_arr = np.asarray(st_arr, dtype=float)
                    if s_arr.ndim == 1 and len(s_arr) == n_dezenas:
                        sub_names.append(f"{ens_name}::stage_{idx_s}")
                        sub_vecs.append(_min_max_scale(np.nan_to_num(s_arr, nan=0.0)))

    if not main_vecs:
        return [], np.zeros((0, n_dezenas), dtype=float), np.zeros(0, dtype=float), [], np.zeros((0, n_dezenas), dtype=float)

    main_mat = np.vstack(main_vecs)
    valid_oobs = [v for v in main_oobs if v is not None]
    if valid_oobs:
        mean_obs = float(np.mean(valid_oobs))
        filled_oobs = np.asarray([v if v is not None else mean_obs for v in main_oobs], dtype=float)
        rel_dev = (filled_oobs - mean_obs) / max(mean_obs, 1e-8)
        oob_norm = np.clip(0.5 + 2.5 * rel_dev, 0.0, 1.0)
    else:
        oob_norm = np.zeros(len(main_vecs), dtype=float)

    sub_mat = np.vstack(sub_vecs) if sub_vecs else np.zeros((0, n_dezenas), dtype=float)
    return main_names, main_mat, oob_norm, sub_names, sub_mat


class LotteryPretrainedMetaCommitteeModel:
    """
    Meta-Ensemble em Comitê Estocástico de Ensembles Pré-Treinados (Método 35, padrão M=100).
    Constrói um comitê de M membros (padrão M=100, suportando também M=50 ou qualquer M>0)
    diretamente sobre os ensembles já treinados (e seus sub-estimadores internos), combinando:
    1. Random Subspaces (Ho, 1998; Géron Cap. 7, pp. 196-197): cada membro do comitê
       amostra sem reposição um subconjunto distinto dos ensembles pré-treinados e de
       seus estimadores internos, reduzindo correlação inter-membros.
    2. Ponderação Bayesiana Dirichlet com Prior OOB (Zero Lookahead Bias): sorteia pesos
       simplex via Dirichlet(alpha_prior), atribuindo leve reforço aos estimadores que
       possuem validação Out-of-Bag histórica positiva.
    3. Calibração Termodinâmica (Softmax Temperature) + Fusão Híbrida Soft/Borda/Hard:
       cada membro gera escore contínuo, probabilidade calibrada com temperatura tau_m,
       rank Borda e cédula discreta top-k.
    """

    def __init__(self,
                 n_committees: int = 100,
                 subspace_ratio: float = 0.65,
                 include_sub_estimators: bool = True,
                 dirichlet_alpha: float = 1.5,
                 voting: str = 'hybrid',
                 temperature_range: Tuple[float, float] = (0.75, 1.50),
                 random_state: int = 42):
        if n_committees <= 0:
            raise ValueError(f"n_committees deve ser estritamente positivo (> 0). Recebido: {n_committees}")
        if not (0.0 < subspace_ratio <= 1.0):
            raise ValueError(f"subspace_ratio deve estar no intervalo (0.0, 1.0]. Recebido: {subspace_ratio}")
        if dirichlet_alpha <= 0.0:
            raise ValueError(f"dirichlet_alpha deve ser > 0. Recebido: {dirichlet_alpha}")
        if voting not in ('soft', 'hard', 'borda', 'hybrid'):
            raise ValueError(f"voting deve ser 'soft', 'hard', 'borda' ou 'hybrid'. Recebido: {voting}")
        if len(temperature_range) != 2 or temperature_range[0] <= 0.0 or temperature_range[0] > temperature_range[1]:
            raise ValueError(f"temperature_range inválido: {temperature_range}")

        self.n_committees = int(n_committees)
        self.subspace_ratio = float(subspace_ratio)
        self.include_sub_estimators = bool(include_sub_estimators)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.voting = voting
        self.temperature_range = temperature_range
        self.random_state = int(random_state)

        self.trained_ensembles_: Dict[str, Any] = {}
        self.base_ensemble_names_: List[str] = []
        self.sub_estimator_count_: int = 0
        self.ensemble_contributions_: Dict[str, float] = {}
        self.committee_members_: List[Dict[str, Any]] = []
        self.convergence_curve_: List[Dict[str, float]] = []
        self.committee_std_: Optional[np.ndarray] = None
        self.mean_se_: float = 0.0
        self.scores_df_: Optional[pd.DataFrame] = None
        self.top_k_: Optional[List[int]] = None

    def _fit_default_base_ensembles(self, binary_matrix: np.ndarray, cfg: Dict) -> Dict[str, Any]:
        n_concursos = binary_matrix.shape[0]
        base_bank: Dict[str, Any] = {}

        m_bag = LotteryBaggingModel(n_estimators=10, random_state=self.random_state).fit(binary_matrix, cfg=cfg)
        m_paste = LotteryPastingModel(n_estimators=10, max_samples=0.8, random_state=self.random_state).fit(binary_matrix, cfg=cfg)
        m_ada = LotteryAdaBoostModel(n_estimators=10, learning_rate=1.0, random_state=self.random_state).fit(binary_matrix, cfg=cfg)
        m_gb = LotteryGradientBoostingModel(n_estimators=10, learning_rate=0.5, random_state=self.random_state).fit(binary_matrix, cfg=cfg)
        m_stoch_gb = LotteryStochasticGradientBoostingModel(n_estimators=10, learning_rate=0.5, subsample=0.7, random_state=self.random_state).fit(binary_matrix, cfg=cfg)
        m_logit_gb = LotteryLogitGradientBoostingModel(n_estimators=10, learning_rate=0.3, random_state=self.random_state).fit(binary_matrix, cfg=cfg)
        m_cycle_gb = LotteryResidualCyclingGradientBoostingModel(n_estimators=8, learning_rate=0.4, random_state=self.random_state).fit(binary_matrix, cfg=cfg)
        m_mom = LotteryMomentumEnsembleModel(n_estimators=10, random_state=self.random_state).fit(binary_matrix, cfg=cfg)
        m_bal = LotteryBalancedEnsembleModel(n_estimators=10, random_state=self.random_state).fit(binary_matrix, cfg=cfg)
        m_bagged_lgb = LotteryBaggedLogitGBEnsembleModel(n_estimators=10, random_state=self.random_state).fit(binary_matrix, cfg=cfg)
        m_dual_lgb = LotteryDualLogitGBEnsembleModel(
            n_estimators_bagged=10, weight_base=0.45, random_state=self.random_state
        ).fit(binary_matrix, cfg=cfg, base_model=m_logit_gb, bagged_model=m_bagged_lgb)

        base_bank.update({
            'bagging': m_bag,
            'pasting': m_paste,
            'adaboost': m_ada,
            'gradboost': m_gb,
            'stoch_gb': m_stoch_gb,
            'logit_gb': m_logit_gb,
            'cycle_gb': m_cycle_gb,
            'mom_ens': m_mom,
            'bal_ens': m_bal,
            'bagged_logit_gb': m_bagged_lgb,
            'dual_logit_gb': m_dual_lgb,
        })

        if n_concursos >= 15:
            for key, cls_fn in [
                ('stacking', lambda: LotteryStackingModel(holdout_ratio=0.25, random_state=self.random_state).fit(binary_matrix, cfg=cfg)),
                ('batch_gd', lambda: LotteryBatchGDEnsembleModel(n_epochs=80, learning_rate=0.1, random_state=self.random_state).fit(binary_matrix, cfg=cfg)),
                ('minibatch_gd', lambda: LotteryMiniBatchGDEnsembleModel(n_epochs=30, batch_size=32, learning_rate=0.05, random_state=self.random_state).fit(binary_matrix, cfg=cfg)),
                ('stoch_gd', lambda: LotteryStochasticGDEnsembleModel(n_epochs=15, learning_rate_0=0.05, decay=0.005, random_state=self.random_state).fit(binary_matrix, cfg=cfg)),
            ]:
                try:
                    base_bank[key] = cls_fn()
                except Exception:
                    pass

        return base_bank

    def fit(self,
            binary_matrix: np.ndarray,
            cfg: Optional[Dict] = None,
            trained_ensembles: Optional[Dict[str, Any]] = None) -> 'LotteryPretrainedMetaCommitteeModel':
        """
        Estima o Meta-Comitê de M membros baseado nos ensembles já treinados.
        """
        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        cfg = cfg or {}
        _, n_dezenas = binary_matrix.shape
        k = int(cfg.get('k', 6))

        if trained_ensembles is not None:
            if not isinstance(trained_ensembles, dict) or len(trained_ensembles) == 0:
                raise ValueError("trained_ensembles deve ser um dicionário não-vazio com ensembles ajustados.")
            self.trained_ensembles_ = dict(trained_ensembles)
        else:
            self.trained_ensembles_ = self._fit_default_base_ensembles(binary_matrix, cfg)

        main_names, main_mat, oob_norm, sub_names, sub_mat = _extract_trained_ensemble_signals(
            self.trained_ensembles_, n_dezenas, include_sub_estimators=self.include_sub_estimators
        )
        if len(main_names) == 0:
            raise ValueError("trained_ensembles não contém nenhum ensemble ajustado válido.")

        self.base_ensemble_names_ = main_names
        self.sub_estimator_count_ = len(sub_names)

        k_main = len(main_names)
        k_sub = len(sub_names)
        q_main = max(1, min(k_main, int(np.ceil(self.subspace_ratio * k_main))))
        if k_main >= 3:
            q_main = max(2, q_main)
        q_sub = max(1, min(k_sub, int(np.ceil(self.subspace_ratio * k_sub)))) if k_sub > 0 else 0

        alpha_prior = self.dirichlet_alpha * (1.0 + 0.35 * oob_norm)
        rng = np.random.RandomState(self.random_state)
        t_min, t_max = self.temperature_range

        all_soft = np.zeros((self.n_committees, n_dezenas), dtype=float)
        all_probs = np.zeros((self.n_committees, n_dezenas), dtype=float)
        all_borda = np.zeros((self.n_committees, n_dezenas), dtype=float)
        hard_votes = np.zeros(n_dezenas, dtype=float)
        accum_weights = np.zeros(k_main, dtype=float)

        self.committee_members_ = []
        self.convergence_curve_ = []

        running_soft = np.zeros(n_dezenas, dtype=float)
        prev_top_set: Optional[set] = None

        for m in range(self.n_committees):
            idx_main = rng.choice(k_main, size=q_main, replace=False)
            w_main = rng.dirichlet(alpha_prior[idx_main])
            accum_weights[idx_main] += w_main
            s_main = w_main @ main_mat[idx_main]

            if q_sub > 0:
                idx_sub = rng.choice(k_sub, size=q_sub, replace=False)
                w_sub = rng.dirichlet(np.full(q_sub, self.dirichlet_alpha))
                s_sub = w_sub @ sub_mat[idx_sub]
                beta_sub = float(rng.uniform(0.20, 0.45))
                s_raw = (1.0 - beta_sub) * _min_max_scale(s_main) + beta_sub * _min_max_scale(s_sub)
            else:
                s_raw = _min_max_scale(s_main)

            s_norm = _min_max_scale(s_raw)
            temp_m = float(rng.uniform(t_min, t_max))
            std_s = float(np.std(s_norm))
            z_s = (s_norm - float(np.mean(s_norm))) / (std_s if std_s > 1e-12 else 1.0)
            logits = np.clip(z_s / temp_m, -20.0, 20.0)
            exp_l = np.exp(logits - np.max(logits))
            p_m = exp_l / np.sum(exp_l)

            order = np.argsort(s_norm, kind='mergesort')[::-1]
            borda_m = np.zeros(n_dezenas, dtype=float)
            denom = float(max(1, n_dezenas - 1))
            for r_idx, d_idx in enumerate(order):
                borda_m[d_idx] = 1.0 - (r_idx / denom)

            k_ballot = min(max(1, k), n_dezenas)
            top_m = sorted((order[:k_ballot] + 1).tolist())
            for dez in top_m:
                hard_votes[dez - 1] += 1.0

            all_soft[m] = s_norm
            all_probs[m] = p_m
            all_borda[m] = borda_m

            self.committee_members_.append({
                'member_idx': m,
                'n_main_selected': q_main,
                'n_sub_selected': q_sub,
                'temperature': round(temp_m, 4),
                'top_k': top_m,
            })

            running_soft += s_norm
            cur_mean = running_soft / float(m + 1)
            cur_top_set = set((np.argsort(cur_mean, kind='mergesort')[::-1][:k_ballot] + 1).tolist())
            if prev_top_set is not None:
                union_sz = len(cur_top_set | prev_top_set)
                jaccard = len(cur_top_set & prev_top_set) / float(union_sz) if union_sz > 0 else 1.0
            else:
                jaccard = 1.0
            prev_top_set = cur_top_set
            self.convergence_curve_.append({
                'committee_size': m + 1,
                'top_k_jaccard': round(float(jaccard), 4),
                'score_se': round(float(np.mean(np.std(all_soft[:m + 1], axis=0) / np.sqrt(m + 1))), 6)
            })

        soft_mean = np.mean(all_soft, axis=0)
        prob_mean = np.mean(all_probs, axis=0)
        borda_mean = np.mean(all_borda, axis=0)
        vote_rate = hard_votes / float(self.n_committees)

        self.committee_std_ = np.std(all_soft, axis=0)
        se_vec = self.committee_std_ / np.sqrt(float(self.n_committees))
        self.mean_se_ = float(np.mean(se_vec))
        self.ensemble_contributions_ = {
            main_names[i]: float(np.round(accum_weights[i] / float(self.n_committees), 5))
            for i in range(k_main)
        }

        hybrid_raw = (
            0.35 * _min_max_scale(soft_mean)
            + 0.25 * _min_max_scale(prob_mean)
            + 0.20 * _min_max_scale(borda_mean)
            + 0.20 * vote_rate
        )
        hybrid_norm = _min_max_scale(hybrid_raw)

        dezenas = np.arange(1, n_dezenas + 1)
        self.scores_df_ = pd.DataFrame({
            'dezena': dezenas,
            'escore_meta_comite': np.round(hybrid_norm, 5),
            'probabilidade_comite': np.round(prob_mean, 5),
            'escore_soft': np.round(soft_mean, 5),
            'escore_borda': np.round(borda_mean, 5),
            'votos_hard': hard_votes.astype(int),
            'taxa_consenso_pct': np.round(vote_rate * 100.0, 2),
            'incerteza_se': np.round(se_vec, 6),
        }).sort_values(['escore_meta_comite', 'escore_soft', 'dezena'], ascending=[False, False, True]).reset_index(drop=True)

        k_sel = min(max(1, k), n_dezenas) if k > 0 else 0
        self.top_k_ = self.predict_top_k(k_sel, voting=self.voting) if k_sel > 0 else []
        return self

    def predict_top_k(self, k: Optional[int] = None, voting: Optional[str] = None) -> List[int]:
        """Retorna as top K dezenas segundo o critério de votação do Meta-Comitê."""
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")
        if k is None:
            k = len(self.top_k_) if self.top_k_ is not None else 6
        if k <= 0:
            return []
        k = min(k, len(self.scores_df_))

        use_voting = voting or self.voting
        if use_voting == 'hard':
            df_sorted = self.scores_df_.sort_values(
                ['votos_hard', 'escore_meta_comite', 'dezena'], ascending=[False, False, True]
            )
        elif use_voting == 'soft':
            df_sorted = self.scores_df_.sort_values(
                ['escore_soft', 'escore_meta_comite', 'dezena'], ascending=[False, False, True]
            )
        elif use_voting == 'borda':
            df_sorted = self.scores_df_.sort_values(
                ['escore_borda', 'escore_meta_comite', 'dezena'], ascending=[False, False, True]
            )
        elif use_voting == 'hybrid':
            df_sorted = self.scores_df_.sort_values(
                ['escore_meta_comite', 'votos_hard', 'dezena'], ascending=[False, False, True]
            )
        else:
            raise ValueError(f"voting deve ser 'soft', 'hard', 'borda' ou 'hybrid'. Recebido: {use_voting}")

        return sorted([int(x) for x in df_sorted.head(k)['dezena'].tolist()])

    def get_scores_df(self) -> pd.DataFrame:
        """Retorna o DataFrame detalhado com os escores do Meta-Comitê."""
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")
        return self.scores_df_.copy()


LotterySuperEnsembleCommitteeModel = LotteryPretrainedMetaCommitteeModel


class LotteryDeepEnsembleCommitteeModel:
    """
    Meta-Ensemble Comitê Profundo Multi-Família Re-Estimado (Método 36, padrão M=100).
    Treina um comitê robusto de M membros (padrão M=100, suportando também M=50 ou qualquer M>0)
    sobre réplicas bootstrap cronológicas de binary_matrix, ciclando por 6 paradigmas de
    ensembles de alta performance do projeto e ancorando sobre os ensembles já treinados:
      1. Logit Gradient Boosting (Binomial Deviance com perturbação de eta e estágios)
      2. Residual Cycling / Stochastic Gradient Boosting (ciclagem residual multi-modelo)
      3. Momentum Multi-Janela com Bootstrap (decaimento exponencial multi-escala)
      4. Balanceado Multi-Escala com Bootstrap (frequência ponderada + atraso relativo)
      5. Gradiente Descendente com Subespaço de Atributos Analíticos (GD Bootstrap)
      6. Bayesiano Adaptativo + Geometria de Gianella com Bootstrap

    Inclui validação Out-of-Bag (OOB) vetorizada por membro (Zero Lookahead Bias),
    ponderação OOB suave e fusão convexa com a âncora dos ensembles já treinados.
    """

    SUPPORTED_FAMILIES = (
        'logit_gb',
        'cycling_gb',
        'stoch_gb',
        'momentum_bootstrap',
        'balanced_bootstrap',
        'gd_bootstrap',
        'bayes_gianella_bootstrap',
    )

    def __init__(self,
                 n_committees: int = 100,
                 max_samples: float = 0.85,
                 anchor_weight: float = 0.30,
                 weight_by_oob: bool = True,
                 oob_score: bool = True,
                 voting: str = 'hybrid',
                 families: Optional[List[str]] = None,
                 random_state: int = 42):
        if n_committees <= 0:
            raise ValueError(f"n_committees deve ser estritamente positivo (> 0). Recebido: {n_committees}")
        if not (0.0 < max_samples <= 1.0):
            raise ValueError(f"max_samples deve estar no intervalo (0.0, 1.0]. Recebido: {max_samples}")
        if not (0.0 <= anchor_weight <= 1.0):
            raise ValueError(f"anchor_weight deve estar entre 0.0 e 1.0. Recebido: {anchor_weight}")
        if voting not in ('soft', 'hard', 'hybrid'):
            raise ValueError(f"voting deve ser 'soft', 'hard' ou 'hybrid'. Recebido: {voting}")

        if families is None:
            families = [
                'logit_gb',
                'cycling_gb',
                'momentum_bootstrap',
                'balanced_bootstrap',
                'gd_bootstrap',
                'bayes_gianella_bootstrap',
            ]
        else:
            if len(families) == 0:
                raise ValueError("families não pode ser vazio.")
            for fam in families:
                if fam not in self.SUPPORTED_FAMILIES:
                    raise ValueError(f"Família '{fam}' inválida. Suportadas: {self.SUPPORTED_FAMILIES}")

        self.n_committees = int(n_committees)
        self.max_samples = float(max_samples)
        self.anchor_weight = float(anchor_weight)
        self.weight_by_oob = bool(weight_by_oob)
        self.oob_score = bool(oob_score)
        self.voting = voting
        self.families = list(families)
        self.random_state = int(random_state)

        self.estimators_: List[Dict[str, Any]] = []
        self.oob_score_: float = 0.0
        self.family_oob_scores_: Dict[str, float] = {}
        self.convergence_curve_: List[Dict[str, float]] = []
        self.committee_std_: Optional[np.ndarray] = None
        self.mean_se_: float = 0.0
        self.scores_df_: Optional[pd.DataFrame] = None
        self.top_k_: Optional[List[int]] = None

    def _compute_anchor_scores(self,
                               binary_matrix: np.ndarray,
                               cfg: Dict,
                               trained_ensembles: Optional[Dict[str, Any]]) -> np.ndarray:
        n_concursos, n_dezenas = binary_matrix.shape
        if trained_ensembles is not None:
            if not isinstance(trained_ensembles, dict) or len(trained_ensembles) == 0:
                raise ValueError("trained_ensembles deve ser um dicionário não-vazio com ensembles ajustados.")
            main_names, main_mat, oob_norm, _, _ = _extract_trained_ensemble_signals(
                trained_ensembles, n_dezenas, include_sub_estimators=False
            )
            if len(main_names) == 0:
                raise ValueError("trained_ensembles não contém nenhum ensemble ajustado válido.")
            weights = 1.0 + 0.35 * oob_norm
            weights /= np.sum(weights)
            return _min_max_scale(weights @ main_mat)

        # Fallback autônomo rápido quando trained_ensembles não é fornecido
        anchor_models: Dict[str, Any] = {
            'logit_gb': LotteryLogitGradientBoostingModel(n_estimators=10, learning_rate=0.3, random_state=self.random_state).fit(binary_matrix, cfg=cfg),
            'cycle_gb': LotteryResidualCyclingGradientBoostingModel(n_estimators=8, learning_rate=0.4, random_state=self.random_state).fit(binary_matrix, cfg=cfg),
            'mom_ens': LotteryMomentumEnsembleModel(n_estimators=10, random_state=self.random_state).fit(binary_matrix, cfg=cfg),
            'bal_ens': LotteryBalancedEnsembleModel(n_estimators=10, random_state=self.random_state).fit(binary_matrix, cfg=cfg),
            'bagged_lgb': LotteryBaggedLogitGBEnsembleModel(n_estimators=10, random_state=self.random_state).fit(binary_matrix, cfg=cfg),
        }
        if n_concursos >= 15:
            try:
                anchor_models['batch_gd'] = LotteryBatchGDEnsembleModel(n_epochs=60, learning_rate=0.1, random_state=self.random_state).fit(binary_matrix, cfg=cfg)
            except Exception:
                pass
        _, main_mat, oob_norm, _, _ = _extract_trained_ensemble_signals(
            anchor_models, n_dezenas, include_sub_estimators=False
        )
        weights = 1.0 + 0.35 * oob_norm
        weights /= np.sum(weights)
        return _min_max_scale(weights @ main_mat)

    def fit(self,
            binary_matrix: np.ndarray,
            cfg: Optional[Dict] = None,
            trained_ensembles: Optional[Dict[str, Any]] = None) -> 'LotteryDeepEnsembleCommitteeModel':
        """
        Treina o Comitê Profundo de M membros com bootstrap cronológico e validação OOB.
        """
        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        cfg = cfg or {}
        n_concursos, n_dezenas = binary_matrix.shape
        k = int(cfg.get('k', 6))
        k_ballot = min(max(1, k), n_dezenas)

        rng = np.random.RandomState(self.random_state)
        m_samples = max(1, int(self.max_samples * n_concursos))

        anchor_scores = (
            self._compute_anchor_scores(binary_matrix, cfg, trained_ensembles)
            if self.anchor_weight > 0.0
            else np.zeros(n_dezenas, dtype=float)
        )

        self.estimators_ = []
        self.convergence_curve_ = []
        all_scores = np.zeros((self.n_committees, n_dezenas), dtype=float)
        member_oob_means: List[Optional[float]] = []
        hard_votes = np.zeros(n_dezenas, dtype=float)

        oob_hits_total = 0.0
        oob_eval_count = 0
        fam_oob_hits: Dict[str, float] = {f: 0.0 for f in self.families}
        fam_oob_counts: Dict[str, int] = {f: 0 for f in self.families}

        mom_half_lives = [15.0, 25.0, 40.0, 60.0, 90.0, 120.0, 180.0, 250.0]
        bal_alphas = [0.25, 0.35, 0.50, 0.65, 0.75]
        bal_half_lives = [30.0, 60.0, 100.0, 150.0, 250.0]
        group_size = int(cfg.get('gianella_group_size', 10 if n_dezenas % 10 == 0 else (5 if n_dezenas % 5 == 0 else n_dezenas)))
        alpha_0 = float(cfg.get('alpha_0', 1.0))
        beta_0 = float(cfg.get('beta_0', 9.0))

        running_sum = np.zeros(n_dezenas, dtype=float)
        prev_top_set: Optional[set] = None

        for m in range(self.n_committees):
            fam = self.families[m % len(self.families)]
            actual_fam = fam
            boot_idx = np.sort(rng.choice(n_concursos, size=m_samples, replace=True))
            oob_idx = np.setdiff1d(np.arange(n_concursos), boot_idx)
            X_b = binary_matrix[boot_idx]
            sub_seed = int(rng.randint(0, 1_000_000))

            if fam == 'logit_gb':
                n_iter = int(rng.randint(6, 13))
                lr = float(rng.uniform(0.20, 0.45))
                est = LotteryLogitGradientBoostingModel(
                    n_estimators=n_iter, learning_rate=lr, random_state=sub_seed
                ).fit(X_b, cfg=cfg)
                probs = est.scores_df_.sort_values('dezena')['probabilidade_calibrada'].values
                score_m = _min_max_scale(probs)

            elif fam == 'cycling_gb':
                n_iter = int(rng.randint(6, 11))
                lr = float(rng.uniform(0.25, 0.50))
                est = LotteryResidualCyclingGradientBoostingModel(
                    n_estimators=n_iter, learning_rate=lr, random_state=sub_seed
                ).fit(X_b, cfg=cfg)
                vals = est.scores_df_.sort_values('dezena')['escore_cycling_gradboost'].values
                score_m = _min_max_scale(vals)

            elif fam == 'stoch_gb':
                n_iter = int(rng.randint(6, 13))
                lr = float(rng.uniform(0.30, 0.65))
                subs = float(rng.uniform(0.60, 0.90))
                est = LotteryStochasticGradientBoostingModel(
                    n_estimators=n_iter, learning_rate=lr, subsample=subs, random_state=sub_seed
                ).fit(X_b, cfg=cfg)
                vals = est.scores_df_.sort_values('dezena')['escore_stoch_gradboost'].values
                score_m = _min_max_scale(vals)

            elif fam == 'momentum_bootstrap':
                hl = float(mom_half_lives[m % len(mom_half_lives)] * rng.uniform(0.85, 1.15))
                decay = 0.5 ** (1.0 / max(1.0, hl))
                t_w = decay ** np.arange(len(boot_idx) - 1, -1, -1)
                sum_w = float(np.sum(t_w))
                w_freq = np.sum(X_b * t_w[:, None], axis=0) / (sum_w if sum_w > 0 else 1.0)
                score_m = _min_max_scale(w_freq)

            elif fam == 'balanced_bootstrap':
                alpha_b = float(np.clip(bal_alphas[m % len(bal_alphas)] + rng.uniform(-0.08, 0.08), 0.10, 0.90))
                hl_b = float(bal_half_lives[(m // len(bal_alphas)) % len(bal_half_lives)] * rng.uniform(0.85, 1.15))
                est = FrequencyDelayModel(half_life=max(5.0, hl_b), alpha=alpha_b).fit(X_b)
                vals = est.scores_df_.sort_values('dezena')['escore_composto'].values
                score_m = _min_max_scale(vals)

            elif fam == 'gd_bootstrap':
                # Extrai snapshots estritamente dentro da réplica X_b (Zero vazamento de oob_idx)
                X_gd = X_b[-min(len(X_b), 600):]
                n_gd = len(X_gd)
                if n_gd >= 2:
                    sample_indices = np.unique(
                        np.linspace(max(1, n_gd - 30), n_gd - 1,
                                    num=min(8, max(1, n_gd - 1)), dtype=int)
                    )
                    gd_snaps = [
                        (_extract_gd_analytical_features(X_gd[:idx], cfg), X_gd[idx].astype(float))
                        for idx in sample_indices
                    ]
                    gd_tg = _extract_gd_analytical_features(X_gd, cfg)
                    n_snap = len(gd_snaps)
                    snap_idx = rng.choice(n_snap, size=max(3, n_snap), replace=True)
                    X_tr = np.vstack([gd_snaps[i][0] for i in snap_idx])
                    y_tr = np.concatenate([gd_snaps[i][1] for i in snap_idx])
                    d_feats = X_tr.shape[1]
                    q_feats = max(8, min(d_feats, int(np.ceil(0.75 * d_feats))))
                    feat_idx = np.sort(rng.choice(d_feats, size=q_feats, replace=False))
                    X_tr_sub = X_tr[:, feat_idx]
                    X_tg_sub = gd_tg[:, feat_idx]
                    w = rng.normal(0.0, 0.01, size=q_feats)
                    b_val = 0.0
                    lr_gd = float(rng.uniform(0.06, 0.14))
                    l2_gd = float(rng.uniform(5e-4, 4e-3))
                    m_rows = float(X_tr_sub.shape[0])
                    for _ in range(60):
                        z = np.clip(X_tr_sub @ w + b_val, -25.0, 25.0)
                        p = 1.0 / (1.0 + np.exp(-z))
                        err = p - y_tr
                        w -= lr_gd * ((X_tr_sub.T @ err) / m_rows + l2_gd * w)
                        b_val -= lr_gd * float(np.mean(err))
                    p_tg = 1.0 / (1.0 + np.exp(-np.clip(X_tg_sub @ w + b_val, -25.0, 25.0)))
                    score_m = _min_max_scale(p_tg)
                else:
                    actual_fam = 'bayes_gianella_bootstrap'

            if actual_fam == 'bayes_gianella_bootstrap':
                hl_bayes = float(rng.uniform(80.0, 220.0))
                m_bayes = BayesianDecisionModel(
                    alpha_0=alpha_0, beta_0=beta_0, decay_half_life=hl_bayes
                ).fit(X_b)
                b_vals = m_bayes.posterior_df_.sort_values('dezena')['esperanca_posteriori'].values
                b_scaled = _min_max_scale(b_vals)
                if group_size > 0 and n_dezenas % group_size == 0 and k_ballot <= n_dezenas:
                    try:
                        m_gia = GianellaColorTemplateModel(
                            n_dezenas=n_dezenas, k=k_ballot, group_size=group_size
                        ).fit(X_b)
                        g_scaled = _min_max_scale(m_gia.dezena_scores_)
                        score_m = _min_max_scale(0.65 * b_scaled + 0.35 * g_scaled)
                    except Exception:
                        score_m = b_scaled
                else:
                    score_m = b_scaled

            score_m = np.nan_to_num(score_m, nan=0.0, posinf=1.0, neginf=0.0)
            order_m = np.argsort(score_m, kind='mergesort')[::-1]
            top_m = sorted((order_m[:k_ballot] + 1).tolist())
            for dez in top_m:
                hard_votes[dez - 1] += 1.0

            all_scores[m] = score_m

            m_oob_mean: Optional[float] = None
            if self.oob_score and len(oob_idx) > 0:
                top_mask = np.zeros(n_dezenas, dtype=float)
                top_mask[np.array(top_m, dtype=int) - 1] = 1.0
                hits_vec = binary_matrix[oob_idx] @ top_mask
                m_oob_sum = float(hits_vec.sum())
                n_oob_len = len(oob_idx)
                m_oob_mean = m_oob_sum / float(n_oob_len)
                oob_hits_total += m_oob_sum
                oob_eval_count += n_oob_len
                fam_oob_hits[actual_fam] = fam_oob_hits.get(actual_fam, 0.0) + m_oob_sum
                fam_oob_counts[actual_fam] = fam_oob_counts.get(actual_fam, 0) + n_oob_len
            member_oob_means.append(m_oob_mean)

            self.estimators_.append({
                'member_idx': m,
                'family': actual_fam,
                'n_oob': len(oob_idx),
                'oob_mean_hits': round(m_oob_mean, 4) if m_oob_mean is not None else None,
                'top_k': top_m,
            })

            running_sum += score_m
            cur_mean = running_sum / float(m + 1)
            cur_top_set = set((np.argsort(cur_mean, kind='mergesort')[::-1][:k_ballot] + 1).tolist())
            if prev_top_set is not None:
                union_sz = len(cur_top_set | prev_top_set)
                jaccard = len(cur_top_set & prev_top_set) / float(union_sz) if union_sz > 0 else 1.0
            else:
                jaccard = 1.0
            prev_top_set = cur_top_set
            self.convergence_curve_.append({
                'committee_size': m + 1,
                'top_k_jaccard': round(float(jaccard), 4),
                'score_se': round(float(np.mean(np.std(all_scores[:m + 1], axis=0) / np.sqrt(m + 1))), 6),
            })

        self.oob_score_ = float(oob_hits_total / oob_eval_count) if oob_eval_count > 0 else 0.0
        self.family_oob_scores_ = {
            f: round(float(fam_oob_hits[f] / fam_oob_counts[f]), 4) if fam_oob_counts[f] > 0 else 0.0
            for f in fam_oob_counts
        }

        # Pesos OOB suaves (Zero Lookahead Bias)
        valid_oobs = [x for x in member_oob_means if x is not None]
        if self.weight_by_oob and len(valid_oobs) > 0:
            mean_valid_oob = float(np.mean(valid_oobs))
            oob_arr = np.asarray(
                [x if x is not None else mean_valid_oob for x in member_oob_means],
                dtype=float
            )
            std_oob = float(np.std(oob_arr))
            if std_oob > 1e-8:
                z_oob = (oob_arr - float(np.mean(oob_arr))) / std_oob
                m_weights = np.clip(np.exp(0.35 * z_oob), 0.5, 2.0)
            else:
                m_weights = np.ones(self.n_committees, dtype=float)
            m_weights /= np.sum(m_weights)
        else:
            m_weights = np.full(self.n_committees, 1.0 / float(self.n_committees), dtype=float)

        committee_soft = m_weights @ all_scores
        vote_rate = hard_votes / float(self.n_committees)
        committee_pure = _min_max_scale(0.70 * _min_max_scale(committee_soft) + 0.30 * vote_rate)

        self.committee_std_ = np.std(all_scores, axis=0)
        se_vec = self.committee_std_ / np.sqrt(float(self.n_committees))
        self.mean_se_ = float(np.mean(se_vec))

        if self.anchor_weight > 0.0:
            final_score = _min_max_scale((1.0 - self.anchor_weight) * committee_pure + self.anchor_weight * anchor_scores)
        else:
            final_score = committee_pure

        dezenas = np.arange(1, n_dezenas + 1)
        self.scores_df_ = pd.DataFrame({
            'dezena': dezenas,
            'escore_comite_profundo': np.round(final_score, 5),
            'escore_comite_puro': np.round(committee_pure, 5),
            'escore_ancora': np.round(anchor_scores, 5),
            'votos_hard': hard_votes.astype(int),
            'taxa_consenso_pct': np.round(vote_rate * 100.0, 2),
            'incerteza_se': np.round(se_vec, 6),
        }).sort_values(['escore_comite_profundo', 'escore_comite_puro', 'dezena'], ascending=[False, False, True]).reset_index(drop=True)

        self.top_k_ = self.predict_top_k(k_ballot, voting=self.voting) if k > 0 else []
        return self

    def predict_top_k(self, k: Optional[int] = None, voting: Optional[str] = None) -> List[int]:
        """Retorna as top K dezenas segundo o critério de votação do Comitê Profundo."""
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")
        if k is None:
            k = len(self.top_k_) if self.top_k_ is not None else 6
        if k <= 0:
            return []
        k = min(k, len(self.scores_df_))

        use_voting = voting or self.voting
        if use_voting == 'hard':
            df_sorted = self.scores_df_.sort_values(
                ['votos_hard', 'escore_comite_profundo', 'dezena'], ascending=[False, False, True]
            )
        elif use_voting == 'soft':
            df_sorted = self.scores_df_.sort_values(
                ['escore_comite_puro', 'escore_comite_profundo', 'dezena'], ascending=[False, False, True]
            )
        elif use_voting == 'hybrid':
            df_sorted = self.scores_df_.sort_values(
                ['escore_comite_profundo', 'votos_hard', 'dezena'], ascending=[False, False, True]
            )
        else:
            raise ValueError(f"voting deve ser 'soft', 'hard' ou 'hybrid'. Recebido: {use_voting}")

        return sorted([int(x) for x in df_sorted.head(k)['dezena'].tolist()])

    def get_scores_df(self) -> pd.DataFrame:
        """Retorna o DataFrame detalhado com os escores do Comitê Profundo."""
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")
        return self.scores_df_.copy()


LotteryHighCommitteeBootstrappedMetaModel = LotteryDeepEnsembleCommitteeModel


def _resolve_gianella_group_size(n_dezenas: int, requested_size: int) -> int:
    """Retorna um tamanho de grupo divisor de n_dezenas <= requested_size."""
    if requested_size > 0 and n_dezenas % requested_size == 0:
        return int(requested_size)
    upper = min(max(1, int(requested_size)), int(n_dezenas))
    for cand in range(upper, 0, -1):
        if n_dezenas % cand == 0:
            return cand
    return 1


class LotteryGianellaCommitteeModel:
    """
    Meta-Ensemble Comitê de Gianella (Método 37, padrão M=100).
    Combina a Geometria do Acaso multi-gabarito (Gianella, 2013) com:
      1. Multi-Paradigm Signal Bank + Encolhimento Bayesiano / James-Stein (Shrinkage):
         Unifica os 15 ensembles pré-treinados com sinais fundamentais (Bayes Global,
         Bayes Adaptativo, Frequência/Atraso Balanceado e Momentum, Regressão Logística),
         aplicando encolhimento em direção à priori uniforme neutra para cortar ruído amostral.
      2. Distribuição Multi-Template (Top-K Gabaritos Diofantinos + Transição Markoviana):
         Em vez de fixar apenas o 1º gabarito combinatório, amostra proporcionalmente entre
         os Top-3 gabaritos mais prováveis com suavização bayesiana e viés de transição.
      3. Filtros Combinatórios Ortogonais (TCL Soma + Paridade):
         Projeta os bilhetes dentro das cotas exatas de cores de Gianella satisfazendo a
         faixa central da Soma pelo Teorema Central do Limite e o equilíbrio Par/Ímpar.
      4. Comitê Monte Carlo (padrão M=100, Random Subspaces + Dirichlet + Calibração Térmica):
         Reduz a variância do estimador e quantifica a incerteza padrão (SE).
    """

    def __init__(self,
                 n_committees: int = 100,
                 top_templates: int = 3,
                 shrinkage_alpha: float = 0.25,
                 subspace_ratio: float = 0.65,
                 include_sub_estimators: bool = True,
                 dirichlet_alpha: float = 1.5,
                 temperature_range: Tuple[float, float] = (0.75, 1.50),
                 use_structural_filter: bool = True,
                 tcl_z_bound: float = 1.35,
                 voting: str = 'hybrid',
                 random_state: int = 42):
        if n_committees <= 0:
            raise ValueError(f"n_committees deve ser estritamente positivo (> 0). Recebido: {n_committees}")
        if top_templates <= 0:
            raise ValueError(f"top_templates deve ser > 0. Recebido: {top_templates}")
        if not (0.0 <= shrinkage_alpha <= 1.0):
            raise ValueError(f"shrinkage_alpha deve estar em [0.0, 1.0]. Recebido: {shrinkage_alpha}")
        if not (0.0 < subspace_ratio <= 1.0):
            raise ValueError(f"subspace_ratio deve estar em (0.0, 1.0]. Recebido: {subspace_ratio}")
        if dirichlet_alpha <= 0.0:
            raise ValueError(f"dirichlet_alpha deve ser > 0. Recebido: {dirichlet_alpha}")
        if tcl_z_bound <= 0.0:
            raise ValueError(f"tcl_z_bound deve ser > 0. Recebido: {tcl_z_bound}")
        if voting not in ('hybrid', 'structural', 'soft', 'hard', 'borda'):
            raise ValueError(
                f"voting deve ser 'hybrid', 'structural', 'soft', 'hard' ou 'borda'. Recebido: {voting}"
            )
        if len(temperature_range) != 2 or temperature_range[0] <= 0.0 or temperature_range[0] > temperature_range[1]:
            raise ValueError(f"temperature_range inválido: {temperature_range}")

        self.n_committees = int(n_committees)
        self.top_templates = int(top_templates)
        self.shrinkage_alpha = float(shrinkage_alpha)
        self.subspace_ratio = float(subspace_ratio)
        self.include_sub_estimators = bool(include_sub_estimators)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.temperature_range = temperature_range
        self.use_structural_filter = bool(use_structural_filter)
        self.tcl_z_bound = float(tcl_z_bound)
        self.voting = voting
        self.random_state = int(random_state)

        self.trained_ensembles_: Dict[str, Any] = {}
        self.signal_names_: List[str] = []
        self.sub_estimator_count_: int = 0
        self.ensemble_contributions_: Dict[str, float] = {}
        self.group_size_: int = 10
        self.n_colors_: int = 6
        self.k_fit_: int = 6
        self.template_distribution_: List[Tuple[Tuple[int, ...], float]] = []
        self.template_usage_: Dict[str, int] = {}
        self.best_template_: Optional[Tuple[int, ...]] = None
        self.color_scores_comite_: Optional[np.ndarray] = None
        self.sum_bounds_: Tuple[float, float] = (0.0, 0.0)
        self.even_bounds_: Tuple[int, int] = (0, 0)
        self.sum_sigma_: float = 1.0
        self.committee_members_: List[Dict[str, Any]] = []
        self.convergence_curve_: List[Dict[str, float]] = []
        self.committee_std_: Optional[np.ndarray] = None
        self.mean_se_: float = 0.0
        self.scores_df_: Optional[pd.DataFrame] = None
        self.top_k_: Optional[List[int]] = None
        self.top_ticket_metrics_: Dict[str, Any] = {}

    def _compute_structural_bounds(self, n_dezenas: int, k: int) -> Tuple[Tuple[float, float], Tuple[int, int], float]:
        k_eff = min(max(1, int(k)), int(n_dezenas))
        mu_sum = float(k_eff) * float(n_dezenas + 1) / 2.0
        var_sum = float(k_eff) * float((n_dezenas + 1) * max(0, n_dezenas - k_eff)) / 12.0
        sigma_sum = float(np.sqrt(max(var_sum, 1.0)))
        sum_low = mu_sum - self.tcl_z_bound * sigma_sum
        sum_high = mu_sum + self.tcl_z_bound * sigma_sum

        n_evens_universe = n_dezenas // 2
        exp_even = float(k_eff) * (float(n_evens_universe) / float(max(1, n_dezenas)))
        tol_even = max(1, int(round(0.22 * k_eff)))
        even_low = max(0, int(np.floor(exp_even)) - tol_even)
        even_high = min(k_eff, int(np.ceil(exp_even)) + tol_even)
        return (sum_low, sum_high), (even_low, even_high), sigma_sum

    def _build_multitemplate_distribution(self,
                                          binary_matrix: np.ndarray,
                                          g_model: GianellaColorTemplateModel) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
        df_t = g_model.template_df_
        top_df = df_t.head(min(self.top_templates, len(df_t))).copy()
        templates = [tuple(int(x) for x in pt) for pt in top_df['part_tuple'].tolist()]
        theo_probs = np.asarray(top_df['prob_teorica'].values, dtype=float)
        theo_probs = theo_probs / max(float(theo_probs.sum()), 1e-12)

        n_draws = binary_matrix.shape[0]
        emp_counts = np.asarray(top_df['ocorrencias'].values, dtype=float)
        emp_smoothed = (emp_counts + 1.0) / float(max(1, n_draws) + len(templates))
        emp_smoothed = emp_smoothed / max(float(emp_smoothed.sum()), 1e-12)

        base_probs = 0.70 * theo_probs + 0.30 * emp_smoothed

        if n_draws >= 2 and len(templates) > 1:
            seq_templates: List[str] = []
            for d in range(n_draws):
                counts = sorted(
                    [int(binary_matrix[d, c * self.group_size_:(c + 1) * self.group_size_].sum())
                     for c in range(self.n_colors_)],
                    reverse=True,
                )
                seq_templates.append('-'.join(str(x) for x in counts))
            last_tpl = seq_templates[-1]
            tpl_strs = ['-'.join(str(x) for x in t) for t in templates]
            trans_counts = np.zeros(len(templates), dtype=float)
            for idx_d in range(n_draws - 1):
                if seq_templates[idx_d] == last_tpl and seq_templates[idx_d + 1] in tpl_strs:
                    trans_counts[tpl_strs.index(seq_templates[idx_d + 1])] += 1.0
            prior_w = 20.0
            markov_probs = (trans_counts + prior_w * base_probs) / float(trans_counts.sum() + prior_w)
            combined = 0.85 * base_probs + 0.15 * markov_probs
        else:
            combined = base_probs

        combined = combined / max(float(combined.sum()), 1e-12)
        return templates, combined

    def _collect_shrinkage_signals(self,
                                   binary_matrix: np.ndarray,
                                   cfg: Dict,
                                   trained_ensembles: Optional[Dict[str, Any]],
                                   base_signals: Optional[Dict[str, Any]],
                                   n_dezenas: int,
                                   k: int) -> Tuple[List[str], np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
        n_concursos = binary_matrix.shape[0]
        if trained_ensembles is not None:
            if not isinstance(trained_ensembles, dict) or len(trained_ensembles) == 0:
                raise ValueError("trained_ensembles deve ser um dicionário não-vazio.")
            self.trained_ensembles_ = dict(trained_ensembles)
        else:
            helper = LotteryPretrainedMetaCommitteeModel(n_committees=10, random_state=self.random_state)
            self.trained_ensembles_ = helper._fit_default_base_ensembles(binary_matrix, cfg)

        ens_names, ens_mat, ens_oob, sub_names, sub_mat = _extract_trained_ensemble_signals(
            self.trained_ensembles_, n_dezenas, include_sub_estimators=self.include_sub_estimators
        )

        bs = base_signals or {}
        m1 = bs.get('m1')
        if m1 is None or getattr(m1, 'scores_df_', None) is None:
            m1 = FrequencyDelayModel(half_life=100.0, alpha=0.5).fit(binary_matrix)
        m3_static = bs.get('m3_static')
        if m3_static is None or getattr(m3_static, 'posterior_df_', None) is None:
            m3_static = BayesianDecisionModel(
                alpha_0=cfg.get('alpha_0', 1.0), beta_0=cfg.get('beta_0', 9.0)
            ).fit(binary_matrix)
        m3_dyn = bs.get('m3_dyn')
        if m3_dyn is None or getattr(m3_dyn, 'posterior_df_', None) is None:
            m3_dyn = BayesianDecisionModel(
                alpha_0=cfg.get('alpha_0', 1.0), beta_0=cfg.get('beta_0', 9.0), decay_half_life=150.0
            ).fit(binary_matrix)
        m2 = bs.get('m2')
        if m2 is None and n_concursos >= 25:
            try:
                m2 = MultiLabelMLModel(estimator='logistic_regression', random_state=self.random_state).fit(binary_matrix)
            except Exception:
                m2 = None

        fund_names: List[str] = []
        fund_vecs: List[np.ndarray] = []

        # Bayes Global (também usado como ancora bayesiana de shrinkage)
        bg_vec = np.zeros(n_dezenas, dtype=float)
        for _, r in m3_static.posterior_df_.iterrows():
            d_i = int(r['dezena']) - 1
            if 0 <= d_i < n_dezenas:
                bg_vec[d_i] = float(r['esperanca_posteriori'])
        bayes_anchor = _min_max_scale(bg_vec)
        fund_names.append('bayes_global')
        fund_vecs.append(bayes_anchor)

        bd_vec = np.zeros(n_dezenas, dtype=float)
        for _, r in m3_dyn.posterior_df_.iterrows():
            d_i = int(r['dezena']) - 1
            if 0 <= d_i < n_dezenas:
                bd_vec[d_i] = float(r['esperanca_posteriori'])
        fund_names.append('bayes_adaptive')
        fund_vecs.append(_min_max_scale(bd_vec))

        bal_vec = np.zeros(n_dezenas, dtype=float)
        mom_vec = np.zeros(n_dezenas, dtype=float)
        for _, r in m1.scores_df_.iterrows():
            d_i = int(r['dezena']) - 1
            if 0 <= d_i < n_dezenas:
                bal_vec[d_i] = float(r['escore_composto'])
                mom_vec[d_i] = float(r['freq_ponderada'])
        fund_names.extend(['freq_balanced', 'freq_momentum'])
        fund_vecs.extend([_min_max_scale(bal_vec), _min_max_scale(mom_vec)])

        if m2 is not None:
            try:
                df_ml = m2.get_probabilities_df()
                ml_vec = np.zeros(n_dezenas, dtype=float)
                for _, r in df_ml.iterrows():
                    d_i = int(r['dezena']) - 1
                    if 0 <= d_i < n_dezenas:
                        ml_vec[d_i] = float(r['probabilidade_calibrada_sorteio'])
                fund_names.append('ml_logistic')
                fund_vecs.append(_min_max_scale(ml_vec))
            except Exception:
                pass

        all_names = ens_names + fund_names
        if len(ens_names) > 0:
            main_mat = np.vstack([ens_mat, np.vstack(fund_vecs)])
            mean_oob = float(np.mean(ens_oob)) if len(ens_oob) > 0 else 0.5
            fund_oobs = np.full(len(fund_names), max(0.35, mean_oob), dtype=float)
            oob_vec = np.concatenate([ens_oob, fund_oobs])
        else:
            main_mat = np.vstack(fund_vecs)
            oob_vec = np.full(len(fund_names), 0.5, dtype=float)

        # Prior neutra combinada (uniforme 0.5 + ancora bayesiana conjugada)
        prior_anchor = 0.60 * np.full(n_dezenas, 0.5, dtype=float) + 0.40 * bayes_anchor
        main_shrunk = (1.0 - self.shrinkage_alpha) * main_mat + self.shrinkage_alpha * prior_anchor[None, :]
        if len(sub_mat) > 0:
            sub_shrunk = (1.0 - self.shrinkage_alpha) * sub_mat + self.shrinkage_alpha * prior_anchor[None, :]
        else:
            sub_shrunk = sub_mat

        return all_names, main_shrunk, oob_vec, sub_names, sub_shrunk, prior_anchor

    def _structural_violation(self, ticket: List[int]) -> float:
        s_val = float(sum(ticket))
        e_val = sum(1 for x in ticket if x % 2 == 0)
        s_low, s_high = self.sum_bounds_
        e_low, e_high = self.even_bounds_
        v_sum = max(0.0, s_low - s_val, s_val - s_high) / max(self.sum_sigma_, 1.0)
        v_even = float(max(0, e_low - e_val, e_val - e_high))
        return v_sum + 1.5 * v_even

    def _project_gianella_ticket(self,
                                 scores_vec: np.ndarray,
                                 color_order: np.ndarray,
                                 template: Tuple[int, ...],
                                 n_dezenas: int,
                                 k_eval: int,
                                 apply_filter: bool = True) -> List[int]:
        if k_eval <= 0:
            return []
        if k_eval > n_dezenas:
            k_eval = n_dezenas

        # Caso k_eval difira do k calibrado ou cotas do template nao somem k_eval,
        # usa selecao direta pelo ranking de escores para manter consistencia de tamanho.
        if sum(template) != k_eval:
            top_idx = np.argsort(scores_vec, kind='mergesort')[::-1][:k_eval]
            return sorted((top_idx + 1).tolist())

        selected_by_color: Dict[int, List[int]] = {}
        unused_by_color: Dict[int, List[int]] = {}
        ticket: List[int] = []

        for idx, c_idx in enumerate(color_order):
            c_int = int(c_idx)
            quota = int(template[idx]) if idx < len(template) else 0
            start_d = c_int * self.group_size_ + 1
            end_d = min(n_dezenas, (c_int + 1) * self.group_size_) + 1
            c_balls = list(range(start_d, end_d))
            c_balls.sort(key=lambda d: (-float(scores_vec[d - 1]), d))
            picked = c_balls[:quota]
            rest = c_balls[quota:]
            selected_by_color[c_int] = picked
            unused_by_color[c_int] = rest
            ticket.extend(picked)

        if apply_filter and self.use_structural_filter and k_eval == self.k_fit_:
            cur_viol = self._structural_violation(ticket)
            max_swaps = min(k_eval, 8)
            for _ in range(max_swaps):
                if cur_viol <= 1e-9:
                    break
                best_swap = None
                best_key = (cur_viol, float('inf'))
                for c_int, picked in selected_by_color.items():
                    rest = unused_by_color[c_int]
                    if not picked or not rest:
                        continue
                    for out_b in picked:
                        for in_b in rest:
                            cand_ticket = [in_b if x == out_b else x for x in ticket]
                            v_new = self._structural_violation(cand_ticket)
                            if v_new < cur_viol - 1e-9:
                                score_loss = float(scores_vec[out_b - 1] - scores_vec[in_b - 1])
                                cand_key = (round(v_new, 6), score_loss)
                                if cand_key < best_key:
                                    best_key = cand_key
                                    best_swap = (c_int, out_b, in_b, v_new, cand_ticket)
                if best_swap is None:
                    break
                c_int, out_b, in_b, cur_viol, ticket = best_swap
                selected_by_color[c_int].remove(out_b)
                selected_by_color[c_int].append(in_b)
                unused_by_color[c_int].remove(in_b)
                unused_by_color[c_int].append(out_b)

        return sorted([int(x) for x in ticket[:k_eval]])

    def fit(self,
            binary_matrix: np.ndarray,
            cfg: Optional[Dict] = None,
            trained_ensembles: Optional[Dict[str, Any]] = None,
            base_signals: Optional[Dict[str, Any]] = None) -> 'LotteryGianellaCommitteeModel':
        """Ajusta o Comitê de Gianella M=100 com distribuição multi-template, shrinkage e filtros TCL/Paridade."""
        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:
            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        cfg = cfg or {}
        _, n_dezenas = binary_matrix.shape
        k = int(cfg.get('k', 6))
        req_group = int(cfg.get('gianella_group_size', 10 if n_dezenas >= 40 else 5))

        self.k_fit_ = k
        self.group_size_ = _resolve_gianella_group_size(n_dezenas, req_group)
        self.n_colors_ = n_dezenas // self.group_size_
        self.sum_bounds_, self.even_bounds_, self.sum_sigma_ = self._compute_structural_bounds(n_dezenas, k)

        # 1. Gabarito Multi-Template Gianella
        bs = base_signals or {}
        g_model = bs.get('gianella_model')
        if (g_model is None or getattr(g_model, 'template_df_', None) is None
                or getattr(g_model, 'n_dezenas', None) != n_dezenas
                or getattr(g_model, 'k', None) != k
                or getattr(g_model, 'group_size', None) != self.group_size_):
            g_model = GianellaColorTemplateModel(
                n_dezenas=n_dezenas, k=k, group_size=self.group_size_, random_state=self.random_state
            ).fit(binary_matrix)

        templates, tpl_probs = self._build_multitemplate_distribution(binary_matrix, g_model)
        self.template_distribution_ = [(templates[i], float(tpl_probs[i])) for i in range(len(templates))]
        self.best_template_ = templates[0]

        # 2. Banco de Sinais Multi-Paradigma + Shrinkage Bayesiano
        all_names, main_mat, oob_vec, sub_names, sub_mat, prior_anchor = self._collect_shrinkage_signals(
            binary_matrix, cfg, trained_ensembles, bs, n_dezenas, k
        )
        self.signal_names_ = all_names
        self.sub_estimator_count_ = len(sub_names)

        k_main = len(all_names)
        k_sub = len(sub_names)
        q_main = max(1, min(k_main, int(np.ceil(self.subspace_ratio * k_main))))
        if k_main >= 3:
            q_main = max(2, q_main)
        q_sub = max(1, min(k_sub, int(np.ceil(self.subspace_ratio * k_sub)))) if k_sub > 0 else 0

        alpha_prior = self.dirichlet_alpha * (1.0 + 0.35 * oob_vec)
        rng = np.random.RandomState(self.random_state)
        t_min, t_max = self.temperature_range

        all_soft = np.zeros((self.n_committees, n_dezenas), dtype=float)
        all_probs = np.zeros((self.n_committees, n_dezenas), dtype=float)
        all_borda = np.zeros((self.n_committees, n_dezenas), dtype=float)
        hard_votes = np.zeros(n_dezenas, dtype=float)
        color_scores_accum = np.zeros(self.n_colors_, dtype=float)
        accum_weights = np.zeros(k_main, dtype=float)
        self.template_usage_ = {'-'.join(str(x) for x in t): 0 for t in templates}
        self.committee_members_ = []
        self.convergence_curve_ = []

        running_soft = np.zeros(n_dezenas, dtype=float)
        prev_top_set: Optional[set] = None
        k_ballot = min(max(1, k), n_dezenas)

        for m in range(self.n_committees):
            idx_main = rng.choice(k_main, size=q_main, replace=False)
            w_main = rng.dirichlet(alpha_prior[idx_main])
            accum_weights[idx_main] += w_main
            s_main = w_main @ main_mat[idx_main]

            if q_sub > 0:
                idx_sub = rng.choice(k_sub, size=q_sub, replace=False)
                w_sub = rng.dirichlet(np.full(q_sub, self.dirichlet_alpha))
                s_sub = w_sub @ sub_mat[idx_sub]
                beta_sub = float(rng.uniform(0.15, 0.35))
                s_raw = (1.0 - beta_sub) * _min_max_scale(s_main) + beta_sub * _min_max_scale(s_sub)
            else:
                s_raw = _min_max_scale(s_main)

            s_shrunk = (1.0 - 0.15 * self.shrinkage_alpha) * _min_max_scale(s_raw) + (0.15 * self.shrinkage_alpha) * prior_anchor
            s_norm = _min_max_scale(s_shrunk)

            temp_m = float(rng.uniform(t_min, t_max))
            std_s = float(np.std(s_norm))
            z_s = (s_norm - float(np.mean(s_norm))) / (std_s if std_s > 1e-12 else 1.0)
            logits = np.clip(z_s / temp_m, -20.0, 20.0)
            exp_l = np.exp(logits - np.max(logits))
            p_m = exp_l / np.sum(exp_l)

            order = np.argsort(s_norm, kind='mergesort')[::-1]
            borda_m = np.zeros(n_dezenas, dtype=float)
            denom = float(max(1, n_dezenas - 1))
            for r_idx, d_idx in enumerate(order):
                borda_m[d_idx] = 1.0 - (r_idx / denom)

            # Sortear gabarito da distribuicao multi-template Gianella e projetar bilhete estrutural
            tpl_idx = int(rng.choice(len(templates), p=tpl_probs))
            tpl_m = templates[tpl_idx]
            tpl_str = '-'.join(str(x) for x in tpl_m)
            self.template_usage_[tpl_str] += 1

            c_means = np.asarray([
                float(np.mean(s_norm[c * self.group_size_:(c + 1) * self.group_size_]))
                for c in range(self.n_colors_)
            ], dtype=float)
            color_scores_accum += c_means
            c_jitter = c_means + rng.normal(0.0, 0.015, size=self.n_colors_)
            c_order_m = np.argsort(c_jitter, kind='mergesort')[::-1]

            ticket_m = self._project_gianella_ticket(
                s_norm, c_order_m, tpl_m, n_dezenas, k_ballot, apply_filter=self.use_structural_filter
            )
            for dez in ticket_m:
                hard_votes[dez - 1] += 1.0

            all_soft[m] = s_norm
            all_probs[m] = p_m
            all_borda[m] = borda_m

            self.committee_members_.append({
                'member_idx': m,
                'template': tpl_str,
                'temperature': round(temp_m, 4),
                'top_k': ticket_m,
            })

            running_soft += s_norm
            cur_mean = running_soft / float(m + 1)
            cur_top_set = set((np.argsort(cur_mean, kind='mergesort')[::-1][:k_ballot] + 1).tolist())
            if prev_top_set is not None:
                union_sz = len(cur_top_set | prev_top_set)
                jaccard = len(cur_top_set & prev_top_set) / float(union_sz) if union_sz > 0 else 1.0
            else:
                jaccard = 1.0
            prev_top_set = cur_top_set
            self.convergence_curve_.append({
                'committee_size': m + 1,
                'top_k_jaccard': round(float(jaccard), 4),
                'score_se': round(float(np.mean(np.std(all_soft[:m + 1], axis=0) / np.sqrt(m + 1))), 6),
            })

        soft_mean = np.mean(all_soft, axis=0)
        prob_mean = np.mean(all_probs, axis=0)
        borda_mean = np.mean(all_borda, axis=0)
        vote_rate = hard_votes / float(self.n_committees)

        self.color_scores_comite_ = color_scores_accum / float(self.n_committees)
        self.committee_std_ = np.std(all_soft, axis=0)
        se_vec = self.committee_std_ / np.sqrt(float(self.n_committees))
        self.mean_se_ = float(np.mean(se_vec))
        self.ensemble_contributions_ = {
            all_names[i]: float(np.round(accum_weights[i] / float(self.n_committees), 5))
            for i in range(k_main)
        }

        hybrid_raw = (
            0.35 * _min_max_scale(soft_mean)
            + 0.25 * _min_max_scale(prob_mean)
            + 0.15 * _min_max_scale(borda_mean)
            + 0.25 * vote_rate
        )
        hybrid_norm = _min_max_scale(hybrid_raw)

        dezenas = np.arange(1, n_dezenas + 1)
        color_groups = (dezenas - 1) // self.group_size_
        self.scores_df_ = pd.DataFrame({
            'dezena': dezenas,
            'grupo_cor': color_groups,
            'escore_gianella_comite': np.round(hybrid_norm, 5),
            'probabilidade_comite': np.round(prob_mean, 5),
            'escore_soft': np.round(soft_mean, 5),
            'escore_borda': np.round(borda_mean, 5),
            'votos_gianella': hard_votes.astype(int),
            'taxa_consenso_pct': np.round(vote_rate * 100.0, 2),
            'incerteza_se': np.round(se_vec, 6),
        }).sort_values(['escore_gianella_comite', 'votos_gianella', 'dezena'], ascending=[False, False, True]).reset_index(drop=True)

        self.top_k_ = self.predict_top_k(k_ballot, voting=self.voting) if k > 0 else []
        s_top = sum(self.top_k_)
        e_top = sum(1 for x in self.top_k_ if x % 2 == 0)
        self.top_ticket_metrics_ = {
            'top_k': self.top_k_,
            'best_template': '-'.join(str(x) for x in self.best_template_),
            'sum': s_top,
            'sum_bounds': tuple(round(v, 2) for v in self.sum_bounds_),
            'evens': e_top,
            'even_bounds': self.even_bounds_,
            'passes_tcl_sum': bool(self.sum_bounds_[0] <= s_top <= self.sum_bounds_[1]),
            'passes_parity': bool(self.even_bounds_[0] <= e_top <= self.even_bounds_[1]),
            'mean_se': round(self.mean_se_, 6),
        }
        return self

    def predict_top_k(self, k: Optional[int] = None, voting: Optional[str] = None) -> List[int]:
        """Retorna as top K dezenas segundo o critério estrutural Gianella + TCL/Paridade ou votação."""
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")
        if k is None:
            k = self.k_fit_
        if k <= 0:
            return []
        n_dezenas = len(self.scores_df_)
        k = min(int(k), n_dezenas)

        use_voting = voting or self.voting
        if use_voting in ('hybrid', 'structural'):
            score_col = 'escore_gianella_comite' if use_voting == 'hybrid' else 'taxa_consenso_pct'
            ordered_scores = np.zeros(n_dezenas, dtype=float)
            for _, row in self.scores_df_.iterrows():
                ordered_scores[int(row['dezena']) - 1] = float(row[score_col])
            c_order = np.argsort(self.color_scores_comite_, kind='mergesort')[::-1]
            apply_f = self.use_structural_filter or (use_voting == 'structural')
            return self._project_gianella_ticket(
                ordered_scores, c_order, self.best_template_, n_dezenas, k, apply_filter=apply_f
            )
        elif use_voting == 'hard':
            df_sorted = self.scores_df_.sort_values(
                ['votos_gianella', 'escore_gianella_comite', 'dezena'], ascending=[False, False, True]
            )
        elif use_voting == 'soft':
            df_sorted = self.scores_df_.sort_values(
                ['escore_soft', 'probabilidade_comite', 'dezena'], ascending=[False, False, True]
            )
        elif use_voting == 'borda':
            df_sorted = self.scores_df_.sort_values(
                ['escore_borda', 'escore_gianella_comite', 'dezena'], ascending=[False, False, True]
            )
        else:
            raise ValueError(
                f"voting deve ser 'hybrid', 'structural', 'soft', 'hard' ou 'borda'. Recebido: {use_voting}"
            )
        return sorted([int(x) for x in df_sorted.head(k)['dezena'].tolist()])

    def get_scores_df(self) -> pd.DataFrame:
        """Retorna o DataFrame detalhado com os escores do Comitê de Gianella."""
        if self.scores_df_ is None:
            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")
        return self.scores_df_.copy()


LotteryGianellaMetaCommitteeModel = LotteryGianellaCommitteeModel






class LotteryResidualCyclingGradientBoostingModel:

    """

    Ensemble Multi-Model Residual Cycling Gradient Boosting para Seleção de Dezenas em Loterias.

    Baseado em Aurélien Géron (2019), Hands-On Machine Learning (2ª Ed.), Capítulo 7 ('Gradient Boosting')

    e nos princípios de boosting com preditores base heterogêneos (Breiman, 1997; Friedman, 1999).



    Fundamentação Teórica e Algorítmica:

    1. Ciclagem Sequencial de Modelos Especialistas de Base:

       Ao invés de aplicar o mesmo estimador residual uniforme em todas as rodadas, cada estágio m de boosting

       corrige os resíduos remanescentes empregando um modelo analítico-base distinto do projeto:

       - Estágio m mod 4 = 1: M2 Regressão Logística L2 / ML ('ml_logistic')

         Ajusta projeção supervisionada e correlação cruzada temporal sobre o sinal residual.

       - Estágio m mod 4 = 2: M4 Geometria do Acaso de Gianella ('gianella')

         Compensa o déficit residual nas partições diofantinas de cores/quadrantes espaciais.

       - Estágio m mod 4 = 3: M1 Frequência / Atraso Relativo e Gap ('delay_momentum')

         Foca nas dezenas com atraso relativo crítico e gradiente de recuperação residual.

       - Estágio m mod 4 = 0: M3 Bayesiano Adaptativo ('bayes_adaptive')

         Atualiza a crença a posteriori sobre a densidade residual via conjugada Beta-Binomial adaptativa.

    2. Correção Cumulativa com Regularização por Encolhimento (Shrinkage, Géron p. 205):

       F_m = F_{m-1} + learning_rate * (h_norm - 0.5)

       R^{(m)} = R^{(m-1)} - learning_rate * (h_norm - 0.5)

    3. Diversidade Estrutural do Ensemble:

       A alternância entre modelos estatísticos, bayesianos, geométricos e conexionistas/ML garante que

       nenhum viés metodológico único domine o processo de boosting, capturando resíduos multifacetados.

    """



    SUPPORTED_BASE_MODELS = ('ml_logistic', 'gianella', 'delay_momentum', 'bayes_adaptive')



    def __init__(self,

                 n_estimators: int = 8,

                 learning_rate: float = 0.4,

                 base_cycle: Optional[List[str]] = None,

                 random_state: int = 42):

        """

        Parameters

        ----------

        n_estimators : int

            Número total de estágios de ciclagem residual (default 8, correspondente a 2 ciclos completos dos 4 modelos).

        learning_rate : float

            Taxa de aprendizado / encolhimento eta (default 0.4).

        base_cycle : Optional[List[str]]

            Sequência ordenada de modelos-base para ciclagem residual.

            Padrão: ['ml_logistic', 'gianella', 'delay_momentum', 'bayes_adaptive'].

        random_state : int

            Semente para reprodutibilidade.

        """

        if n_estimators <= 0:

            raise ValueError(f"n_estimators deve ser estritamente positivo (> 0). Recebido: {n_estimators}")

        if learning_rate <= 0:

            raise ValueError(f"learning_rate deve ser estritamente positivo (> 0). Recebido: {learning_rate}")



        if base_cycle is None:

            base_cycle = ['ml_logistic', 'gianella', 'delay_momentum', 'bayes_adaptive']

        else:

            if len(base_cycle) == 0:

                raise ValueError("base_cycle não pode ser vazio.")

            for mod in base_cycle:

                if mod not in self.SUPPORTED_BASE_MODELS:

                    raise ValueError(f"Modelo '{mod}' inválido em base_cycle. Suportados: {self.SUPPORTED_BASE_MODELS}")



        self.n_estimators = n_estimators

        self.learning_rate = learning_rate

        self.base_cycle = list(base_cycle)

        self.random_state = random_state



        self.staged_scores_: List[np.ndarray] = []

        self.stage_models_: List[str] = []

        self.scores_df_: Optional[pd.DataFrame] = None

        self.top_k_: Optional[List[int]] = None



    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None) -> 'LotteryResidualCyclingGradientBoostingModel':

        """

        Ajusta o Gradient Boosting ciclando pelos modelos base analíticos a cada estágio residual.

        """

        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:

            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")



        cfg = cfg or {}

        n_concursos, n_dezenas = binary_matrix.shape

        k = cfg.get('k', 6)



        # 1. Modelo inicial F_0: Estimativa a priori bayesiana suavizada

        alpha_0 = cfg.get('alpha_0', 1.0)

        beta_0 = cfg.get('beta_0', 9.0)

        F_current = (binary_matrix.sum(axis=0) + alpha_0) / (n_concursos + alpha_0 + beta_0)

        F_current = _min_max_scale(F_current)



        # Resíduos iniciais R^{(0)} = Y - F_0

        residuals = binary_matrix.astype(float) - F_current[None, :]

        self.staged_scores_ = [F_current.copy()]

        self.stage_models_ = []



        # Pré-cálculo de atrasos correntes para o modelo M1 (delay)

        delays = np.zeros(n_dezenas, dtype=float)

        for d in range(n_dezenas):

            idx_drawn = np.where(binary_matrix[:, d] == 1)[0]

            delays[d] = float(n_concursos - 1 - idx_drawn[-1]) if len(idx_drawn) > 0 else float(n_concursos)



        # Divisão em quadrantes/cores de Gianella (M4)

        group_size = cfg.get('gianella_group_size', max(1, n_dezenas // 4))



        for m in range(self.n_estimators):

            active_model = self.base_cycle[m % len(self.base_cycle)]

            self.stage_models_.append(active_model)



            half_life = 40.0 + (m * 15.0)

            decay = 0.5 ** (1.0 / half_life)

            time_weights = decay ** np.arange(n_concursos - 1, -1, -1)

            time_weights /= np.sum(time_weights)



            # 2. Ajuste do especialista residual de acordo com a metodologia ativa

            if active_model == 'ml_logistic':

                # Estágio M2: Classificador / Projeção residual cruzada

                # Mede a autocorrelação entre sorteios passados e os resíduos presentes

                if n_concursos >= 5:

                    X_lag = binary_matrix[:-1, :].astype(float)

                    R_next = residuals[1:, :]

                    # Projeção L2-regularizada: W = (X^T X + lambda I)^{-1} X^T R

                    cov_xx = np.dot(X_lag.T, X_lag) + 5.0 * np.eye(n_dezenas)

                    cov_xr = np.dot(X_lag.T, R_next)

                    last_state = binary_matrix[-1, :].astype(float)

                    try:

                        w_weights = np.linalg.solve(cov_xx, cov_xr)

                        h_m = np.dot(last_state, w_weights)

                    except Exception:

                        h_m = np.sum(residuals * time_weights[:, None], axis=0)

                else:

                    h_m = np.sum(residuals * time_weights[:, None], axis=0)



            elif active_model == 'gianella':

                # Estágio M4: Geometria do Acaso / Cores de Gianella

                # Identifica grupos de cores com déficit residual acumulado

                raw_grad = np.sum(residuals * time_weights[:, None], axis=0)

                h_m = raw_grad.copy()

                n_groups = int(np.ceil(n_dezenas / group_size))

                for g in range(n_groups):

                    start_d = g * group_size

                    end_d = min(n_dezenas, (g + 1) * group_size)

                    zone_indices = np.arange(start_d, end_d)

                    zone_residual = np.mean(raw_grad[zone_indices])

                    # Impulsiona as dezenas de quadrantes que apresentaram déficit positivo de resíduo

                    h_m[zone_indices] += 0.5 * zone_residual



            elif active_model == 'delay_momentum':

                # Estágio M1: Frequência e Atraso Relativo

                # Combina o momentum residual recente com o atraso corrente (gap de recorrência)

                raw_grad = np.sum(residuals * time_weights[:, None], axis=0)

                delay_norm = _min_max_scale(delays)

                # Dezenas atrasadas com gradiente residual positivo recebem forte aceleração

                h_m = 0.6 * raw_grad + 0.4 * (raw_grad * (1.0 + delay_norm))



            elif active_model == 'bayes_adaptive':

                # Estágio M3: Decisão Bayesiana Adaptativa

                # Atualização a posteriori conjugada sobre a energia positiva dos resíduos

                pos_residuals = np.maximum(0.0, residuals)

                bayes_pseudo_counts = np.sum(pos_residuals * time_weights[:, None], axis=0)

                # Escore a posteriori Beta-Binomial suavizado

                h_m = (bayes_pseudo_counts + alpha_0) / (np.sum(bayes_pseudo_counts) / n_dezenas + alpha_0 + beta_0)



            else:

                h_m = np.sum(residuals * time_weights[:, None], axis=0)



            h_norm = _min_max_scale(h_m)



            # 3. Atualização do ensemble com regularização por encolhimento (Shrinkage, Géron p. 205)

            F_current = F_current + self.learning_rate * (h_norm - 0.5)

            F_current = np.clip(F_current, 0.0, None)



            # 4. Atualização dos resíduos para o próximo modelo especialista

            residuals = residuals - self.learning_rate * (h_norm[None, :] - 0.5)

            self.staged_scores_.append(_min_max_scale(F_current).copy())



        final_scores = _min_max_scale(F_current)

        dezenas = np.arange(1, n_dezenas + 1)

        self.scores_df_ = pd.DataFrame({

            'dezena': dezenas,

            'escore_cycling_gradboost': np.round(final_scores, 5),

            'escore_acumulado': np.round(F_current, 5)

        }).sort_values('escore_cycling_gradboost', ascending=False).reset_index(drop=True)

        self.top_k_ = sorted(self.scores_df_.head(k)['dezena'].tolist())

        return self



    def predict_top_k(self, k: Optional[int] = None) -> List[int]:

        """

        Retorna as top K dezenas selecionadas pelo Residual Cycling Gradient Boosting.

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

        Iterador sobre as predições do ensemble em cada estágio de treinamento multi-modelo.

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

        """Retorna o DataFrame detalhado com escores do Residual Cycling Gradient Boosting."""

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



class LotteryMomentumEnsembleModel:

    """

    Ensemble Frequência/Atraso (Momentum Multi-Janela com Bootstrap).



    Fundamentação Teórica:

    - O modelo tradicional de momentum avalia a frequência recente através do decaimento temporal

      exponencial w_t = 2^(-(T - t) / lambda).

    - Fixar uma única meia-vida lambda (ex: 100 sorteios) impõe uma escala temporal estática,

      podendo perder acelerações de curtíssimo prazo ou superestimar ruído efêmero.

    - Este Ensemble constrói um comitê heterogêneo de estimadores de momentum combinando:

      1. Escalas temporais múltiplas (multi-scale half-life):

         lambda in [15, 30, 50, 80, 120, 180, 250] (curto, médio e longo prazo).

      2. Amostragem Bootstrap com reposição dos concursos históricos (Breiman 1996; Géron Cap. 7),

         reduzindo a variância das frequências ponderadas.

      3. Agregação por Votação Suave (Soft Voting) com normalização dos escores de momentum.

      4. Avaliação Out-of-Bag (OOB) para validação do comitê.

    """



    def __init__(self,

                 n_estimators: int = 10,

                 half_lives: Optional[List[float]] = None,

                 max_samples: float = 1.0,

                 voting: str = 'soft',

                 oob_score: bool = True,

                 random_state: int = 42):

        if n_estimators <= 0:

            raise ValueError(f"n_estimators deve ser positivo. Recebido: {n_estimators}")

        if not (0.0 < max_samples <= 1.0):

            raise ValueError(f"max_samples deve estar no intervalo (0, 1]. Recebido: {max_samples}")



        self.n_estimators = n_estimators

        self.half_lives = half_lives or [15.0, 30.0, 50.0, 80.0, 120.0, 180.0, 250.0]

        self.max_samples = max_samples

        self.voting = voting

        self.oob_score = oob_score

        self.random_state = random_state



        self.estimators_: List[Dict[str, Any]] = []

        self.oob_score_: float = 0.0

        self.scores_df_: Optional[pd.DataFrame] = None

        self.top_k_: Optional[List[int]] = None



    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None) -> 'LotteryMomentumEnsembleModel':

        """

        Ajusta o ensemble de momentum multi-janela com amostragem bootstrap.

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

            hl = self.half_lives[b % len(self.half_lives)]

            boot_idx = np.sort(rng.choice(n_concursos, size=m_samples, replace=True))

            oob_idx = np.setdiff1d(np.arange(n_concursos), boot_idx)

            X_b = binary_matrix[boot_idx]



            # Frequência ponderada temporalmente para a meia-vida hl

            decay = 0.5 ** (1.0 / hl)

            time_weights = decay ** np.arange(len(boot_idx) - 1, -1, -1)

            sum_w = np.sum(time_weights)

            w_freq = np.sum(X_b * time_weights[:, None], axis=0) / (sum_w if sum_w > 0 else 1.0)



            # Normalização (Z-score e Min-Max)

            std_f = np.std(w_freq)

            z_f = (w_freq - np.mean(w_freq)) / (std_f if std_f > 0 else 1.0)

            score_b = (z_f - np.min(z_f)) / (np.ptp(z_f) if np.ptp(z_f) > 0 else 1.0)

            all_scores.append(score_b)



            top_b = sorted(list(np.argsort(score_b)[::-1][:k] + 1))

            all_top_k.append(top_b)

            self.estimators_.append({

                'estimator_idx': b,

                'half_life': hl,

                'scores': score_b,

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

            'escore_momentum_ensemble': np.round(soft_scores, 5),

            'votos_hard': hard_votes.astype(int)

        }).sort_values('escore_momentum_ensemble', ascending=False).reset_index(drop=True)

        self.top_k_ = sorted(self.scores_df_.head(k)['dezena'].tolist())

        return self



    def predict_top_k(self, k: Optional[int] = None, voting: Optional[str] = None) -> List[int]:

        """

        Retorna as top K dezenas selecionadas pelo ensemble de Momentum.

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

            df_sorted = self.scores_df_.sort_values(['votos_hard', 'escore_momentum_ensemble'], ascending=[False, False])

        else:

            df_sorted = self.scores_df_.sort_values('escore_momentum_ensemble', ascending=False)

        return sorted(df_sorted.head(k)['dezena'].tolist())



    def get_scores_df(self) -> pd.DataFrame:

        """Retorna o DataFrame detalhado com escores do Momentum Ensemble."""

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")

        return self.scores_df_.copy()





class LotteryBalancedEnsembleModel:

    """

    Ensemble Frequência/Atraso (Balanceado Multi-Escala com Bootstrap).



    Fundamentação Teórica:

    - O modelo estatístico balanceado (Método 1) combina a frequência ponderada recente z_freq

      com a pressão do atraso relativo z_delay:

      Score_i = alpha * Z(f_w, i) + (1 - alpha) * Z(R_i)

    - Fixar arbitrariamente alpha = 0.5 e lambda = 100 impõe rigidez na ponderação entre momentum e reversão.

    - Este Ensemble constrói um comitê sistemático variando:

      1. Os fatores de equilíbrio alpha: [0.25, 0.35, 0.50, 0.65, 0.75]

         (alternando entre ênfase em atraso, neutralidade e ênfase em momentum).

      2. As meias-vidas de decaimento temporal lambda: [30, 60, 100, 150, 250].

      3. Amostragem Bootstrap com reposição dos concursos históricos (Breiman 1996; Géron Cap. 7).

      4. Agregação por Votação Suave (Soft Voting) dos escores compostos normalizados.

      5. Avaliação Out-of-Bag (OOB) dos acertos.

    """



    def __init__(self,

                 n_estimators: int = 10,

                 alphas: Optional[List[float]] = None,

                 half_lives: Optional[List[float]] = None,

                 max_samples: float = 1.0,

                 voting: str = 'soft',

                 oob_score: bool = True,

                 random_state: int = 42):

        if n_estimators <= 0:

            raise ValueError(f"n_estimators deve ser positivo. Recebido: {n_estimators}")

        if not (0.0 < max_samples <= 1.0):

            raise ValueError(f"max_samples deve estar no intervalo (0, 1]. Recebido: {max_samples}")



        self.n_estimators = n_estimators

        self.alphas = alphas or [0.25, 0.35, 0.50, 0.65, 0.75]

        self.half_lives = half_lives or [30.0, 60.0, 100.0, 150.0, 250.0]

        self.max_samples = max_samples

        self.voting = voting

        self.oob_score = oob_score

        self.random_state = random_state



        self.estimators_: List[Dict[str, Any]] = []

        self.oob_score_: float = 0.0

        self.scores_df_: Optional[pd.DataFrame] = None

        self.top_k_: Optional[List[int]] = None



    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None) -> 'LotteryBalancedEnsembleModel':

        """

        Ajusta o ensemble balanceado multi-escala com amostragem bootstrap.

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

            alpha_b = self.alphas[b % len(self.alphas)]

            hl_b = self.half_lives[b % len(self.half_lives)]

            boot_idx = np.sort(rng.choice(n_concursos, size=m_samples, replace=True))

            oob_idx = np.setdiff1d(np.arange(n_concursos), boot_idx)

            X_b = binary_matrix[boot_idx]



            m = FrequencyDelayModel(half_life=hl_b, alpha=alpha_b).fit(X_b)

            s_raw = np.zeros(n_dezenas, dtype=float)

            for _, r in m.scores_df_.iterrows():

                s_raw[int(r['dezena']) - 1] = float(r['escore_composto'])



            # Normalização min-max

            score_b = (s_raw - np.min(s_raw)) / (np.ptp(s_raw) if np.ptp(s_raw) > 0 else 1.0)

            all_scores.append(score_b)



            top_b = sorted(list(np.argsort(score_b)[::-1][:k] + 1))

            all_top_k.append(top_b)

            self.estimators_.append({

                'estimator_idx': b,

                'alpha': alpha_b,

                'half_life': hl_b,

                'scores': score_b,

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

            'escore_balanceado_ensemble': np.round(soft_scores, 5),

            'votos_hard': hard_votes.astype(int)

        }).sort_values('escore_balanceado_ensemble', ascending=False).reset_index(drop=True)

        self.top_k_ = sorted(self.scores_df_.head(k)['dezena'].tolist())

        return self



    def predict_top_k(self, k: Optional[int] = None, voting: Optional[str] = None) -> List[int]:

        """

        Retorna as top K dezenas selecionadas pelo ensemble Balanceado.

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

            df_sorted = self.scores_df_.sort_values(['votos_hard', 'escore_balanceado_ensemble'], ascending=[False, False])

        else:

            df_sorted = self.scores_df_.sort_values('escore_balanceado_ensemble', ascending=False)

        return sorted(df_sorted.head(k)['dezena'].tolist())



    def get_scores_df(self) -> pd.DataFrame:

        """Retorna o DataFrame detalhado com escores do Balanced Ensemble."""

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")

        return self.scores_df_.copy()



def _extract_gd_analytical_features(X_hist: np.ndarray, cfg: Dict) -> np.ndarray:

    """

    Extrai a matriz completa de 33 atributos analíticos derivados dos modelos

    estatísticos, bayesianos, geométricos e heurísticos anteriores para alimentar

    os ensembles de Gradiente Descendente (Batch, Mini-Batch e Estocástico).

    Retorna uma matriz de shape (n_dezenas, 33) normalizada no intervalo [0, 1].

    """

    T_curr, N_curr = X_hist.shape

    feats = []

    

    # 1-7: Multi-scale momentum (half-lives 15, 30, 50, 80, 120, 180, 250)

    for hl in [15.0, 30.0, 50.0, 80.0, 120.0, 180.0, 250.0]:

        decay = 0.5 ** (1.0 / hl)

        w = decay ** np.arange(T_curr - 1, -1, -1)

        sw = np.sum(w)

        s = np.sum(X_hist * w[:, None], axis=0) / (sw if sw > 0 else 1.0)

        feats.append(_min_max_scale(s))

        

    # 8-9: Overdue / Delays

    delays = np.zeros(N_curr, dtype=float)

    mean_delays = np.zeros(N_curr, dtype=float)

    for j in range(N_curr):

        idx = np.where(X_hist[:, j] == 1)[0]

        if len(idx) > 0:

            delays[j] = (T_curr - 1) - idx[-1]

            diffs = np.diff(idx)

            mean_delays[j] = np.mean(diffs) if len(diffs) > 0 else (T_curr / len(idx))

        else:

            delays[j] = T_curr

            mean_delays[j] = T_curr

    ratio_delay = delays / np.where(mean_delays > 0, mean_delays, 1.0)

    feats.append(_min_max_scale(delays))

    feats.append(_min_max_scale(ratio_delay))

    

    # 10-14: Balanced multi-alpha

    mom_std = feats[2] # hl=50

    over_std = feats[8]

    for alpha in [0.25, 0.35, 0.50, 0.65, 0.75]:

        b_score = alpha * mom_std + (1.0 - alpha) * over_std

        feats.append(_min_max_scale(b_score))

        

    # 15: Bayes Global

    a0 = cfg.get('alpha_0', 1.0)

    b0 = cfg.get('beta_0', 9.0)

    p_global = (X_hist.sum(axis=0) + a0) / (T_curr + a0 + b0)

    feats.append(_min_max_scale(p_global))

    

    # 16-17: Bayes Adaptive (decay 100, 150)

    for hl in [100.0, 150.0]:

        decay = 0.5 ** (1.0 / hl)

        w = decay ** np.arange(T_curr - 1, -1, -1)

        eff_n = np.sum(w)

        eff_s = np.sum(X_hist * w[:, None], axis=0)

        p_dyn = (eff_s + a0) / (eff_n + a0 + b0)

        feats.append(_min_max_scale(p_dyn))

        

    # 18-20: Gianella group distributions & template affinity

    grp_sz = cfg.get('gianella_group_size', 10)

    grp_counts = np.zeros(N_curr, dtype=float)

    for j in range(N_curr):

        grp_idx = (j // grp_sz)

        grp_start = grp_idx * grp_sz

        grp_end = min(N_curr, grp_start + grp_sz)

        grp_counts[j] = np.sum(X_hist[:, grp_start:grp_end])

    feats.append(_min_max_scale(grp_counts))

    rec_grp = np.sum(X_hist[-min(10, T_curr):, :], axis=0)

    feats.append(_min_max_scale(rec_grp))

    feats.append(_min_max_scale(rec_grp / np.where(grp_counts > 0, grp_counts, 1.0)))

    

    # 21-24: Moving Averages (Lags 5, 10, 25, 50)

    for win in [5, 10, 25, 50]:

        w_act = min(win, T_curr)

        ma = np.mean(X_hist[-w_act:, :], axis=0)

        feats.append(_min_max_scale(ma))

        

    # 25-28: Hybrids / Cross Interactions

    feats.append(_min_max_scale(feats[14] * feats[17]))

    feats.append(_min_max_scale(feats[14] * feats[2]))

    feats.append(_min_max_scale(feats[17] * feats[2]))

    feats.append(_min_max_scale(feats[16] * feats[20]))

    

    # 29-33: Residual / Consensus proxies

    mean_all = np.mean(feats[:10], axis=0)

    feats.append(_min_max_scale(mean_all))

    feats.append(_min_max_scale(np.median(feats[:15], axis=0)))

    feats.append(_min_max_scale(feats[2] - feats[14]))

    feats.append(_min_max_scale(feats[20] - feats[23]))

    feats.append(_min_max_scale(np.std(feats[:10], axis=0)))

    

    return np.column_stack(feats)





class LotteryBatchGDEnsembleModel:

    """

    Ensemble Gradiente Descendente Batch (Batch Gradient Descent Ensemble).

    Fundamentado em Aurélien Géron (2019), Hands-On Machine Learning (2ª Ed.), Capítulo 4:

    'Batch Gradient Descent' (pp. 118-122).



    Fundamentação Teórica:

    - O Batch Gradient Descent utiliza a TOTALIDADE dos concursos do conjunto de treinamento

      em cada época para computar o gradiente exato da função de perda (Log-Loss / Entropia Cruzada Binária)

      com regularização L2 (Ridge penalty):

      grad_w = (1 / M) * X_train.T @ (p - y) + l2_reg * w

      grad_b = (1 / M) * sum(p - y)

    - Garante convergência suave, monótona e direta para o mínimo global da superfície de perda convexa.

    - Utiliza o espaço de atributos analíticos dos 33 métodos anteriores como matriz de meta-preditores.

    """

    def __init__(self,

                 n_epochs: int = 150,

                 learning_rate: float = 0.1,

                 l2_reg: float = 1e-3,

                 n_samples_history: int = 15,

                 random_state: int = 42):

        self.n_epochs = n_epochs

        self.learning_rate = learning_rate

        self.l2_reg = l2_reg

        self.n_samples_history = n_samples_history

        self.random_state = random_state



        self.weights_: Optional[np.ndarray] = None

        self.bias_: float = 0.0

        self.loss_history_: List[float] = []

        self.scores_df_: Optional[pd.DataFrame] = None

        self.top_k_: Optional[List[int]] = None



    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None, base_features: Optional[np.ndarray] = None) -> 'LotteryBatchGDEnsembleModel':

        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:

            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        cfg = cfg or {}

        T, N = binary_matrix.shape

        k = cfg.get('k', 6)



        sample_indices = np.linspace(max(10, T - 45), T - 1, num=min(self.n_samples_history, max(3, T - 10)), dtype=int)

        X_train_rows, y_train_rows = [], []

        for idx in sample_indices:

            feats_idx = _extract_gd_analytical_features(binary_matrix[:idx], cfg)

            X_train_rows.append(feats_idx)

            y_train_rows.append(binary_matrix[idx])



        X_train = np.vstack(X_train_rows)

        y_train = np.concatenate(y_train_rows).astype(float)

        M_samples, D_feats = X_train.shape



        rng = np.random.RandomState(self.random_state)

        w = rng.normal(0, 0.01, size=D_feats)

        b = 0.0

        self.loss_history_ = []



        for _ in range(self.n_epochs):

            z = np.clip(X_train @ w + b, -25.0, 25.0)

            p = 1.0 / (1.0 + np.exp(-z))

            err = p - y_train

            loss = -np.mean(y_train * np.log(p + 1e-12) + (1.0 - y_train) * np.log(1.0 - p + 1e-12)) + 0.5 * self.l2_reg * np.sum(w ** 2)

            self.loss_history_.append(float(loss))

            grad_w = (X_train.T @ err) / M_samples + self.l2_reg * w

            grad_b = float(np.mean(err))

            w -= self.learning_rate * grad_w

            b -= self.learning_rate * grad_b



        self.weights_ = w

        self.bias_ = b



        X_target = base_features if base_features is not None else _extract_gd_analytical_features(binary_matrix, cfg)

        z_target = np.clip(X_target @ w + b, -25.0, 25.0)

        p_target = 1.0 / (1.0 + np.exp(-z_target))

        scores_final = _min_max_scale(p_target)



        dezenas = np.arange(1, N + 1)

        self.scores_df_ = pd.DataFrame({

            'dezena': dezenas,

            'escore_batch_gd': np.round(scores_final, 5),

            'probabilidade_calibrada': np.round(p_target, 5),

            'logit_predito': np.round(z_target, 5)

        }).sort_values('escore_batch_gd', ascending=False).reset_index(drop=True)

        self.top_k_ = sorted([int(x) for x in self.scores_df_.head(k)['dezena'].tolist()])

        return self



    def predict_top_k(self, k: Optional[int] = None) -> List[int]:

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")

        if k is None:

            k = len(self.top_k_) if self.top_k_ is not None else 6

        if k <= 0:

            return []

        k = min(k, len(self.scores_df_))

        return sorted([int(x) for x in self.scores_df_.head(k)['dezena'].tolist()])



    def get_scores_df(self) -> pd.DataFrame:

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")

        return self.scores_df_.copy()





class LotteryMiniBatchGDEnsembleModel:

    """

    Ensemble Gradiente Descendente Mini-Batch (Mini-Batch Gradient Descent Ensemble).

    Fundamentado em Aurélien Géron (2019), Hands-On Machine Learning (2ª Ed.), Capítulo 4:

    'Mini-batch Gradient Descent' (pp. 124-125).



    Fundamentação Teórica:

    - O Mini-Batch Gradient Descent computa os gradientes sobre pequenos lotes aleatórios

      de instâncias (batch_size tipicamente 16 ou 32).

    - Vantagens Teóricas:

      1. Ganho de desempenho via vetorização de operações matriciais em hardware moderno.

      2. Trajetória com ruído estocástico controlado, permitindo escapar de mínimos locais rasos

         sem a instabilidade extrema do SGD puro.

      3. Regularização intrínseca contra memorização de ruído amostral de concursos efêmeros.

    """

    def __init__(self,

                 n_epochs: int = 40,

                 batch_size: int = 32,

                 learning_rate: float = 0.05,

                 l2_reg: float = 1e-3,

                 n_samples_history: int = 15,

                 random_state: int = 42):

        self.n_epochs = n_epochs

        self.batch_size = batch_size

        self.learning_rate = learning_rate

        self.l2_reg = l2_reg

        self.n_samples_history = n_samples_history

        self.random_state = random_state



        self.weights_: Optional[np.ndarray] = None

        self.bias_: float = 0.0

        self.loss_history_: List[float] = []

        self.scores_df_: Optional[pd.DataFrame] = None

        self.top_k_: Optional[List[int]] = None



    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None, base_features: Optional[np.ndarray] = None) -> 'LotteryMiniBatchGDEnsembleModel':

        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:

            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        cfg = cfg or {}

        T, N = binary_matrix.shape

        k = cfg.get('k', 6)



        sample_indices = np.linspace(max(10, T - 45), T - 1, num=min(self.n_samples_history, max(3, T - 10)), dtype=int)

        X_train_rows, y_train_rows = [], []

        for idx in sample_indices:

            feats_idx = _extract_gd_analytical_features(binary_matrix[:idx], cfg)

            X_train_rows.append(feats_idx)

            y_train_rows.append(binary_matrix[idx])



        X_train = np.vstack(X_train_rows)

        y_train = np.concatenate(y_train_rows).astype(float)

        M_samples, D_feats = X_train.shape



        rng = np.random.RandomState(self.random_state)

        w = rng.normal(0, 0.01, size=D_feats)

        b = 0.0

        self.loss_history_ = []



        for _ in range(self.n_epochs):

            perm = rng.permutation(M_samples)

            epoch_loss = 0.0

            n_batches = 0

            for start_i in range(0, M_samples, self.batch_size):

                b_idx = perm[start_i:start_i + self.batch_size]

                X_b, y_b = X_train[b_idx], y_train[b_idx]

                z_b = np.clip(X_b @ w + b, -25.0, 25.0)

                p_b = 1.0 / (1.0 + np.exp(-z_b))

                err_b = p_b - y_b

                grad_w = (X_b.T @ err_b) / len(b_idx) + self.l2_reg * w

                grad_b = float(np.mean(err_b))

                w -= self.learning_rate * grad_w

                b -= self.learning_rate * grad_b

                epoch_loss += -np.mean(y_b * np.log(p_b + 1e-12) + (1.0 - y_b) * np.log(1.0 - p_b + 1e-12))

                n_batches += 1

            self.loss_history_.append(float(epoch_loss / max(1, n_batches)))



        self.weights_ = w

        self.bias_ = b



        X_target = base_features if base_features is not None else _extract_gd_analytical_features(binary_matrix, cfg)

        z_target = np.clip(X_target @ w + b, -25.0, 25.0)

        p_target = 1.0 / (1.0 + np.exp(-z_target))

        scores_final = _min_max_scale(p_target)



        dezenas = np.arange(1, N + 1)

        self.scores_df_ = pd.DataFrame({

            'dezena': dezenas,

            'escore_minibatch_gd': np.round(scores_final, 5),

            'probabilidade_calibrada': np.round(p_target, 5),

            'logit_predito': np.round(z_target, 5)

        }).sort_values('escore_minibatch_gd', ascending=False).reset_index(drop=True)

        self.top_k_ = sorted([int(x) for x in self.scores_df_.head(k)['dezena'].tolist()])

        return self



    def predict_top_k(self, k: Optional[int] = None) -> List[int]:

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")

        if k is None:

            k = len(self.top_k_) if self.top_k_ is not None else 6

        if k <= 0:

            return []

        k = min(k, len(self.scores_df_))

        return sorted([int(x) for x in self.scores_df_.head(k)['dezena'].tolist()])



    def get_scores_df(self) -> pd.DataFrame:

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() primeiro.")

        return self.scores_df_.copy()





class LotteryStochasticGDEnsembleModel:

    """

    Ensemble Gradiente Descendente Estocástico (Stochastic Gradient Descent Ensemble / Online SGD).

    Fundamentado em Aurélien Géron (2019), Hands-On Machine Learning (2ª Ed.), Capítulo 4:

    'Stochastic Gradient Descent' (pp. 122-124).



    Fundamentação Teórica:

    - O SGD escolhe uma única instância aleatória em cada iteração e calcula o gradiente baseado

      apenas nessa observação individual:

      grad_w_i = (p_i - y_i) * x_i + l2_reg * w

      grad_b_i = p_i - y_i

    - Utiliza cronograma de taxa de aprendizado decrescente (Learning Schedule, Géron Eq. 4-2):

      eta_t = eta_0 / (1 + decay * t)

    - Adaptação imediata a choques de regime recentes, com capacidade intrínseca de escapar

      de ótimos locais estáticos.

    """

    def __init__(self,

                 n_epochs: int = 15,

                 learning_rate_0: float = 0.05,

                 decay: float = 0.005,

                 l2_reg: float = 1e-3,

                 n_samples_history: int = 15,

                 random_state: int = 42):

        self.n_epochs = n_epochs

        self.learning_rate_0 = learning_rate_0

        self.decay = decay

        self.l2_reg = l2_reg

        self.n_samples_history = n_samples_history

        self.random_state = random_state



        self.weights_: Optional[np.ndarray] = None

        self.bias_: float = 0.0

        self.loss_history_: List[float] = []

        self.scores_df_: Optional[pd.DataFrame] = None

        self.top_k_: Optional[List[int]] = None



    def fit(self, binary_matrix: np.ndarray, cfg: Optional[Dict] = None, base_features: Optional[np.ndarray] = None) -> 'LotteryStochasticGDEnsembleModel':

        if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:

            raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")

        cfg = cfg or {}

        T, N = binary_matrix.shape

        k = cfg.get('k', 6)



        sample_indices = np.linspace(max(10, T - 45), T - 1, num=min(self.n_samples_history, max(3, T - 10)), dtype=int)

        X_train_rows, y_train_rows = [], []

        for idx in sample_indices:

            feats_idx = _extract_gd_analytical_features(binary_matrix[:idx], cfg)

            X_train_rows.append(feats_idx)

            y_train_rows.append(binary_matrix[idx])



        X_train = np.vstack(X_train_rows)

        y_train = np.concatenate(y_train_rows).astype(float)

        M_samples, D_feats = X_train.shape



        rng = np.random.RandomState(self.random_state)

        w = rng.normal(0, 0.01, size=D_feats)

        b = 0.0

        self.loss_history_ = []



        t_step = 0

        for _ in range(self.n_epochs):

            perm = rng.permutation(M_samples)

            epoch_loss = 0.0

            for idx_i in perm:

                x_i = X_train[idx_i]

                y_i = y_train[idx_i]

                eta = self.learning_rate_0 / (1.0 + self.decay * t_step)

                t_step += 1

                z_i = float(np.clip(np.dot(x_i, w) + b, -25.0, 25.0))

                p_i = 1.0 / (1.0 + np.exp(-z_i))

                err_i = p_i - y_i

                w -= eta * (err_i * x_i + self.l2_reg * w)

                b -= eta * err_i

                epoch_loss += -(y_i * np.log(p_i + 1e-12) + (1.0 - y_i) * np.log(1.0 - p_i + 1e-12))

            self.loss_history_.append(float(epoch_loss / M_samples))



        self.weights_ = w

        self.bias_ = b



        X_target = base_features if base_features is not None else _extract_gd_analytical_features(binary_matrix, cfg)

        z_target = np.clip(X_target @ w + b, -25.0, 25.0)

        p_target = 1.0 / (1.0 + np.exp(-z_target))

        scores_final = _min_max_scale(p_target)



        dezenas = np.arange(1, N + 1)

        self.scores_df_ = pd.DataFrame({

            'dezena': dezenas,

            'escore_stochastic_gd': np.round(scores_final, 5),

            'probabilidade_calibrada': np.round(p_target, 5),

            'logit_predito': np.round(z_target, 5)

        }).sort_values('escore_stochastic_gd', ascending=False).reset_index(drop=True)

        self.top_k_ = sorted([int(x) for x in self.scores_df_.head(k)['dezena'].tolist()])

        return self



    def predict_top_k(self, k: Optional[int] = None) -> List[int]:

        if self.scores_df_ is None:

            raise RuntimeError("O modelo precisa ser ajustado com fit() antes de gerar predições.")

        if k is None:

            k = len(self.top_k_) if self.top_k_ is not None else 6

        if k <= 0:

            return []

        k = min(k, len(self.scores_df_))

        return sorted([int(x) for x in self.scores_df_.head(k)['dezena'].tolist()])



    def get_scores_df(self) -> pd.DataFrame:

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





def predict_stochastic_gradient_boosting(binary_matrix: np.ndarray,

                                         cfg: Dict,

                                         k: Optional[int] = None,

                                         n_estimators: int = 10,

                                         learning_rate: float = 0.5,

                                         subsample: float = 0.7,

                                         random_state: int = 42) -> List[int]:

    """Predição direta das top K dezenas via Ensemble Stochastic Gradient Boosting (Géron p. 207)."""

    k_actual = cfg['k'] if k is None else k

    model = LotteryStochasticGradientBoostingModel(n_estimators=n_estimators,

                                                   learning_rate=learning_rate,

                                                   subsample=subsample,

                                                   random_state=random_state)

    model.fit(binary_matrix, cfg=cfg)

    return model.predict_top_k(k_actual)





def predict_logit_gradient_boosting(binary_matrix: np.ndarray,

                                    cfg: Dict,

                                    k: Optional[int] = None,

                                    n_estimators: int = 10,

                                    learning_rate: float = 0.3,

                                    random_state: int = 42) -> List[int]:

    """Predição direta das top K dezenas via Ensemble Logit / Binomial Deviance Gradient Boosting."""

    k_actual = cfg['k'] if k is None else k

    model = LotteryLogitGradientBoostingModel(n_estimators=n_estimators,

                                              learning_rate=learning_rate,

                                              random_state=random_state)

    model.fit(binary_matrix, cfg=cfg)

    return model.predict_top_k(k_actual)





predict_binomial_deviance_gradient_boosting = predict_logit_gradient_boosting





def predict_bagged_logit_gb_ensemble(binary_matrix: np.ndarray,

                                     cfg: Dict,

                                     k: Optional[int] = None,

                                     n_estimators: int = 10,

                                     base_n_iterations: int = 10,

                                     subsample: float = 0.85,

                                     random_state: int = 42) -> List[int]:

    """Predição direta das top K dezenas via Meta-Ensemble Bagged Logit Gradient Boosting."""

    k_actual = cfg['k'] if k is None else k

    model = LotteryBaggedLogitGBEnsembleModel(

        n_estimators=n_estimators,

        base_n_iterations=base_n_iterations,

        subsample=subsample,

        random_state=random_state

    )

    model.fit(binary_matrix, cfg=cfg)

    return model.predict_top_k(k_actual)







def predict_residual_cycling_gradient_boosting(binary_matrix: np.ndarray,

                                               cfg: Dict,

                                               k: Optional[int] = None,

                                               n_estimators: int = 8,

                                               learning_rate: float = 0.4,

                                               random_state: int = 42) -> List[int]:

    """Predição direta das top K dezenas via Ensemble Residual Cycling Gradient Boosting."""

    k_actual = cfg['k'] if k is None else k

    model = LotteryResidualCyclingGradientBoostingModel(n_estimators=n_estimators,

                                                        learning_rate=learning_rate,

                                                        random_state=random_state)

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



def predict_momentum_ensemble(binary_matrix: np.ndarray,

                              cfg: Dict,

                              k: Optional[int] = None,

                              n_estimators: int = 10,

                              voting: str = 'soft',

                              random_state: int = 42) -> List[int]:

    """Predição direta das top K dezenas via Ensemble Frequência/Atraso Momentum."""

    k_actual = cfg['k'] if k is None else k

    model = LotteryMomentumEnsembleModel(n_estimators=n_estimators, voting=voting, random_state=random_state)

    model.fit(binary_matrix, cfg=cfg)

    return model.predict_top_k(k_actual)





def predict_balanced_ensemble(binary_matrix: np.ndarray,

                              cfg: Dict,

                              k: Optional[int] = None,

                              n_estimators: int = 10,

                              voting: str = 'soft',

                              random_state: int = 42) -> List[int]:

    """Predição direta das top K dezenas via Ensemble Frequência/Atraso Balanceado."""

    k_actual = cfg['k'] if k is None else k

    model = LotteryBalancedEnsembleModel(n_estimators=n_estimators, voting=voting, random_state=random_state)

    model.fit(binary_matrix, cfg=cfg)

    return model.predict_top_k(k_actual)



def predict_batch_gd_ensemble(binary_matrix: np.ndarray,

                              cfg: Dict,

                              k: Optional[int] = None,

                              n_epochs: int = 150,

                              learning_rate: float = 0.1,

                              l2_reg: float = 1e-3,

                              random_state: int = 42) -> List[int]:

    """Predição direta das top K dezenas via Ensemble Batch Gradient Descent."""

    k_actual = cfg['k'] if k is None else k

    model = LotteryBatchGDEnsembleModel(n_epochs=n_epochs, learning_rate=learning_rate,

                                        l2_reg=l2_reg, random_state=random_state)

    model.fit(binary_matrix, cfg=cfg)

    return model.predict_top_k(k_actual)





def predict_minibatch_gd_ensemble(binary_matrix: np.ndarray,

                                  cfg: Dict,

                                  k: Optional[int] = None,

                                  n_epochs: int = 40,

                                  batch_size: int = 32,

                                  learning_rate: float = 0.05,

                                  l2_reg: float = 1e-3,

                                  random_state: int = 42) -> List[int]:

    """Predição direta das top K dezenas via Ensemble Mini-Batch Gradient Descent."""

    k_actual = cfg['k'] if k is None else k

    model = LotteryMiniBatchGDEnsembleModel(n_epochs=n_epochs, batch_size=batch_size,

                                            learning_rate=learning_rate, l2_reg=l2_reg,

                                            random_state=random_state)

    model.fit(binary_matrix, cfg=cfg)

    return model.predict_top_k(k_actual)





def predict_stochastic_gd_ensemble(binary_matrix: np.ndarray,

                                   cfg: Dict,

                                   k: Optional[int] = None,

                                   n_epochs: int = 15,

                                   learning_rate_0: float = 0.05,

                                   decay: float = 0.005,

                                   l2_reg: float = 1e-3,

                                   random_state: int = 42) -> List[int]:

    """Predição direta das top K dezenas via Ensemble Stochastic Gradient Descent (Online SGD)."""

    k_actual = cfg['k'] if k is None else k

    model = LotteryStochasticGDEnsembleModel(n_epochs=n_epochs, learning_rate_0=learning_rate_0,

                                             decay=decay, l2_reg=l2_reg, random_state=random_state)

    model.fit(binary_matrix, cfg=cfg)

    return model.predict_top_k(k_actual)





def predict_dual_logit_gb_ensemble(binary_matrix: np.ndarray,

                                   cfg: Dict,

                                   k: Optional[int] = None,

                                   n_estimators_bagged: int = 10,

                                   base_n_iterations: int = 10,

                                   weight_base: float = 0.45,

                                   learning_rate_base: float = 0.3,

                                   subsample: float = 0.85,

                                   random_state: int = 42) -> List[int]:

    """Predição direta das top K dezenas via Meta-Ensemble Dual Logit Gradient Boosting (Método 34)."""

    k_actual = cfg['k'] if k is None else k

    model = LotteryDualLogitGBEnsembleModel(

        n_estimators_bagged=n_estimators_bagged,

        base_n_iterations=base_n_iterations,

        weight_base=weight_base,

        learning_rate_base=learning_rate_base,

        subsample=subsample,

        random_state=random_state

    )

    model.fit(binary_matrix, cfg=cfg)

    return model.predict_top_k(k_actual)


def predict_pretrained_meta_committee(binary_matrix: np.ndarray,
                                    cfg: Dict,
                                    k: Optional[int] = None,
                                    n_committees: int = 100,
                                    subspace_ratio: float = 0.65,
                                    include_sub_estimators: bool = True,
                                    dirichlet_alpha: float = 1.5,
                                    voting: str = 'hybrid',
                                    trained_ensembles: Optional[Dict[str, Any]] = None,
                                    random_state: int = 42) -> List[int]:
    """Predição direta das top K dezenas via Meta-Comitê de Ensembles Pré-Treinados (Método 35, padrão M=100)."""
    k_actual = cfg['k'] if k is None else k
    model = LotteryPretrainedMetaCommitteeModel(
        n_committees=n_committees,
        subspace_ratio=subspace_ratio,
        include_sub_estimators=include_sub_estimators,
        dirichlet_alpha=dirichlet_alpha,
        voting=voting,
        random_state=random_state,
    )
    model.fit(binary_matrix, cfg=cfg, trained_ensembles=trained_ensembles)
    return model.predict_top_k(k_actual)


def predict_deep_ensemble_committee(binary_matrix: np.ndarray,
                                    cfg: Dict,
                                    k: Optional[int] = None,
                                    n_committees: int = 100,
                                    max_samples: float = 0.85,
                                    anchor_weight: float = 0.30,
                                    weight_by_oob: bool = True,
                                    voting: str = 'hybrid',
                                    trained_ensembles: Optional[Dict[str, Any]] = None,
                                    random_state: int = 42) -> List[int]:
    """Predição direta das top K dezenas via Comitê Profundo Multi-Família Re-Estimado (Método 36, padrão M=100)."""
    k_actual = cfg['k'] if k is None else k
    model = LotteryDeepEnsembleCommitteeModel(
        n_committees=n_committees,
        max_samples=max_samples,
        anchor_weight=anchor_weight,
        weight_by_oob=weight_by_oob,
        voting=voting,
        random_state=random_state,
    )
    model.fit(binary_matrix, cfg=cfg, trained_ensembles=trained_ensembles)
    return model.predict_top_k(k_actual)




def predict_gianella_committee(binary_matrix: np.ndarray,
                               cfg: Dict,
                               k: Optional[int] = None,
                               n_committees: int = 100,
                               top_templates: int = 3,
                               shrinkage_alpha: float = 0.25,
                               subspace_ratio: float = 0.65,
                               use_structural_filter: bool = True,
                               voting: str = 'hybrid',
                               trained_ensembles: Optional[Dict[str, Any]] = None,
                               base_signals: Optional[Dict[str, Any]] = None,
                               random_state: int = 42) -> List[int]:
    """Predição direta das top K dezenas via Comitê de Gianella (Método 37, padrão M=100)."""
    k_actual = cfg['k'] if k is None else k
    model = LotteryGianellaCommitteeModel(
        n_committees=n_committees,
        top_templates=top_templates,
        shrinkage_alpha=shrinkage_alpha,
        subspace_ratio=subspace_ratio,
        use_structural_filter=use_structural_filter,
        voting=voting,
        random_state=random_state,
    )
    model.fit(binary_matrix, cfg=cfg, trained_ensembles=trained_ensembles, base_signals=base_signals)
    return model.predict_top_k(k_actual)


def compare_committee_sizes(binary_matrix: np.ndarray,
                            cfg: Dict,
                            sizes: Tuple[int, ...] = (10, 50, 100),
                            trained_ensembles: Optional[Dict[str, Any]] = None,
                            random_state: int = 42) -> pd.DataFrame:
    """
    Compara empiricamente diferentes tamanhos de comitê (ex.: M=10, M=50, M=100)
    para os dois Meta-Ensembles (Meta-Comitê Pré-Treinado e Comitê Profundo Multi-Família),
    reportando score OOB, erro padrão do escore (SE), estabilidade Jaccard e tempo de treino.
    """
    import time
    if not isinstance(binary_matrix, np.ndarray) or binary_matrix.ndim != 2 or binary_matrix.shape[0] == 0:
        raise ValueError("binary_matrix deve ser uma matriz numpy 2D com pelo menos 1 concurso.")
    sizes_list = [int(s) for s in sizes]
    if not sizes_list or any(s <= 0 for s in sizes_list):
        raise ValueError(f"sizes deve conter ao menos um inteiro positivo (> 0). Recebido: {sizes}")

    k = int(cfg.get('k', 6))
    rows = []

    if trained_ensembles is None:
        m_temp = LotteryPretrainedMetaCommitteeModel(n_committees=10, random_state=random_state)
        trained_ensembles = m_temp._fit_default_base_ensembles(binary_matrix, cfg)

    for m_size in sizes_list:
        t0 = time.perf_counter()
        m_meta = LotteryPretrainedMetaCommitteeModel(
            n_committees=m_size, random_state=random_state
        ).fit(binary_matrix, cfg=cfg, trained_ensembles=trained_ensembles)
        dt_meta = time.perf_counter() - t0
        tail_jacc_meta = float(np.mean([c['top_k_jaccard'] for c in m_meta.convergence_curve_[-min(10, len(m_meta.convergence_curve_)):]]))

        rows.append({
            'Arquitetura': 'Meta-Comitê Pré-Treinado',
            'n_committees': int(m_size),
            'top_k': m_meta.predict_top_k(k),
            'oob_score': np.nan,
            'mean_se': round(m_meta.mean_se_, 6),
            'tail_jaccard_stability': round(tail_jacc_meta, 4),
            'fit_time_s': round(dt_meta, 4),
        })

        t1 = time.perf_counter()
        m_deep = LotteryDeepEnsembleCommitteeModel(
            n_committees=m_size, random_state=random_state
        ).fit(binary_matrix, cfg=cfg, trained_ensembles=trained_ensembles)
        dt_deep = time.perf_counter() - t1
        tail_jacc_deep = float(np.mean([c['top_k_jaccard'] for c in m_deep.convergence_curve_[-min(10, len(m_deep.convergence_curve_)):]]))

        rows.append({
            'Arquitetura': 'Comitê Profundo Multi-Família',
            'n_committees': int(m_size),
            'top_k': m_deep.predict_top_k(k),
            'oob_score': round(m_deep.oob_score_, 4),
            'mean_se': round(m_deep.mean_se_, 6),
            'tail_jaccard_stability': round(tail_jacc_deep, 4),
            'fit_time_s': round(dt_deep, 4),
        })

    return pd.DataFrame(rows)









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



    m_stoch_gb = LotteryStochasticGradientBoostingModel(n_estimators=10, learning_rate=0.5, subsample=0.7, random_state=42)

    m_stoch_gb.fit(binary_matrix, cfg=cfg)

    pred_stoch_gb = m_stoch_gb.predict_top_k(k)



    m_logit_gb = LotteryLogitGradientBoostingModel(n_estimators=10, learning_rate=0.3, random_state=42)

    m_logit_gb.fit(binary_matrix, cfg=cfg)

    pred_logit_gb = m_logit_gb.predict_top_k(k)



    m_cycle_gb = LotteryResidualCyclingGradientBoostingModel(n_estimators=8, learning_rate=0.4, random_state=42)

    m_cycle_gb.fit(binary_matrix, cfg=cfg)

    pred_cycle_gb = m_cycle_gb.predict_top_k(k)



    m_stacking = LotteryStackingModel(holdout_ratio=0.25, random_state=42)

    m_stacking.fit(binary_matrix, cfg=cfg)

    pred_stacking = m_stacking.predict_top_k(k)



    m_mom_ens = LotteryMomentumEnsembleModel(n_estimators=10, random_state=42)

    m_mom_ens.fit(binary_matrix, cfg=cfg)

    pred_mom_ens = m_mom_ens.predict_top_k(k)



    m_bal_ens = LotteryBalancedEnsembleModel(n_estimators=10, random_state=42)

    m_bal_ens.fit(binary_matrix, cfg=cfg)

    pred_bal_ens = m_bal_ens.predict_top_k(k)



    m_batch_gd = LotteryBatchGDEnsembleModel(n_epochs=150, learning_rate=0.1, random_state=42)

    m_batch_gd.fit(binary_matrix, cfg=cfg)

    pred_batch_gd = m_batch_gd.predict_top_k(k)



    m_minibatch_gd = LotteryMiniBatchGDEnsembleModel(n_epochs=40, batch_size=32, learning_rate=0.05, random_state=42)

    m_minibatch_gd.fit(binary_matrix, cfg=cfg)

    pred_minibatch_gd = m_minibatch_gd.predict_top_k(k)



    m_stoch_gd = LotteryStochasticGDEnsembleModel(n_epochs=15, learning_rate_0=0.05, decay=0.005, random_state=42)

    m_stoch_gd.fit(binary_matrix, cfg=cfg)

    pred_stoch_gd = m_stoch_gd.predict_top_k(k)



    m_bagged_logit_gb = LotteryBaggedLogitGBEnsembleModel(n_estimators=10, random_state=42)

    m_bagged_logit_gb.fit(binary_matrix, cfg=cfg)

    pred_bagged_logit_gb = m_bagged_logit_gb.predict_top_k(k)



    m_dual_logit_gb = LotteryDualLogitGBEnsembleModel(n_estimators_bagged=10, weight_base=0.45, random_state=42)

    m_dual_logit_gb.fit(binary_matrix, cfg=cfg, base_model=m_logit_gb, bagged_model=m_bagged_logit_gb)

    pred_dual_logit_gb = m_dual_logit_gb.predict_top_k(k)

    # 7. Novos Meta-Ensembles de Alto Comitê (M=100 padrão) baseados nos Ensembles já treinados
    trained_ensembles_dict = {
        'bagging': m_bagging,
        'pasting': m_pasting,
        'adaboost': m_adaboost,
        'gradboost': m_gradboost,
        'stoch_gb': m_stoch_gb,
        'logit_gb': m_logit_gb,
        'cycle_gb': m_cycle_gb,
        'stacking': m_stacking,
        'mom_ens': m_mom_ens,
        'bal_ens': m_bal_ens,
        'batch_gd': m_batch_gd,
        'minibatch_gd': m_minibatch_gd,
        'stoch_gd': m_stoch_gd,
        'bagged_logit_gb': m_bagged_logit_gb,
        'dual_logit_gb': m_dual_logit_gb,
    }

    m_meta_committee_100 = LotteryPretrainedMetaCommitteeModel(n_committees=100, random_state=42)
    m_meta_committee_100.fit(binary_matrix, cfg=cfg, trained_ensembles=trained_ensembles_dict)
    pred_meta_committee_100 = m_meta_committee_100.predict_top_k(k)

    m_deep_committee_100 = LotteryDeepEnsembleCommitteeModel(n_committees=100, random_state=42)
    m_deep_committee_100.fit(binary_matrix, cfg=cfg, trained_ensembles=trained_ensembles_dict)
    pred_deep_committee_100 = m_deep_committee_100.predict_top_k(k)

    base_signals_gianella = {
        'm1': m1,
        'm2': m2,
        'm3_static': m3_static,
        'm3_dyn': m3_dyn,
        'gianella_model': m4,
    }
    m_gianella_committee_100 = LotteryGianellaCommitteeModel(n_committees=100, random_state=42)
    m_gianella_committee_100.fit(
        binary_matrix, cfg=cfg, trained_ensembles=trained_ensembles_dict, base_signals=base_signals_gianella
    )
    pred_gianella_committee_100 = m_gianella_committee_100.predict_top_k(k)



    predictions_ensemble = {

        'Ensemble Bagging': pred_bagging,

        'Ensemble Pasting': pred_pasting,

        'Ensemble AdaBoost': pred_adaboost,

        'Ensemble Gradient Boosting': pred_gradboost,

        'Ensemble Stochastic Gradient Boosting': pred_stoch_gb,

        'Ensemble Logit Gradient Boosting': pred_logit_gb,

        'Ensemble Residual Cycling Gradient Boosting': pred_cycle_gb,

        'Ensemble Stacking': pred_stacking,

        'Ensemble Frequência/Atraso Momentum': pred_mom_ens,

        'Ensemble Frequência/Atraso Balanceado': pred_bal_ens,

        'Ensemble Batch Gradient Descent': pred_batch_gd,

        'Ensemble Mini-Batch Gradient Descent': pred_minibatch_gd,

        'Ensemble Stochastic Gradient Descent': pred_stoch_gd,

        'Ensemble Bagged Logit Gradient Boosting': pred_bagged_logit_gb,

        'Ensemble Dual Logit Gradient Boosting': pred_dual_logit_gb,

        'Ensemble Meta-Comitê Pré-Treinado (M=100)': pred_meta_committee_100,

        'Ensemble Comitê Profundo Multi-Família (M=100)': pred_deep_committee_100,
        'Ensemble Comitê de Gianella (M=100)': pred_gianella_committee_100

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

        predictions['Metodo 24 (Ensemble Stochastic Gradient Boosting - Subsample)'] = pred_stoch_gb

        predictions['Metodo 25 (Ensemble Logit Gradient Boosting - Binomial Deviance)'] = pred_logit_gb

        predictions['Metodo 26 (Ensemble Residual Cycling Gradient Boosting - Multi-Model Residual)'] = pred_cycle_gb

        predictions['Metodo 27 (Ensemble Stacking - Stacked Generalization)'] = pred_stacking

        predictions['Metodo 28 (Ensemble Frequencia/Atraso Momentum - Multi-Janela)'] = pred_mom_ens

        predictions['Metodo 29 (Ensemble Frequencia/Atraso Balanceado - Multi-Escala)'] = pred_bal_ens

        predictions['Metodo 30 (Ensemble Batch Gradient Descent - Full Batch GD)'] = pred_batch_gd

        predictions['Metodo 31 (Ensemble Mini-Batch Gradient Descent - Mini-Batch GD)'] = pred_minibatch_gd

        predictions['Metodo 32 (Ensemble Stochastic Gradient Descent - Online SGD)'] = pred_stoch_gd

        predictions['Metodo 33 (Ensemble Bagged Logit Gradient Boosting - Meta-Ensemble)'] = pred_bagged_logit_gb
        predictions['Metodo 34 (Ensemble Dual Logit Gradient Boosting - Fusão Hierárquica)'] = pred_dual_logit_gb

        predictions['Metodo 35 (Ensemble Meta-Comitê Pré-Treinado - M=100)'] = pred_meta_committee_100

        predictions['Metodo 36 (Ensemble Comitê Profundo Multi-Família - M=100)'] = pred_deep_committee_100
        predictions['Metodo 37 (Ensemble Comite de Gianella - M=100)'] = pred_gianella_committee_100



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

        'm_stoch_gb': m_stoch_gb,

        'm_logit_gb': m_logit_gb,

        'm_cycle_gb': m_cycle_gb,

        'm_stacking': m_stacking,

        'm_mom_ens': m_mom_ens,

        'm_bal_ens': m_bal_ens,

        'm_batch_gd': m_batch_gd,

        'm_minibatch_gd': m_minibatch_gd,

        'm_stoch_gd': m_stoch_gd,

        'm_bagged_logit_gb': m_bagged_logit_gb,

        'm_dual_logit_gb': m_dual_logit_gb,

        'm_meta_committee_100': m_meta_committee_100,

        'm_deep_committee_100': m_deep_committee_100,
        'm_gianella_committee_100': m_gianella_committee_100,

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

