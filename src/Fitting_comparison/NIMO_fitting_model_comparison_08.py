"""
NIMO_fitting_model_comparison.py

気泡径分布データ（NIMO出力CSV）に対して、複数の統計モデル
    1) Rayleigh + Normal 混合モデル（5パラメータ、元スクリプトのモデル）
    2) Gamma分布（2パラメータ）
    3) Lognormal分布（2パラメータ）
    4) Weibull分布（2パラメータ）
をそれぞれ最尤推定（MLE）でフィッティングし、

    - AIC / BIC（Akaike情報量規準 / ベイズ情報量規準）
    - Kolmogorov–Smirnov 適合度検定（統計量・p値）
    - k分割交差検証（held-outデータでの対数尤度）

によってモデル間を定量的に比較する。
さらに、混合モデルの重みパラメータ alpha（= w, Rayleigh成分の混合比率）の
不確実性を、ノンパラメトリック・ブートストラップにより
平均・標準偏差・95%信頼区間として推定する。

--------------------------------------------------------------------
元のスクリプト (NIMO_fitting_latest.py) との違い
--------------------------------------------------------------------
元のスクリプトは「ヒストグラム（ビン化した頻度）に対する最小二乗フィッティング」
を行っていました。この方法では真の尤度が定義できないため AIC/BIC による
モデル比較ができません。本スクリプトでは「生データに対する最尤推定 (MLE)」
に変更し、尤度ベースの評価（AIC/BIC/CV）が可能な形にしています。

（注記）Rayleigh成分は loc_r 未満で 0 になるよう修正していますが、
これは truncation 後に再正規化していないため、厳密には確率密度として
積分が1になりません。元モデルの定義をそのまま踏襲していますが、
論文等で使う場合はこの点を明記するか、再正規化を検討してください。

（注記2）loc_r（Rayleigh成分の立ち上がり位置）には、データの下側
LOC_R_PERCENTILE_CAP パーセンタイル点を上限とする制約を設けています。
これは、尤度関数の多峰性により loc_r がデータの大部分より右側の
不自然な位置（＝小さい気泡側の情報を無視した局所解）に収束してしまう
問題への対策です。

（注記3）Normal成分の平均 mu_n が Rayleigh成分の立ち上がり位置 loc_r
より左に来ないよう、制約 mu_n - loc_r >= 0 を設けています。これは
「完全な役割逆転」（Normalが小さいサイズ側、Rayleighが大きいサイズ側を
説明する解）だけを防ぐための最小限の制約です。以前は mu_n がRayleigh
の『ピーク』より右である必要がある、というより厳しい制約にしていま
したが、これだと2成分のピークが重なって鋭い山を協調して再現するという
自然な解を締め出してしまい、Rayleigh単体で無理に鋭いピークを作ろうと
した結果、不自然な『ジャンプ』が生じる副作用が確認されたため、緩和
しています。

（注記4）scale_r・sigma_n の下限を「データ標準偏差の5%」に設定しています。
これが無いと、正規分布またはRayleigh分布の成分がデータの密集点（ヒスト
グラムの最頻値付近）に対して極端に細いスパイク（デルタ関数的な解）に
収束し、対数尤度が発散するように見かけ上ものすごく良いフィットになって
しまう問題（混合正規分布のMLEに特有の非有界性）が起きます。特に
注記3の制約と組み合わさると、Normal成分がRayleighピークのすぐ右側に
細いスパイクとして張り付く、実データを全く説明しない解が選ばれることが
確認されたため、この下限を追加しました。

（出力ファイル）
各入力CSVごとに、以下も出力します。
    - {filename}_pdf_values_all_models.csv : 元のNIMO_fitting_latest.pyと
      同様の形式（x列＋各分布のPDF列＋Frequency/Histogram列）で、
      Rayleigh成分・Normal成分・Mixed PDFに加え、Gamma/Lognormal/Weibullの
      PDF列も含む。
    - {filename}_mixture_decomposition.png : 混合モデルをRayleigh成分・
      Normal成分に分解して重ね描きした図（各ピーク位置に縦線つき）。

入力フォルダ・出力フォルダは実行時にGUIで選択します。
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, rayleigh, gamma, lognorm, weibull_min, kstest
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ==========================================================
# 設定
# ==========================================================

DATA_COLUMN_INDEX = 3
NIMO_SCALE_FACTOR = 10  # NIMOの場合のスケール係数（ImageJの場合は1にする）

# 「一番最後の列に diameter (um単位) が入っている」形式
# （例: ..._opening_summary.csv の max_diameter_um 列）を判定するためのキーワード。
# 列名にこれらの語（大文字小文字を区別しない）がすべて含まれていれば、
# その列を直径データとして採用する（スケール係数は1＝そのまま使用）。
DIAMETER_COLUMN_KEYWORDS = ("diameter", "um")

# 混合モデルの初期値探索範囲（フル・グリッドサーチに使用。元スクリプトと同じ）
MU_OFFSETS = [-30, -20, -10, 0, 10, 20, 30]
LOC_INIT_CANDIDATES = [0, 10, 20, 30, 40, 50]

# 交差検証・ブートストラップの設定
N_CV_FOLDS = 5
N_BOOTSTRAP = 200          # 計算時間短縮したい場合は 50〜100 に減らしてもよい
CI_LEVEL = 0.95            # 信頼区間の水準
RANDOM_SEED = 42

# loc_r（Rayleigh成分の立ち上がり位置）の上限をデータの下側何パーセンタイルに
# 制限するか。物理的に「小さい気泡側の立ち上がり」を表すはずのloc_rが、
# データの大部分より右側（不自然に高い位置）に逃げてしまう局所解を防ぐための制約。
LOC_R_PERCENTILE_CAP = 20

# scale_r（Rayleigh成分の広がり）・sigma_n（Normal成分の広がり）の下限を、
# そのファイルのデータ標準偏差に対する割合で決める。
# 下限が緩すぎる（例: 0.1固定）と、正規分布の成分がデータの密集点に対して
# 極端に細いスパイク（デルタ関数的な解）に収束し、対数尤度が発散するように
# 見かけ上ものすごく良いフィットになってしまう「尤度の非有界性」問題が起きる。
# これを避けるため、標準偏差の一定割合を最小幅として強制する。
MIN_WIDTH_FRACTION_OF_STD = 0.10   # データ標準偏差の何%を最小幅とするか
MIN_WIDTH_ABS_FLOOR = 3.0          # 上記が過小になる場合の絶対的な下限


# ==========================================================
# Rayleigh + Normal 混合モデル
# ==========================================================

def rayleigh_modified_pdf(x, loc_r, scale_r):
    """loc_r より小さい x では 0 になるように修正した Rayleigh PDF"""
    pdf = rayleigh.pdf(x, loc=loc_r, scale=scale_r)
    return np.where(x < loc_r, 0.0, pdf)


def mixed_pdf(x, w, loc_r, scale_r, mu_n, sigma_n):
    """Rayleigh + Normal の混合確率密度関数"""
    return (
        w * rayleigh_modified_pdf(x, loc_r=loc_r, scale_r=scale_r)
        + (1 - w) * norm.pdf(x, loc=mu_n, scale=sigma_n)
    )


def mixed_cdf_builder(params, x_max, resolution=0.01):
    """
    混合モデルのCDFを数値積分で構築し、補間関数として返す。
    （混合モデルには解析的なCDFが存在しないため、KS検定用に必要）
    """
    grid = np.arange(0, x_max + 5 * resolution, resolution)
    pdf_vals = mixed_pdf(grid, *params)
    cdf_vals = cumulative_trapezoid(pdf_vals, grid, initial=0.0)
    total = cdf_vals[-1] if cdf_vals[-1] > 0 else 1.0
    cdf_vals = np.clip(cdf_vals / total, 0.0, 1.0)

    def cdf_func(x):
        return np.interp(x, grid, cdf_vals, left=0.0, right=1.0)

    return cdf_func


def neg_log_likelihood_mixture(params, x):
    w, loc_r, scale_r, mu_n, sigma_n = params
    pdf_vals = mixed_pdf(x, w, loc_r, scale_r, mu_n, sigma_n)
    pdf_vals = np.clip(pdf_vals, 1e-300, None)
    return -np.sum(np.log(pdf_vals))


def constraint_mu_3sigma(params):
    """制約条件: mu_n - 3 * sigma_n >= 0"""
    w, loc_r, scale_r, mu_n, sigma_n = params
    return mu_n - 3 * sigma_n


def constraint_rayleigh_peak_left_of_normal(params):
    """制約条件: Rayleigh成分のピーク（mode = loc_r + scale_r）が
    Normal成分の平均 mu_n より左に来るようにする。

    （NIMO_fitting_latest_updated.py と同じ、より厳しい版の制約に統一。
    比較スクリプトでは一時的に mu_n >= loc_r まで緩めていたが、
    latest_updated 版の境界条件に合わせるためここでは元の
    mu_n >= loc_r + scale_r に戻している。）"""
    w, loc_r, scale_r, mu_n, sigma_n = params
    rayleigh_mode = loc_r + scale_r
    return mu_n - rayleigh_mode


def compute_min_width(x, fraction=MIN_WIDTH_FRACTION_OF_STD, abs_floor=MIN_WIDTH_ABS_FLOOR):
    """scale_r / sigma_n の下限（最小幅）を、データ標準偏差の一定割合として計算する。
    正規分布・修正Rayleigh分布が『デルタ関数的なスパイク』に収束して
    尤度が発散するのを防ぐための下限。"""
    return max(abs_floor, fraction * np.std(x))


MIXTURE_BOUNDS_TEMPLATE = lambda mu_init_i, loc_r_upper, min_width: [
    (0, 1),                                # w（latest_updated版と同じ範囲。MLEのlog(0)対策が必要な場合は要注意）
    (0, loc_r_upper),                      # loc_r: 0 〜 データの下側 LOC_R_PERCENTILE_CAP %点
    (min_width, max(100, min_width * 2)),  # scale_r > min_width（スパイク防止）
    (mu_init_i - 10, None),                # mu_n
    (min_width, None),                      # sigma_n > min_width（スパイク防止）
]

MIXTURE_CONS = (
    {'type': 'ineq', 'fun': constraint_mu_3sigma},
    {'type': 'ineq', 'fun': constraint_rayleigh_peak_left_of_normal},
)


def fit_mixture_mle_full(x, loc_r_percentile_cap=LOC_R_PERCENTILE_CAP):
    """複数初期値によるフル・グリッドサーチ（元スクリプトと同じ探索範囲）でMLE。
    最小の負対数尤度（=最大尤度）を与えるパラメータを返す。

    loc_r_percentile_cap: loc_rの上限として使うパーセンタイル。
    データの下側 loc_r_percentile_cap %点より大きい位置には、Rayleigh成分の
    立ち上がりは来ないと仮定し、不自然な局所解（loc_rがデータの大部分より
    右側に逃げてしまう解）を除外する。
    """
    loc_r_upper = np.percentile(x, loc_r_percentile_cap)
    min_width = compute_min_width(x)

    mu_init_est = x.mean()
    mu_init_list = [mu_init_est + off for off in MU_OFFSETS]

    best_params, best_nll = None, np.inf
    for mu_init_i in mu_init_list:
        for loc_init_i in LOC_INIT_CANDIDATES:
            if loc_init_i > loc_r_upper:
                continue  # 上限を超える初期値はスキップ
            p0 = [0.5, loc_init_i, max(10, min_width), mu_init_i, max(2, min_width)]
            bounds = MIXTURE_BOUNDS_TEMPLATE(mu_init_i, loc_r_upper, min_width)
            try:
                res = minimize(neg_log_likelihood_mixture, p0, args=(x,),
                                bounds=bounds, constraints=MIXTURE_CONS)
            except Exception:
                continue
            if res.success and res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x

    # 万一、有効な初期値が一つも無かった場合（loc_r_upperが極端に小さい等）の保険
    if best_params is None:
        p0 = [0.5, 0, max(10, min_width), mu_init_est, max(2, min_width)]
        bounds = MIXTURE_BOUNDS_TEMPLATE(mu_init_est, loc_r_upper, min_width)
        try:
            res = minimize(neg_log_likelihood_mixture, p0, args=(x,),
                            bounds=bounds, constraints=MIXTURE_CONS)
            if res.success:
                best_params, best_nll = res.x, res.fun
        except Exception:
            pass

    return best_params, best_nll


def fit_mixture_mle_warmstart(x, p0, loc_r_percentile_cap=LOC_R_PERCENTILE_CAP):
    """既知の初期値（通常はフルフィットの結果）からの単一スタートMLE。
    交差検証・ブートストラップの高速化用。失敗時は None を返す。
    loc_rの上限・scale_r/sigma_nの下限は、渡された x（リサンプルデータや
    学習foldのデータ）ごとに計算し直す。"""
    loc_r_upper = np.percentile(x, loc_r_percentile_cap)
    min_width = compute_min_width(x)
    mu_init_i = p0[3]
    # p0のloc_rが今回のx上限を超えている場合は、上限内に収めてから最適化する
    p0_adj = list(p0)
    if p0_adj[1] > loc_r_upper:
        p0_adj[1] = loc_r_upper * 0.5
    # p0のscale_r / sigma_n が今回の下限を下回っている場合も調整する
    if p0_adj[2] < min_width:
        p0_adj[2] = min_width
    if p0_adj[4] < min_width:
        p0_adj[4] = min_width
    bounds = MIXTURE_BOUNDS_TEMPLATE(mu_init_i, loc_r_upper, min_width)
    try:
        res = minimize(neg_log_likelihood_mixture, p0_adj, args=(x,),
                        bounds=bounds, constraints=MIXTURE_CONS)
    except Exception:
        return None, np.inf
    if not res.success:
        return None, np.inf
    return res.x, res.fun


# ==========================================================
# 比較対象の分布（Gamma / Lognormal / Weibull）
# 位置パラメータ loc は 0 に固定（気泡径は原点から始まる正の量のため）
# ==========================================================

def fit_gamma(x):
    shape, loc, scale = gamma.fit(x, floc=0)
    params = (shape, scale)
    nll = -np.sum(gamma.logpdf(x, shape, loc=0, scale=scale))
    return params, nll


def fit_lognorm(x):
    x_safe = np.where(x <= 0, 1e-6, x)  # log(0) を避けるための微小シフト
    shape, loc, scale = lognorm.fit(x_safe, floc=0)
    params = (shape, scale)
    nll = -np.sum(lognorm.logpdf(x_safe, shape, loc=0, scale=scale))
    return params, nll


def fit_weibull(x):
    shape, loc, scale = weibull_min.fit(x, floc=0)
    params = (shape, scale)
    nll = -np.sum(weibull_min.logpdf(x, shape, loc=0, scale=scale))
    return params, nll


# モデルレジストリ: fit / pdf / cdf / パラメータ数 / 名前を一元管理
MODEL_SPECS = {
    "Rayleigh-Normal Mixture": {
        "k": 5,
        "fit_full": fit_mixture_mle_full,
        "pdf": lambda x, p: mixed_pdf(x, *p),
        "param_names": ["w(alpha)", "loc_r", "scale_r", "mu_n", "sigma_n"],
    },
    "Gamma": {
        "k": 2,
        "fit_full": fit_gamma,
        "pdf": lambda x, p: gamma.pdf(x, p[0], loc=0, scale=p[1]),
        "cdf": lambda x, p: gamma.cdf(x, p[0], loc=0, scale=p[1]),
        "param_names": ["shape", "scale"],
    },
    "Lognormal": {
        "k": 2,
        "fit_full": fit_lognorm,
        "pdf": lambda x, p: lognorm.pdf(x, p[0], loc=0, scale=p[1]),
        "cdf": lambda x, p: lognorm.cdf(x, p[0], loc=0, scale=p[1]),
        "param_names": ["shape", "scale"],
    },
    "Weibull": {
        "k": 2,
        "fit_full": fit_weibull,
        "pdf": lambda x, p: weibull_min.pdf(x, p[0], loc=0, scale=p[1]),
        "cdf": lambda x, p: weibull_min.cdf(x, p[0], loc=0, scale=p[1]),
        "param_names": ["shape", "scale"],
    },
}


# ==========================================================
# データ読み込み
# ==========================================================

def find_diameter_column(columns):
    """列名から diameter(um) 列を探す。見つかれば列名を、無ければ None を返す"""
    for col in columns:
        col_lower = str(col).lower()
        if all(keyword in col_lower for keyword in DIAMETER_COLUMN_KEYWORDS):
            return col
    return None


def load_data(csv_path):
    """CSVファイルから解析対象データ列を読み込む

    2種類の形式に対応:
      1) NIMO出力形式: 列インデックス3の値を10倍して使用
      2) diameter入りサマリー形式（例: ..._opening_summary.csv）:
         列名に "diameter" と "um" を含む列（例: max_diameter_um）を
         そのまま（スケール係数1）使用。最後の列に入っているケースを想定。
    """
    df = pd.read_csv(csv_path)

    diameter_col = find_diameter_column(df.columns)
    if diameter_col is not None:
        data = df[diameter_col].dropna().values
        print(f"  検出: diameter列形式 (列名: '{diameter_col}') としてデータを読み込みます。")
    else:
        data = NIMO_SCALE_FACTOR * df.iloc[:, DATA_COLUMN_INDEX].dropna().values
        # ImageJの場合は代わりに以下を使用:
        # data = df.iloc[:, 10].dropna().values
        print("  検出: NIMO出力形式として列インデックス3を読み込みます。")

    return data


# ==========================================================
# k分割交差検証
# ==========================================================

def make_kfold_indices(n, k, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)


def cross_validate_model(model_name, spec, x, folds):
    """各foldでtrainにフィットし、held-out testの平均対数尤度を返す"""
    fold_logliks = []
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        x_train, x_test = x[train_idx], x[test_idx]

        if model_name == "Rayleigh-Normal Mixture":
            # フル探索は重いので、全データでの最良解を初期値にした単一スタートで高速化
            params, nll = fit_mixture_mle_full(x_train) if i == 0 else (None, None)
            if params is None:
                # 2回目以降は最初のfoldの結果をウォームスタートとして使う
                params, _ = fit_mixture_mle_warmstart(x_train, cross_validate_model._warm_p0)
            cross_validate_model._warm_p0 = params if params is not None else cross_validate_model._warm_p0
            if params is None:
                continue
            pdf_vals = spec["pdf"](x_test, params)
        else:
            params, _ = spec["fit_full"](x_train)
            pdf_vals = spec["pdf"](x_test, params)

        pdf_vals = np.clip(pdf_vals, 1e-300, None)
        mean_loglik = np.mean(np.log(pdf_vals))
        fold_logliks.append(mean_loglik)

    if len(fold_logliks) == 0:
        return np.nan, np.nan
    return float(np.mean(fold_logliks)), float(np.std(fold_logliks))


# ==========================================================
# ブートストラップによる混合モデルパラメータ（alpha含む）の不確実性評価
# ==========================================================

def bootstrap_mixture_params(x, best_params, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    records = []
    for b in range(n_boot):
        x_boot = rng.choice(x, size=n, replace=True)
        params, nll = fit_mixture_mle_warmstart(x_boot, best_params)
        if params is None:
            continue
        records.append(params)

    boot_df = pd.DataFrame(
        records, columns=["w(alpha)", "loc_r", "scale_r", "mu_n", "sigma_n"]
    )

    alpha_lo = (1 - CI_LEVEL) / 2 * 100
    alpha_hi = (1 + CI_LEVEL) / 2 * 100

    summary_rows = []
    for col in boot_df.columns:
        vals = boot_df[col].dropna().values
        summary_rows.append({
            "parameter": col,
            "point_estimate": best_params[list(boot_df.columns).index(col)],
            "bootstrap_mean": np.mean(vals),
            "bootstrap_std": np.std(vals, ddof=1),
            f"CI_{CI_LEVEL*100:.0f}%_lower": np.percentile(vals, alpha_lo),
            f"CI_{CI_LEVEL*100:.0f}%_upper": np.percentile(vals, alpha_hi),
            "n_successful_boot": len(vals),
        })
    summary_df = pd.DataFrame(summary_rows)
    return boot_df, summary_df


# ==========================================================
# モデル比較（AIC / BIC / KS検定 / 交差検証）
# ==========================================================

def compare_models(x):
    n = len(x)
    folds = make_kfold_indices(n, N_CV_FOLDS, RANDOM_SEED)

    fitted_params = {}
    rows = []

    for name, spec in MODEL_SPECS.items():
        params, nll = spec["fit_full"](x)
        if params is None:
            print(f"  [警告] {name}: フィッティング失敗のためスキップ")
            continue
        fitted_params[name] = params

        k = spec["k"]
        logL = -nll
        aic = 2 * k - 2 * logL
        bic = k * np.log(n) - 2 * logL

        # KS検定
        if name == "Rayleigh-Normal Mixture":
            cdf_func = mixed_cdf_builder(params, x_max=x.max())
            ks_stat, ks_p = kstest(x, cdf_func)
        else:
            ks_stat, ks_p = kstest(x, lambda v: spec["cdf"](v, params))

        # 交差検証
        cv_mean, cv_std = cross_validate_model(name, spec, x, folds)

        rows.append({
            "model": name,
            "k": k,
            "n": n,
            "logL": logL,
            "logL_per_point": logL / n,
            "AIC": aic,
            "BIC": bic,
            "KS_stat": ks_stat,
            "KS_pvalue": ks_p,
            "CV_mean_logpdf": cv_mean,   # held-outデータでの平均対数尤度（大きいほど良い）
            "CV_std_logpdf": cv_std,
        })

    result_df = pd.DataFrame(rows)
    if len(result_df) > 0:
        best_aic = result_df["AIC"].min()
        result_df["delta_AIC"] = result_df["AIC"] - best_aic
        rel_likelihood = np.exp(-0.5 * result_df["delta_AIC"])
        result_df["AIC_weight"] = rel_likelihood / rel_likelihood.sum()
        result_df = result_df.sort_values("AIC").reset_index(drop=True)

    return result_df, fitted_params


# ==========================================================
# 結果の保存
# ==========================================================

def save_comparison_plot(filename, x, fitted_params, output_folder):
    bins = np.arange(0, x.max() + 2, 1)
    grid = np.linspace(0, x.max() + 1, 500)

    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=bins, density=True, alpha=0.4, color="gray", label="Data")

    for name, spec in MODEL_SPECS.items():
        if name not in fitted_params:
            continue
        params = fitted_params[name]
        y = spec["pdf"](grid, params)
        plt.plot(grid, y, label=name, linewidth=2)

    plt.title(f"Model comparison: {filename}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(output_folder,
                             os.path.splitext(filename)[0] + "_model_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_bootstrap_plot(filename, boot_df, output_folder):
    if boot_df.empty or "w(alpha)" not in boot_df.columns:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(boot_df["w(alpha)"].dropna(), bins=30, color="steelblue", alpha=0.75)
    plt.title(f"Bootstrap distribution of alpha (w): {filename}")
    plt.xlabel("alpha (Rayleigh weight)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(output_folder,
                             os.path.splitext(filename)[0] + "_alpha_bootstrap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_mixture_decomposition_plot(filename, x, mixture_params, output_folder):
    """Rayleigh-Normal混合モデルを、Rayleigh成分・Normal成分・合成(Mixed)PDFに
    分解して表示する図。loc_r上限制約付きバージョンでのみ使用。"""
    w, loc_r, scale_r, mu_n, sigma_n = mixture_params
    bins = np.arange(0, x.max() + 2, 1)
    grid = np.linspace(0, x.max() + 1, 500)

    rayleigh_component = w * rayleigh_modified_pdf(grid, loc_r, scale_r)
    normal_component = (1 - w) * norm.pdf(grid, mu_n, sigma_n)
    mixed = rayleigh_component + normal_component

    rayleigh_mode = loc_r + scale_r

    plt.figure(figsize=(8, 5))
    plt.hist(x, bins=bins, density=True, alpha=0.35, color="gray", label="Data")
    plt.plot(grid, mixed, color="black", linewidth=2.5, label="Mixed PDF (fit)")
    plt.plot(grid, rayleigh_component, color="#1f77b4", linewidth=2,
              linestyle="--", label=f"Rayleigh component (w={w:.3f})")
    plt.plot(grid, normal_component, color="#2ca02c", linewidth=2,
              linestyle="--", label=f"Normal component (1-w={1 - w:.3f})")
    plt.axvline(rayleigh_mode, color="#1f77b4", linewidth=1, alpha=0.6)
    plt.axvline(mu_n, color="#2ca02c", linewidth=1, alpha=0.6)
    plt.text(rayleigh_mode, plt.ylim()[1] * 0.92, "Rayleigh peak",
              color="#1f77b4", rotation=90, va="top", ha="right", fontsize=8)
    plt.text(mu_n, plt.ylim()[1] * 0.92, "Normal peak",
              color="#2ca02c", rotation=90, va="top", ha="right", fontsize=8)

    plt.title(f"Rayleigh / Normal decomposition: {filename}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(output_folder,
                             os.path.splitext(filename)[0] + "_mixture_decomposition.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_pdf_values_csv(filename, x, fitted_params, output_folder):
    """元の NIMO_fitting_latest.py の出力（x列＋各分布のPDF列＋ヒストグラム列）
    を踏襲し、Gamma / Lognormal / Weibull の列も加えたCSVを出力する。"""
    bins = np.arange(0, x.max() + 2, 1)
    counts, _ = np.histogram(x, bins=bins, density=True)      # Frequency (density)
    hist_counts, _ = np.histogram(x, bins=bins)                # Histogram (raw count)
    grid = np.arange(0, x.max() + 1, 1)                        # counts と同じ長さ

    table = {"x": grid}

    if "Rayleigh-Normal Mixture" in fitted_params:
        w, loc_r, scale_r, mu_n, sigma_n = fitted_params["Rayleigh-Normal Mixture"]
        table["Rayleigh component PDF"] = w * rayleigh_modified_pdf(grid, loc_r, scale_r)
        table["Normal component PDF"] = (1 - w) * norm.pdf(grid, mu_n, sigma_n)
        table["Mixed PDF (Rayleigh+Normal)"] = (
            table["Rayleigh component PDF"] + table["Normal component PDF"]
        )

    if "Gamma" in fitted_params:
        shape, scale = fitted_params["Gamma"]
        table["Gamma PDF"] = gamma.pdf(grid, shape, loc=0, scale=scale)

    if "Lognormal" in fitted_params:
        shape, scale = fitted_params["Lognormal"]
        table["Lognormal PDF"] = lognorm.pdf(grid, shape, loc=0, scale=scale)

    if "Weibull" in fitted_params:
        shape, scale = fitted_params["Weibull"]
        table["Weibull PDF"] = weibull_min.pdf(grid, shape, loc=0, scale=scale)

    table["Frequency"] = counts
    table["Histogram"] = hist_counts

    pdf_table = pd.DataFrame(table)
    out_path = os.path.join(output_folder,
                             os.path.splitext(filename)[0] + "_pdf_values_all_models.csv")
    pdf_table.to_csv(out_path, index=False)
    return pdf_table


# ==========================================================
# フォルダ選択（GUI）
# ==========================================================

def select_folders():
    from tkinter import Tk, filedialog  # 実行時にのみ読み込む（GUI不要な用途での依存を避ける）
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    input_folder = filedialog.askdirectory(title="CSVファイルが入ったフォルダを選択してください")
    if not input_folder:
        print("入力フォルダが選択されませんでした。処理を中止します。")
        sys.exit(1)

    output_folder = filedialog.askdirectory(title="結果の出力先フォルダを選択してください")
    if not output_folder:
        print("出力フォルダが選択されませんでした。処理を中止します。")
        sys.exit(1)

    root.destroy()
    return input_folder, output_folder


# ==========================================================
# メイン処理
# ==========================================================

def main():
    input_folder, output_folder = select_folders()
    print(f"入力フォルダ: {input_folder}")
    print(f"出力フォルダ: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    all_comparison_rows = []
    all_alpha_rows = []

    for filename in os.listdir(input_folder):
        if not filename.endswith(".csv"):
            continue

        full_path = os.path.join(input_folder, filename)
        print(f"\n=== 処理中: {filename} ===")

        try:
            x = load_data(full_path)
            x = np.asarray(x, dtype=float)
            x = x[x >= 0]

            print("  モデルをフィッティング中（Mixture / Gamma / Lognormal / Weibull）...")
            comparison_df, fitted_params = compare_models(x)

            if comparison_df.empty:
                print(f"  {filename}: 全モデルのフィッティングに失敗しました")
                continue

            print(comparison_df[["model", "k", "AIC", "BIC", "delta_AIC",
                                  "AIC_weight", "KS_pvalue", "CV_mean_logpdf"]]
                  .to_string(index=False))

            comp_path = os.path.join(
                output_folder, os.path.splitext(filename)[0] + "_model_comparison.csv")
            comparison_df.to_csv(comp_path, index=False)

            save_comparison_plot(filename, x, fitted_params, output_folder)
            save_pdf_values_csv(filename, x, fitted_params, output_folder)

            if "Rayleigh-Normal Mixture" in fitted_params:
                save_mixture_decomposition_plot(
                    filename, x, fitted_params["Rayleigh-Normal Mixture"], output_folder)

            comparison_df_tagged = comparison_df.copy()
            comparison_df_tagged.insert(0, "filename", filename)
            all_comparison_rows.append(comparison_df_tagged)

            # --- 混合モデルの alpha（w）の不確実性評価 ---
            if "Rayleigh-Normal Mixture" in fitted_params:
                best_mix_params = fitted_params["Rayleigh-Normal Mixture"]
                print(f"  ブートストラップ実行中 (n={N_BOOTSTRAP}) ...")
                boot_df, boot_summary = bootstrap_mixture_params(x, best_mix_params)

                boot_path = os.path.join(
                    output_folder, os.path.splitext(filename)[0] + "_mixture_bootstrap.csv")
                boot_df.to_csv(boot_path, index=False)

                summary_path = os.path.join(
                    output_folder, os.path.splitext(filename)[0] + "_mixture_bootstrap_summary.csv")
                boot_summary.to_csv(summary_path, index=False)

                save_bootstrap_plot(filename, boot_df, output_folder)

                alpha_row = boot_summary[boot_summary["parameter"] == "w(alpha)"].copy()
                alpha_row.insert(0, "filename", filename)
                all_alpha_rows.append(alpha_row)

                print("  alpha (Rayleigh weight) の不確実性:")
                print(alpha_row.to_string(index=False))

        except Exception as e:
            print(f"  {filename} の解析中にエラー: {e}")

    if all_comparison_rows:
        all_comparison_df = pd.concat(all_comparison_rows, ignore_index=True)
        all_comparison_df.to_csv(
            os.path.join(output_folder, "all_files_model_comparison.csv"), index=False)

    if all_alpha_rows:
        all_alpha_df = pd.concat(all_alpha_rows, ignore_index=True)
        all_alpha_df.to_csv(
            os.path.join(output_folder, "all_files_alpha_uncertainty.csv"), index=False)

    print(f"\n✅ 全ファイルの解析が完了しました。結果は以下に保存されました：\n→ {output_folder}")


if __name__ == "__main__":
    main()
