import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, rayleigh
from scipy.optimize import minimize
from tkinter import Tk, filedialog


# ==========================================================
# 設定
# ==========================================================

# データ列の指定: NIMO出力の場合は列インデックス3（10倍して使用）
# ImageJ出力を使う場合は下記 load_data() 内のコメントを参照して切り替える
DATA_COLUMN_INDEX = 3
NIMO_SCALE_FACTOR = 10  # NIMOの場合のスケール係数（ImageJの場合は1にする）

# 正規分布の初期値の探索範囲（データ平均からのオフセット）
MU_OFFSETS = [-30, -20, -10, 0, 10, 20, 30]

# Rayleigh分布 loc の初期値候補
LOC_INIT_CANDIDATES = [0, 10, 20, 30, 40, 50]

# loc_r（Rayleigh成分の立ち上がり位置）の上限をデータの下側何パーセンタイルに
# 制限するか。NIMO_fitting_model_comparison.py と同じ設定値。
LOC_R_PERCENTILE_CAP = 20

# scale_r（Rayleigh成分の広がり）・sigma_n（Normal成分の広がり）の下限を、
# そのファイルのデータ標準偏差に対する割合で決める。
# NIMO_fitting_model_comparison.py と同じ設定値。
MIN_WIDTH_FRACTION_OF_STD = 0.10   # データ標準偏差の何%を最小幅とするか
MIN_WIDTH_ABS_FLOOR = 3.0          # 上記が過小になる場合の絶対的な下限


# ==========================================================
# モデル関数
# ==========================================================

def rayleigh_modified_pdf(x, loc_r, scale_r):
    """loc_r より小さい x では 0 になるように修正した Rayleigh PDF"""
    pdf = rayleigh.pdf(x, loc=loc_r, scale=scale_r)
    return np.where(x < loc_r, 0, pdf)


def mixed_pdf(x, w, loc_r, scale_r, mu_n, sigma_n):
    """Rayleigh + Normal の混合確率密度関数"""
    return (
        w * rayleigh_modified_pdf(x, loc_r=loc_r, scale_r=scale_r)
        + (1 - w) * norm.pdf(x, loc=mu_n, scale=sigma_n)
    )


def loss(params, x, y):
    """残差平方和（フィッティング誤差）"""
    a, loc_r, scale_r, mu_n, sigma_n = params
    y_pred = mixed_pdf(x, a, loc_r, scale_r, mu_n, sigma_n)
    return np.sum((y - y_pred) ** 2)


def constraint_mu_3sigma(params):
    """制約条件: mu_n - 3 * sigma_n >= 0"""
    a, loc_r, scale_r, mu_n, sigma_n = params
    return mu_n - 3 * sigma_n


def constraint_rayleigh_peak_left_of_normal(params):
    """制約条件: Rayleigh成分のピーク（mode = loc_r + scale_r）が
    Normal成分の平均 mu_n より左に来るようにする。
    （NIMO_fitting_model_comparison.py と同じ制約）"""
    a, loc_r, scale_r, mu_n, sigma_n = params
    rayleigh_mode = loc_r + scale_r
    return mu_n - rayleigh_mode


def compute_min_width(x, fraction=MIN_WIDTH_FRACTION_OF_STD, abs_floor=MIN_WIDTH_ABS_FLOOR):
    """scale_r / sigma_n の下限（最小幅）を、データ標準偏差の一定割合として計算する。
    （NIMO_fitting_model_comparison.py と同じ考え方）"""
    return max(abs_floor, fraction * np.std(x))


# ==========================================================
# データ読み込み
# ==========================================================

def load_data(csv_path):
    """CSVファイルから解析対象データ列を読み込む"""
    df = pd.read_csv(csv_path)
    # NIMOの場合（列インデックス3を10倍）
    data = NIMO_SCALE_FACTOR * df.iloc[:, DATA_COLUMN_INDEX].dropna().values
    # ImageJの場合は代わりに以下を使用:
    # data = df.iloc[:, 10].dropna().values
    return data


# ==========================================================
# 1ファイル分のフィッティング処理
# ==========================================================

def fit_single_file(data):
    """複数の初期値でフィッティングを行い、最良の結果を返す"""
    bins = np.arange(0, max(data) + 2, 1)
    counts, bin_edges = np.histogram(data, bins, density=True)

    mu_init_est = data.mean()
    mu_init_list = [mu_init_est + offset for offset in MU_OFFSETS]

    x = np.arange(0, int(np.ceil(max(data))) + 1, 1)
    y = counts

    # --- 境界条件の計算(データに応じて動的に決める) ---
    loc_r_upper = np.percentile(data, LOC_R_PERCENTILE_CAP)
    min_width = compute_min_width(data)

    best_result = None
    best_loss = np.inf
    best_params = None

    cons = (
        {'type': 'ineq', 'fun': constraint_mu_3sigma},
        {'type': 'ineq', 'fun': constraint_rayleigh_peak_left_of_normal},
    )

    for mu_init_i in mu_init_list:
        for loc_init_i in LOC_INIT_CANDIDATES:
            if loc_init_i > loc_r_upper:
                continue  # 上限を超える初期値はスキップ

            p0 = [0.5, loc_init_i, max(10, min_width), mu_init_i, max(2, min_width)]
            bounds = [
                (0, 1),                                    # a ∈ [0,1]
                (0, loc_r_upper),                           # loc_r: 0 〜 データの下側 LOC_R_PERCENTILE_CAP %点
                (min_width, max(100, min_width * 2)),        # scale_r > min_width（スパイク防止）
                (mu_init_i - 10, None),                     # mu_n の下限
                (min_width, None),                           # sigma_n > min_width（スパイク防止）
            ]

            result = minimize(loss, p0, args=(x, y), bounds=bounds, constraints=cons)
            print(f"  mu_init={mu_init_i:.2f}, loc_init={loc_init_i}, "
                  f"loss={loss(result.x, x, y):.4f}, weight={result.x[0]:.3f}")

            if result.success and result.fun < best_loss:
                best_loss = result.fun
                best_result = result
                best_params = result.x

    return best_params, counts, bins


# ==========================================================
# 結果の保存（PDF値CSVと図）
# ==========================================================

def save_results(filename, data, fitted_params, counts, bins, output_folder):
    x_pdf = np.arange(0, max(data) + 1, 1)
    y_pdf = mixed_pdf(x_pdf, *fitted_params)

    bin_edges = np.arange(0, max(data) + 2, 1)
    hist_counts, _ = np.histogram(data, bins=bin_edges)

    x = np.arange(0, int(np.ceil(max(data))) + 1, 1)
    y_fit = mixed_pdf(x, *fitted_params)
    y_rayleigh = fitted_params[0] * rayleigh_modified_pdf(x, fitted_params[1], fitted_params[2])
    y_norm = (1 - fitted_params[0]) * norm.pdf(x, fitted_params[3], fitted_params[4])

    # --- PDF値をCSVに出力 ---
    pdf_table = pd.DataFrame({
        'x': x_pdf,
        'Rayleigh PDF': y_rayleigh,
        'Normal PDF': y_norm,
        'mixed PDF': y_pdf,
        'Frequency': counts,
        'Histogram': hist_counts,
    })
    pdf_filename = os.path.splitext(filename)[0] + "_pdf_values.csv"
    pdf_table.to_csv(os.path.join(output_folder, pdf_filename), index=False)

    # --- 図の保存 ---
    plt.figure()
    plt.hist(data, bins, density=True, alpha=0.5, label='Data')
    plt.plot(x, y_rayleigh, 'b-', label='Rayleigh PDF')
    plt.plot(x, y_norm, 'g-', label='Normal PDF')
    plt.plot(x, y_fit, 'r-', label='Fitted PDF')
    plt.title(f'Fit: {filename}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.splitext(filename)[0] + "_fit.png"
    plt.savefig(os.path.join(output_folder, plot_filename))
    plt.close()

    print(f"Fitted parameters ({filename}):")
    print(f"  Weight (Rayleigh): {fitted_params[0]:.3f}")
    print(f"  Rayleigh loc:      {fitted_params[1]:.3f}")
    print(f"  Rayleigh scale:    {fitted_params[2]:.3f}")
    print(f"  Normal mean:       {fitted_params[3]:.3f}")
    print(f"  Normal std dev:    {fitted_params[4]:.3f}")


# ==========================================================
# フォルダ選択（GUI）
# ==========================================================

def select_folders():
    """GUIダイアログで入力フォルダ・出力フォルダを選択する"""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # ダイアログを最前面に表示

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
    fit_param_path = os.path.join(output_folder, "fit_results.csv")

    results = []

    for filename in os.listdir(input_folder):
        if not filename.endswith(".csv"):
            continue

        full_path = os.path.join(input_folder, filename)
        print(f"\n=== 処理中: {filename} ===")

        try:
            data = load_data(full_path)
            fitted_params, counts, bins = fit_single_file(data)

            if fitted_params is None:
                print(f"{filename}: フィッティングに失敗しました（最適化が成功しませんでした）")
                continue

            save_results(filename, data, fitted_params, counts, bins, output_folder)
            results.append([filename, *fitted_params])

        except Exception as e:
            print(f"{filename} の解析中にエラー: {e}")

    # --- 全ファイルのパラメータをCSVに出力 ---
    columns = ["filename", "weight_rayleigh", "loc_rayleigh", "scale_rayleigh", "mu_normal", "sigma_normal"]
    df_results = pd.DataFrame(results, columns=columns)
    df_results.to_csv(fit_param_path, index=False)

    print(f"\n✅ 全ファイルの解析が完了しました。結果は以下に保存されました：\n→ {fit_param_path}")


if __name__ == "__main__":
    main()
