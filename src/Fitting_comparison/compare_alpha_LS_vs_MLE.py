"""
compare_alpha_LS_vs_MLE.py

最小二乗法(NIMO_fitting_latest_updated.py の fit_results.csv)による
alpha(weight_rayleigh)の点推定と、
最尤推定(NIMO_fitting_model_comparison.py の all_files_alpha_uncertainty.csv)
によるalphaの点推定・ブートストラップ不確実性を突き合わせ、

    - 散布図(一致直線 y=x つき、MLE側はブートストラップの95%CIをエラーバー表示)
    - Bland-Altman図(平均 vs 差、手法間の系統的なズレの有無を見る)
    - Pearson相関係数・Spearman順位相関・平均絶対誤差(MAE)・
      平均差(bias)・一致の限界(limits of agreement)

を算出・出力する。論文Supporting InformationのS5
(最小二乗法 vs MLEの妥当性検証)に対応する。

追加で all_files_model_comparison.csv があれば、各ファイルで
Rayleigh-Normal Mixtureモデルが実際にAIC最良だったか(delta_AIC==0)を
散布図の点の色分けに使う（両手法が一致しないファイルが、そもそも
混合モデルの当てはまり自体が悪いファイルなのかを確認できる）。

入力CSV(いずれも同じフォルダに置く想定):
    - fit_results.csv                  (必須)
    - all_files_alpha_uncertainty.csv  (必須)
    - all_files_model_comparison.csv   (任意、あれば色分けに使う)

使い方:
    python compare_alpha_LS_vs_MLE.py
    実行するとGUIダイアログで、上記CSVが入ったフォルダと、
    出力先フォルダを順に選択する。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

FIT_RESULTS_NAME = "fit_results.csv"
ALPHA_UNCERTAINTY_NAME = "all_files_alpha_uncertainty.csv"
MODEL_COMPARISON_NAME = "all_files_model_comparison.csv"  # 任意


# ==========================================================
# データ読み込み・突き合わせ
# ==========================================================

def load_and_merge(csv_folder):
    fit_path = os.path.join(csv_folder, FIT_RESULTS_NAME)
    alpha_path = os.path.join(csv_folder, ALPHA_UNCERTAINTY_NAME)
    comp_path = os.path.join(csv_folder, MODEL_COMPARISON_NAME)

    if not os.path.exists(fit_path):
        print(f"見つかりません: {fit_path}")
        sys.exit(1)
    if not os.path.exists(alpha_path):
        print(f"見つかりません: {alpha_path}")
        sys.exit(1)

    fit_df = pd.read_csv(fit_path)[["filename", "weight_rayleigh"]].rename(
        columns={"weight_rayleigh": "alpha_LS"})

    alpha_df = pd.read_csv(alpha_path)
    alpha_df = alpha_df[alpha_df["parameter"] == "w(alpha)"].copy()
    alpha_df = alpha_df.rename(columns={
        "point_estimate": "alpha_MLE",
        "bootstrap_mean": "alpha_MLE_bootstrap_mean",
        "bootstrap_std": "alpha_MLE_bootstrap_std",
        "CI_95%_lower": "alpha_MLE_CI_lower",
        "CI_95%_upper": "alpha_MLE_CI_upper",
    })[["filename", "alpha_MLE", "alpha_MLE_bootstrap_mean", "alpha_MLE_bootstrap_std",
        "alpha_MLE_CI_lower", "alpha_MLE_CI_upper"]]

    merged = pd.merge(fit_df, alpha_df, on="filename", how="inner")

    n_fit, n_alpha, n_merged = len(fit_df), len(alpha_df), len(merged)
    if n_merged < max(n_fit, n_alpha):
        print(f"[注意] filenameが完全には一致していません: "
              f"fit_results={n_fit}件, alpha_uncertainty={n_alpha}件, "
              f"突き合わせ後={n_merged}件。ファイル名の表記ゆれ(大文字小文字・"
              f"スペースなど)がないか確認してください。")

    # 任意: モデル比較CSVから、混合モデルがAIC最良だったか(delta_AIC==0)を付与
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        mix_df = comp_df[comp_df["model"] == "Rayleigh-Normal Mixture"][
            ["filename", "delta_AIC", "AIC_weight", "KS_pvalue"]]
        merged = pd.merge(merged, mix_df, on="filename", how="left")
        merged["mixture_is_AIC_best"] = merged["delta_AIC"].fillna(np.inf) < 1e-6
    else:
        print(f"[情報] {MODEL_COMPARISON_NAME} が見つからないため、"
              f"色分けなしで作図します。")
        merged["mixture_is_AIC_best"] = True

    return merged


# ==========================================================
# 統計量の算出
# ==========================================================

def compute_agreement_stats(merged):
    a_ls = merged["alpha_LS"].values
    a_mle = merged["alpha_MLE"].values
    diff = a_mle - a_ls          # MLE - LS (bias)
    mean_pair = (a_mle + a_ls) / 2

    pearson_r, pearson_p = stats.pearsonr(a_ls, a_mle)
    spearman_rho, spearman_p = stats.spearmanr(a_ls, a_mle)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    bias = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    loa_lower = bias - 1.96 * sd_diff   # Bland-Altman limits of agreement
    loa_upper = bias + 1.96 * sd_diff

    stats_summary = {
        "n": len(merged),
        "Pearson_r": pearson_r,
        "Pearson_p": pearson_p,
        "Spearman_rho": spearman_rho,
        "Spearman_p": spearman_p,
        "MAE": mae,
        "RMSE": rmse,
        "bias_MLE_minus_LS": bias,
        "SD_diff": sd_diff,
        "LoA_lower_95pct": loa_lower,
        "LoA_upper_95pct": loa_upper,
    }
    return stats_summary, diff, mean_pair


# ==========================================================
# 図の作成
# ==========================================================

def plot_scatter_agreement(merged, stats_summary, output_folder):
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = np.where(merged["mixture_is_AIC_best"], "#1f77b4", "#d62728")
    err = [merged["alpha_MLE"] - merged["alpha_MLE_CI_lower"],
           merged["alpha_MLE_CI_upper"] - merged["alpha_MLE"]]

    ax.errorbar(merged["alpha_LS"], merged["alpha_MLE"], yerr=err,
                fmt="none", ecolor="gray", elinewidth=1, capsize=2, zorder=1)
    ax.scatter(merged["alpha_LS"], merged["alpha_MLE"], c=colors,
               s=40, zorder=2, edgecolor="black", linewidth=0.5)

    lims = [0, 1]
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x (perfect agreement)")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("alpha (least-squares point estimate)")
    ax.set_ylabel("alpha (MLE point estimate, error bar = bootstrap 95% CI)")
    ax.set_title("Agreement between least-squares and MLE estimates of alpha")

    textstr = (f"n = {stats_summary['n']}\n"
               f"Pearson r = {stats_summary['Pearson_r']:.3f} "
               f"(p = {stats_summary['Pearson_p']:.1e})\n"
               f"MAE = {stats_summary['MAE']:.3f}\n"
               f"bias (MLE − LS) = {stats_summary['bias_MLE_minus_LS']:.3f}")
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    if (~merged["mixture_is_AIC_best"]).any():
        ax.scatter([], [], c="#1f77b4", edgecolor="black", label="Mixture = AIC-best")
        ax.scatter([], [], c="#d62728", edgecolor="black", label="Mixture NOT AIC-best")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    basename = "alpha_LS_vs_MLE_scatter"
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, basename + ".png"), dpi=150)
    plt.close(fig)

    origin_df = merged[["filename", "alpha_LS", "alpha_MLE",
                         "alpha_MLE_CI_lower", "alpha_MLE_CI_upper"]].copy()
    origin_df["identity_line_x"] = np.linspace(0, 1, len(origin_df))
    origin_df["identity_line_y"] = origin_df["identity_line_x"]
    origin_df.to_csv(os.path.join(output_folder, basename + ".csv"), index=False)


def plot_bland_altman(merged, diff, mean_pair, stats_summary, output_folder):
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = np.where(merged["mixture_is_AIC_best"], "#1f77b4", "#d62728")
    ax.scatter(mean_pair, diff, c=colors, s=40, edgecolor="black", linewidth=0.5)

    ax.axhline(stats_summary["bias_MLE_minus_LS"], color="black", linewidth=1.2,
               label=f"bias = {stats_summary['bias_MLE_minus_LS']:.3f}")
    ax.axhline(stats_summary["LoA_upper_95pct"], color="gray", linestyle="--",
               linewidth=1, label="95% limits of agreement")
    ax.axhline(stats_summary["LoA_lower_95pct"], color="gray", linestyle="--",
               linewidth=1)

    ax.set_xlabel("Mean of alpha_LS and alpha_MLE")
    ax.set_ylabel("Difference (alpha_MLE − alpha_LS)")
    ax.set_title("Bland–Altman plot: agreement between the two estimation methods")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    basename = "alpha_LS_vs_MLE_bland_altman"
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, basename + ".png"), dpi=150)
    plt.close(fig)

    origin_df = pd.DataFrame({
        "filename": merged["filename"],
        "mean (LS, MLE)": mean_pair,
        "difference (MLE - LS)": diff,
    })
    origin_df["bias"] = np.nan
    origin_df["LoA_upper"] = np.nan
    origin_df["LoA_lower"] = np.nan
    origin_df.loc[0, "bias"] = stats_summary["bias_MLE_minus_LS"]
    origin_df.loc[0, "LoA_upper"] = stats_summary["LoA_upper_95pct"]
    origin_df.loc[0, "LoA_lower"] = stats_summary["LoA_lower_95pct"]
    origin_df.to_csv(os.path.join(output_folder, basename + ".csv"), index=False)


# ==========================================================
# フォルダ選択（GUI）
# ==========================================================

def select_folders():
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    csv_folder = filedialog.askdirectory(
        title=f"{FIT_RESULTS_NAME} と {ALPHA_UNCERTAINTY_NAME} が入ったフォルダを選択してください")
    if not csv_folder:
        print("CSVフォルダが選択されませんでした。処理を中止します。")
        sys.exit(1)

    output_folder = filedialog.askdirectory(title="図の出力先フォルダを選択してください")
    if not output_folder:
        print("出力フォルダが選択されませんでした。処理を中止します。")
        sys.exit(1)

    root.destroy()
    return csv_folder, output_folder


# ==========================================================
# メイン
# ==========================================================

def main():
    csv_folder, output_folder = select_folders()
    os.makedirs(output_folder, exist_ok=True)

    merged = load_and_merge(csv_folder)
    stats_summary, diff, mean_pair = compute_agreement_stats(merged)

    print("=== 一致度の統計量 ===")
    for k, v in stats_summary.items():
        print(f"  {k}: {v}")

    plot_scatter_agreement(merged, stats_summary, output_folder)
    plot_bland_altman(merged, diff, mean_pair, stats_summary, output_folder)

    merged["alpha_diff_MLE_minus_LS"] = diff
    merged["alpha_mean_LS_MLE"] = mean_pair
    merged.to_csv(os.path.join(output_folder, "alpha_LS_vs_MLE_merged_data.csv"), index=False)

    summary_df = pd.DataFrame([stats_summary])
    summary_df.to_csv(os.path.join(output_folder, "alpha_LS_vs_MLE_summary_stats.csv"), index=False)

    print(f"\n✅ 検証図・統計量を出力しました → {output_folder}")


if __name__ == "__main__":
    main()
