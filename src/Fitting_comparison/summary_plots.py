"""
summary_plots.py

NIMO_fitting_model_comparison.py の出力である
    - all_files_model_comparison.csv
    - all_files_alpha_uncertainty.csv
を読み込み、意味の近い指標ごとにグラフをまとめて出力する。

グループ分けの考え方:
    1. AIC / BIC            -> 同じ「情報量規準」スケールなので同じ図に
    2. delta_AIC / AIC重み  -> 相対評価指標として同じ図に
    3. KS検定 (統計量とp値) -> 0〜1で範囲は同じだが意味が異なるため別図
    4. logL_per_point / CV_mean_logpdf -> in-sample vs out-of-sample の過学習チェック
    4b. CV_mean_logpdf の4モデル直接比較 -> ファイルごとにペアで比較(折れ線) と
        ファイル内最良モデル基準のΔCV(箱ひげ+ジッタープロット)の2種類
    5. alpha (混合比率) の不確実性 -> 他と単位が違うため独立したフォレストプロット
    6. AICで最良となったモデルの回数 -> モデル選択の頑健性を見る

使い方:
    python summary_plots.py
    実行するとGUIダイアログが表示され、
        1) all_files_model_comparison.csv / all_files_alpha_uncertainty.csv が
           入っているフォルダ
        2) グラフの出力先フォルダ
    を順に選択する。

出力される各PNG図には、同じフォルダに同じベース名(拡張子だけ.csv)の
Origin用CSVが一緒に出力される。例: summary_AIC_BIC.png と同じ列構成の
summary_AIC_BIC.csv。Originにそのまま貼り付ければ同じ図を再現できる。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_ORDER = ["Rayleigh-Normal Mixture", "Gamma", "Lognormal", "Weibull"]
MODEL_COLORS = {
    "Rayleigh-Normal Mixture": "#d62728",
    "Gamma": "#1f77b4",
    "Lognormal": "#2ca02c",
    "Weibull": "#9467bd",
}


def _sorted_models(models):
    return [m for m in MODEL_ORDER if m in models] + \
           [m for m in models if m not in MODEL_ORDER]


def _out_path(folder, filename):
    """folder と filename を結合し、Windowsで安全なパスにして返す。

    OneDriveの入れ子フォルダなどでパスが長くなったり(MAX_PATH=260超え)、
    tkinterのaskdirectoryがフォルダ部分をスラッシュ("/")混じりで返すのに
    ファイル名部分だけos.path.joinがバックスラッシュで継ぎ足す
    ("...\\summary\\summary_xxx.png"のような混在)ことで、
    PillowのImage.save/open が [Errno 22] Invalid argument で
    失敗するケースがあるため、
      1) os.path.normpath でセパレータを統一
      2) Windowsで長すぎる場合は \\\\?\\ プレフィックスを付けてOSの
         パス長制限を回避
    する。
    """
    path = os.path.normpath(os.path.join(folder, filename))
    if os.name == "nt" and len(path) > 240 and not path.startswith("\\\\?\\"):
        if path.startswith("\\\\"):
            path = "\\\\?\\UNC\\" + path[2:]
        else:
            path = "\\\\?\\" + path
    return path


def _save_png_with_origin_csv(fig_basename, output_folder, origin_df):
    """指定したbasenameで図(.png、すでに保存済み)に対応するOrigin用CSVを
    同じフォルダ・同じbasenameで保存する(拡張子だけ.csvに変わる)。"""
    csv_path = _out_path(output_folder, fig_basename + ".csv")
    origin_df.to_csv(csv_path, index=False)


# ==========================================================
# グループ1: AIC / BIC
# ==========================================================

def plot_aic_bic(df, output_folder):
    models = _sorted_models(df["model"].unique())
    x = np.arange(len(models))
    width = 0.35

    aic_mean = [df[df.model == m]["AIC"].mean() for m in models]
    aic_std = [df[df.model == m]["AIC"].std(ddof=0) for m in models]
    bic_mean = [df[df.model == m]["BIC"].mean() for m in models]
    bic_std = [df[df.model == m]["BIC"].std(ddof=0) for m in models]

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, aic_mean, width, yerr=aic_std, capsize=4,
            label="AIC", color="#4c72b0")
    plt.bar(x + width / 2, bic_mean, width, yerr=bic_std, capsize=4,
            label="BIC", color="#dd8452")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Information criterion (lower = better)")
    plt.title("AIC / BIC comparison across files (mean ± SD)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    basename = "summary_AIC_BIC"
    plt.savefig(_out_path(output_folder, basename + ".png"), dpi=150)
    plt.close()

    origin_df = pd.DataFrame({
        "model": models,
        "AIC_mean": aic_mean, "AIC_std": aic_std,
        "BIC_mean": bic_mean, "BIC_std": bic_std,
    })
    _save_png_with_origin_csv(basename, output_folder, origin_df)


# ==========================================================
# グループ2: delta_AIC / AIC重み
# ==========================================================

def plot_delta_aic_and_weight(df, output_folder):
    models = _sorted_models(df["model"].unique())

    # delta_AIC: モデルごとの平均
    delta_mean = [df[df.model == m]["delta_AIC"].mean() for m in models]
    plt.figure(figsize=(7, 5))
    plt.bar(models, delta_mean, color=[MODEL_COLORS.get(m, "gray") for m in models])
    plt.ylabel("mean delta_AIC (0 = best model in each file)")
    plt.title("Delta AIC across files (lower = better)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    basename1 = "summary_delta_AIC"
    plt.savefig(_out_path(output_folder, basename1 + ".png"), dpi=150)
    plt.close()

    origin_df1 = pd.DataFrame({"model": models, "delta_AIC_mean": delta_mean})
    _save_png_with_origin_csv(basename1, output_folder, origin_df1)

    # AIC weight: ファイルごとの100%積み上げ棒グラフ
    pivot = df.pivot_table(index="filename", columns="model",
                            values="AIC_weight", aggfunc="mean").fillna(0)
    pivot = pivot[[m for m in models if m in pivot.columns]]

    plt.figure(figsize=(max(7, len(pivot) * 0.6), 5))
    bottom = np.zeros(len(pivot))
    for m in pivot.columns:
        plt.bar(pivot.index, pivot[m], bottom=bottom,
                label=m, color=MODEL_COLORS.get(m, "gray"))
        bottom += pivot[m].values
    plt.ylabel("AIC weight (relative likelihood)")
    plt.title("AIC weight by file (stacked, sums to 1 per file)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    basename2 = "summary_AIC_weight_by_file"
    plt.savefig(_out_path(output_folder, basename2 + ".png"), dpi=150)
    plt.close()

    origin_df2 = pivot.reset_index()  # 列: filename, 各model(積み上げ棒グラフそのまま)
    _save_png_with_origin_csv(basename2, output_folder, origin_df2)


# ==========================================================
# グループ3: KS検定
# ==========================================================

def plot_ks_test(df, output_folder):
    models = _sorted_models(df["model"].unique())
    p_mean = [df[df.model == m]["KS_pvalue"].mean() for m in models]
    p_std = [df[df.model == m]["KS_pvalue"].std(ddof=0) for m in models]

    plt.figure(figsize=(7, 5))
    plt.bar(models, p_mean, yerr=p_std, capsize=4,
            color=[MODEL_COLORS.get(m, "gray") for m in models])
    plt.axhline(0.05, color="black", linestyle="--", linewidth=1,
                label="p = 0.05 (significance level)")
    plt.ylabel("KS test p-value (mean ± SD)")
    plt.title("Goodness-of-fit (KS test) across files")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    basename = "summary_KS_pvalue"
    plt.savefig(_out_path(output_folder, basename + ".png"), dpi=150)
    plt.close()

    origin_df = pd.DataFrame({
        "model": models, "KS_pvalue_mean": p_mean, "KS_pvalue_std": p_std,
    })
    _save_png_with_origin_csv(basename, output_folder, origin_df)


# ==========================================================
# グループ4: in-sample vs out-of-sample (過学習チェック)
# ==========================================================

def plot_insample_vs_cv(df, output_folder):
    if "logL_per_point" not in df.columns:
        print("  [注意] logL_per_point 列が見つかりません。"
              "最新版の NIMO_fitting_model_comparison.py で再実行すると表示されます。")
        return

    models = _sorted_models(df["model"].unique())
    x = np.arange(len(models))
    width = 0.35

    insample_mean = [df[df.model == m]["logL_per_point"].mean() for m in models]
    cv_mean = [df[df.model == m]["CV_mean_logpdf"].mean() for m in models]

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, insample_mean, width, label="In-sample (logL / n)",
            color="#55a868")
    plt.bar(x + width / 2, cv_mean, width, label="Out-of-sample (CV, held-out)",
            color="#c44e52")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Mean log-density per point (higher = better)")
    plt.title("In-sample vs. cross-validated fit\n"
              "(large gap = sign of overfitting)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    basename = "summary_insample_vs_CV"
    plt.savefig(_out_path(output_folder, basename + ".png"), dpi=150)
    plt.close()

    origin_df = pd.DataFrame({
        "model": models,
        "In-sample (logL / n)": insample_mean,
        "Out-of-sample (CV, held-out)": cv_mean,
    })
    _save_png_with_origin_csv(basename, output_folder, origin_df)


# ==========================================================
# グループ4b: Cross-validation (held-out log-density) の4モデル直接比較
# ==========================================================

def plot_cv_comparison(df, output_folder):
    """held-out (交差検証) log-densityで4モデルを直接比較する図を2種類出力する。

    CV_mean_logpdfは「held-outデータ1点あたりの平均log-density」なので、
    AIC/BICと違ってサンプル数nそのものには依存しない。ただし各ファイルの
    データの広がり(バブル径の分散)が違うと絶対値の水準そのものがずれるため、
    ファイルをまたいだ「素の平均値」の比較だけでは、ファイル間のばらつきに
    モデル間の差が埋もれてしまう可能性がある。そこで2種類の図を作る:

    (a) summary_CV_paired_by_file:
        各ファイルにつき4モデルのCV値を1本の折れ線でつなぐ（ペア比較）。
        同じファイル・同じhold-outデータでの比較になるので、ファイル間の
        水準の違いに影響されず、モデル間の優劣がそのまま線の上下として見える。
        太い赤の点・線は20ファイル平均±SD。

    (b) summary_delta_CV_by_model:
        各ファイル内で最良モデル(CVが最大)を基準にしたΔCV
        (= そのファイルの最良モデルのCV - 対象モデルのCV。0が最良、
        大きいほど悪い)を、モデルごとに20点(1ファイル1点)プロットしたもの。
        AICのdelta_AICと全く同じ発想をCVに適用したもので、n・データの
        広がりの両方に影響されにくい、最も公平な比較方法。
        Rayleigh–Normal mixtureが一貫して0付近(=最良)に来るかどうかで、
        AIC/BICで見えた優位性が過学習ではないことを示せる。
    """
    if "CV_mean_logpdf" not in df.columns:
        print("  [注意] CV_mean_logpdf 列が見つかりません。")
        return

    models = _sorted_models(df["model"].unique())
    pivot = df.pivot_table(index="filename", columns="model",
                            values="CV_mean_logpdf").reindex(columns=models)
    x = np.arange(len(models))

    # --- (a) ファイルごとに4モデルをつなぐ折れ線（ペア比較） ---
    plt.figure(figsize=(7, 5))
    for _, row in pivot.iterrows():
        plt.plot(x, row.values, color="gray", alpha=0.35, linewidth=1, zorder=1)
    mean_vals = pivot.mean(axis=0).values
    sd_vals = pivot.std(axis=0, ddof=0).values
    plt.errorbar(x, mean_vals, yerr=sd_vals, fmt="o-", color="#c44e52",
                 linewidth=2.5, markersize=8, capsize=4, zorder=2,
                 label="Mean ± SD across files (n=%d)" % len(pivot))
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Mean held-out log-density (higher = better)")
    plt.title("Cross-validated fit by model\n"
              "(each gray line = one file, paired across the 4 models)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    basename_a = "summary_CV_paired_by_file"
    plt.savefig(_out_path(output_folder, basename_a + ".png"), dpi=150)
    plt.close()

    origin_df_a = pivot.reset_index()
    summary_rows = pd.DataFrame({
        "filename": ["MEAN", "SD"],
        **{m: [pivot[m].mean(), pivot[m].std(ddof=0)] for m in models},
    })
    origin_df_a = pd.concat([origin_df_a, summary_rows], ignore_index=True)
    _save_png_with_origin_csv(basename_a, output_folder, origin_df_a)

    # --- (b) ファイル内最良モデル基準のΔCV（AICのdelta_AICと同じ考え方） ---
    df = df.copy()
    df["delta_CV"] = (df.groupby("filename")["CV_mean_logpdf"].transform("max")
                       - df["CV_mean_logpdf"])

    plt.figure(figsize=(7, 5))
    rng = np.random.default_rng(42)
    box_data = [df[df.model == m]["delta_CV"].values for m in models]
    bp = plt.boxplot(box_data, positions=x, widths=0.5, showfliers=False,
                      patch_artist=True, zorder=1)
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(MODEL_COLORS.get(m, "gray"))
        patch.set_alpha(0.25)
    for i, m in enumerate(models):
        vals = df[df.model == m]["delta_CV"].values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        plt.scatter(np.full(len(vals), x[i]) + jitter, vals,
                    color=MODEL_COLORS.get(m, "gray"), alpha=0.85, s=25,
                    zorder=2, edgecolor="white", linewidth=0.3)
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("delta_CV (best-model CV - model CV, per file; 0 = best)")
    plt.title("Cross-validated fit relative to the best model per file\n"
              "(%d files, lower = better)" % df["filename"].nunique())
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    basename_b = "summary_delta_CV_by_model"
    plt.savefig(_out_path(output_folder, basename_b + ".png"), dpi=150)
    plt.close()

    origin_df_b = df.pivot_table(index="filename", columns="model",
                                  values="delta_CV").reindex(columns=models).reset_index()
    _save_png_with_origin_csv(basename_b, output_folder, origin_df_b)


# ==========================================================
# グループ5: alphaの不確実性（フォレストプロット）
# ==========================================================

def plot_alpha_forest(alpha_df, output_folder):
    if alpha_df is None or alpha_df.empty:
        return
    alpha_df = alpha_df.sort_values("bootstrap_mean").reset_index(drop=True)
    y = np.arange(len(alpha_df))

    lower_col = [c for c in alpha_df.columns if c.startswith("CI_") and c.endswith("lower")][0]
    upper_col = [c for c in alpha_df.columns if c.startswith("CI_") and c.endswith("upper")][0]

    err_lower = alpha_df["bootstrap_mean"] - alpha_df[lower_col]
    err_upper = alpha_df[upper_col] - alpha_df["bootstrap_mean"]

    plt.figure(figsize=(7, max(4, len(alpha_df) * 0.4)))
    plt.errorbar(alpha_df["bootstrap_mean"], y,
                 xerr=[err_lower, err_upper],
                 fmt="o", color="#d62728", ecolor="gray", capsize=3)
    plt.yticks(y, alpha_df["filename"])
    plt.xlabel("alpha (Rayleigh mixture weight), mean ± 95% CI")
    plt.title("Uncertainty of alpha across files (bootstrap)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    basename = "summary_alpha_forest_plot"
    plt.savefig(_out_path(output_folder, basename + ".png"), dpi=150)
    plt.close()

    origin_df = pd.DataFrame({
        "filename": alpha_df["filename"],
        "alpha_mean": alpha_df["bootstrap_mean"],
        "CI_lower": alpha_df[lower_col],
        "CI_upper": alpha_df[upper_col],
        "err_lower (mean - CI_lower)": err_lower,
        "err_upper (CI_upper - mean)": err_upper,
    })
    _save_png_with_origin_csv(basename, output_folder, origin_df)


# ==========================================================
# グループ6: AICで最良となったモデルの頻度
# ==========================================================

def plot_best_model_frequency(df, output_folder):
    best_per_file = df.loc[df.groupby("filename")["AIC"].idxmin()]
    counts = best_per_file["model"].value_counts()
    models = _sorted_models(counts.index)
    counts = counts.reindex(models).fillna(0)

    plt.figure(figsize=(7, 5))
    plt.bar(models, counts.values, color=[MODEL_COLORS.get(m, "gray") for m in models])
    plt.ylabel("Number of files where model has lowest AIC")
    plt.title("Best-fitting model frequency (by AIC)")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    basename = "summary_best_model_frequency"
    plt.savefig(_out_path(output_folder, basename + ".png"), dpi=150)
    plt.close()

    origin_df = pd.DataFrame({"model": models, "count": counts.values})
    _save_png_with_origin_csv(basename, output_folder, origin_df)


# ==========================================================
# フォルダ選択（GUI）
# ==========================================================

COMPARISON_CSV_NAME = "all_files_model_comparison.csv"
ALPHA_CSV_NAME = "all_files_alpha_uncertainty.csv"


def select_folders():
    from tkinter import Tk, filedialog  # 実行時にのみ読み込む（GUI不要な用途での依存を避ける）
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    csv_folder = filedialog.askdirectory(
        title=f"{COMPARISON_CSV_NAME} が入ったフォルダを選択してください")
    if not csv_folder:
        print("CSVフォルダが選択されませんでした。処理を中止します。")
        sys.exit(1)

    output_folder = filedialog.askdirectory(
        title="グラフの出力先フォルダを選択してください")
    if not output_folder:
        print("出力フォルダが選択されませんでした。処理を中止します。")
        sys.exit(1)

    root.destroy()
    # tkinterはWindowsでもフォルダパスをスラッシュ("/")区切りで返すため、
    # 以降のos.path.join(バックスラッシュ)との混在を避けるために正規化する。
    return os.path.normpath(csv_folder), os.path.normpath(output_folder)


# ==========================================================
# メイン
# ==========================================================

def main():
    csv_folder, output_folder = select_folders()
    comp_path = os.path.join(csv_folder, COMPARISON_CSV_NAME)
    alpha_path = os.path.join(csv_folder, ALPHA_CSV_NAME)

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(comp_path):
        print(f"見つかりません: {comp_path}\n"
              f"（{COMPARISON_CSV_NAME} が選択したフォルダの直下にあるか確認してください）")
        sys.exit(1)

    df = pd.read_csv(comp_path)

    print("AIC / BIC の図を作成中...")
    plot_aic_bic(df, output_folder)

    print("delta_AIC / AIC重みの図を作成中...")
    plot_delta_aic_and_weight(df, output_folder)

    print("KS検定の図を作成中...")
    plot_ks_test(df, output_folder)

    print("in-sample vs 交差検証の図を作成中...")
    plot_insample_vs_cv(df, output_folder)

    print("交差検証(CV)による4モデル比較の図を作成中...")
    plot_cv_comparison(df, output_folder)

    print("最良モデルの頻度の図を作成中...")
    plot_best_model_frequency(df, output_folder)

    if os.path.exists(alpha_path):
        alpha_df = pd.read_csv(alpha_path)
        print("alphaの不確実性（フォレストプロット）を作成中...")
        plot_alpha_forest(alpha_df, output_folder)
    else:
        print(f"[注意] {alpha_path} が見つからないため alpha の図はスキップします。")

    print(f"\n✅ すべての比較図を出力しました → {output_folder}")


if __name__ == "__main__":
    main()
