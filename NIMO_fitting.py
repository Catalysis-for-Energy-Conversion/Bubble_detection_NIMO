#制約条件ありで、正規分布の初期値をいくつか変えてみる。
#ImageJ用。NIMOと違って列番号の指定が11列目。あと横軸


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, rayleigh
from scipy.optimize import curve_fit
from scipy.optimize import minimize

from tkinter import filedialog, Tk
import pathlib

# 合成確率密度関数（Rayleigh + Normal）
def rayleigh_modified_pdf(x, loc_r, scale_r):
    pdf = rayleigh.pdf(x, loc=loc_r, scale=scale_r)
    # loc_r より小さい x に対して 0 にする
    modified_pdf = np.where(x < loc_r, 0, pdf)
    return modified_pdf


def mixed_pdf(x, w, loc_r, scale_r, mu_n, sigma_n):
    return w * rayleigh_modified_pdf (x, loc_r=loc_r, scale_r=scale_r) + (1 - w) * norm.pdf(x, loc=mu_n, scale=sigma_n)

# フォルダ選択（データフォルダ）
Tk().withdraw()
input_folder = filedialog.askdirectory(title="CSVファイルが入ったフォルダを選択")

#input_folder = r'C:\Users\mizuk\OneDrive - The University of Tokyo\CatEc\Bubble counting\NIMO\Fitting check NIMO 200 mA 250905\PDF folder'


# 出力フォルダ指定（自動作成）
output_folder = filedialog.askdirectory(title="出力フォルダを選択")
#output_folder = r'C:\Users\mizuk\OneDrive - The University of Tokyo\CatEc\Bubble counting\NIMO\Fitting check NIMO 200 mA 250905\Output'
fit_param_path = os.path.join(output_folder, "fit_results.csv")

# 結果の記録用
results = []


# すべてのCSVを処理
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        full_path = os.path.join(input_folder, filename)
        try:
            # データ読み込み（11列目の数値データ）
            data = 10*pd.read_csv(full_path).iloc[:, 3].dropna().values  #NIMOの時に使う
            # data = pd.read_csv(full_path).iloc[:, 10].dropna().values    #ImageJの時に使う
            
            # ヒストグラム作成
            bins = np.arange(0, max(data) + 2, 1)
            counts, bin_edges = np.histogram(data, bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2       
            

            #a_para = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            mu_init_est = data.mean()
            print(mu_init_est)
            # mu_init= [90, 100, 110, 120, 130, 140, 150, 160, 170]            
            mu_init= [mu_init_est -25, mu_init_est -20, mu_init_est - 10, mu_init_est, mu_init_est+10, mu_init_est +20, mu_init_est+ 30, ]

            loss_values = []
            best_result = None
            best_loss = np.inf
            best_params = None
            
            # 誤差（残差の2乗和）
            def loss(params, x, y):
                a, loc_r, scale_r, mu_n, sigma_n = params
                y_pred = mixed_pdf(x, a, loc_r, scale_r, mu_n, sigma_n)
                return np.sum((y - y_pred) ** 2)
            
            # 制約: μ - 3σ > 0 → μ - 3σ - ε ≥ 0 （εはゼロ除け）
            def constraint_mu_3sigma(params):
                a, loc_r, scale_r, mu_n, sigma_n = params
                return mu_n - 3 * sigma_n
            
            # 制約の定義（非線形制約）
            cons = ({
                'type': 'ineq',
                'fun': constraint_mu_3sigma
            })

            for mu_init_i in mu_init:
                loc_init = [0, 20, 40, 50, 70, 90]
                for loc_init_i in loc_init:
                    
                
                    # パラメータの初期値
                    p0 = [0.5, loc_init_i, 10 , mu_init_i, 2]  # a, loc_r, scale_r, mu_n, sigma_n
                    
                    # バウンド（全パラメータが意味ある範囲に）
                    bounds = [
                        (0, 1),         # a ∈ [0,1]
                        (0, mu_init_i-10),      # loc_r ≥ 0
                        (0.1, 100),    # scale_r > 0
                        (mu_init_i-10, None),      # mu_n ≥ 0
                        (0.1, None),    # sigma_n > 0
                    ]
                    
                    # x, y: 実データ
                    # x = bin_centers  # ヒストグラムの bin 中心
                    x = np.arange(0, int(np.ceil(max(data))) + 1, 1)
                    y = counts  # bin に対する密度（正規化）
                    
                    # print(len(x))
                    # print(len(y)) 
                    # 最適化実行
    
                    # print(p0)
    
                    # try: 
                    result = minimize(loss, p0, args=(x, y), bounds=bounds, constraints=cons)
                    int_params = result.x
    
                    print(mu_init_i, loss(int_params , x, y),int_params[0])
                    
                    if result.success and result.fun < best_loss:
                        best_loss = result.fun
                        best_result = result
                        best_params = result.x
                        
                    # except Exception as e:
                        # print(f"{filename}: mu_init={mu_init} でエラー: {e}")
                    
                    # print(best_params)
    
                # 最適なパラメータ
            fitted_params = best_params
            print(fitted_params)

                
            
  # PDF値の出力: 0からmax(data)まで1刻み
             
            bin_edges = np.arange(0, max(data) + 2, 1)  # +2 to include last edge
            hist_counts, _ = np.histogram(data, bins=bin_edges)
            
            # ステップ2: PDF計算（同じ x 軸で）
            # x_pdf = bin_edges[:-1]  # 
            # y_pdf = mixed_pdf(x_pdf, *params) 
            
            x_pdf = np.arange(0, max(data) + 1, 1)
            y_pdf = mixed_pdf(x_pdf, *fitted_params)

            # print('x_pdf=', len(x_pdf))            
            # print('y_pdf=', len(y_pdf))

            # 図の描画と保存
            x = np.arange(0, int(np.ceil(max(data))) + 1, 1)
            y_fit = mixed_pdf(x, *fitted_params)
            y_reyleigh = fitted_params[0]*rayleigh_modified_pdf(x, fitted_params[1], fitted_params[2])
            
            # y_reyleigh = rayleigh.pdf(x, params[1], params[2])
            y_norm = (1-fitted_params[0])* norm.pdf(x, fitted_params[3], fitted_params[4])

            
            pdf_table = pd.DataFrame({'x': x_pdf, 'Rayleigh PDF': y_reyleigh,'Normal PDF': y_norm,'mixed PDF': y_pdf, 'Frequency':counts,'Histogram':hist_counts})
            pdf_filename = os.path.splitext(filename)[0] + "_pdf_values.csv"
            pdf_table.to_csv(os.path.join(output_folder, pdf_filename), index=False)


            # 結果記録
            results.append([filename, *fitted_params])

            
            
            plt.figure()
            plt.hist(data, bins, density=True, alpha=0.5, label='Data')
            
            plt.plot(x, y_reyleigh, 'b-', label='Rayleigh PDF')
            plt.plot(x, y_norm, 'g-', label='Normal PDF')
            plt.plot(x, y_fit, 'r-', label='Fitted PDF')

            # print(len(x))
            # print('y_fit=', len(y_fit))            
            # print('y_rayleigh=', len(y_reyleigh))
            # print('y_norm=', len(y_norm))

            
            plt.title(f'Fit: {filename}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            plot_filename = os.path.splitext(filename)[0] + "_fit.png"
            plt.savefig(os.path.join(output_folder, plot_filename))
            plt.show()
           
            plt.close()
            
            # --- パラメータの出力 ---
            print(f"Fitted parameters:")
            print(f"  Weight (Rayleigh): {fitted_params[0]:.3f}")
            print(f"  Rayleigh loc:      {fitted_params[1]:.3f}")
            print(f"  Rayleigh scale:    {fitted_params[2]:.3f}")
            print(f"  Normal mean:       {fitted_params[3]:.3f}")
            print(f"  Normal std dev:    {fitted_params[4]:.3f}")
            print(f"  Sum error:    {loss(fitted_params, x, y):.3f}")
            

        except Exception as e:
            print(f"{filename} の解析中にエラー: {e}")

# CSVにパラメータ出力
columns = ["filename", "weight_rayleigh", "loc_rayleigh", "scale_rayleigh", "mu_normal", "sigma_normal"]
df_results = pd.DataFrame(results, columns=columns)
df_results.to_csv(fit_param_path, index=False)

print(f"\n✅ 全ファイルの解析が完了しました。結果は以下に保存されました：\n→ {fit_param_path}")
