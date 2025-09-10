import os
import pandas as pd
import numpy as np
import quapy as qp
from scipy.stats import wilcoxon

# === CONFIGURATION ===
DATA_DIR = "results/ucimulti"

methods = ['EM','EM_BCTS','PSEM','TSEM','DMAPEM','DEM','CSEM','EREM']
labels = ['EMQ','Calib','Smooth','Temp','MAP','Damp','Conf','Ent']
classifiers = ["LR", "NN"]
datasets = qp.datasets.UCI_MULTICLASS_DATASETS

# === COLLECT D(b) VALUES ===
deltas = []

for clf in classifiers:
    for dataset in datasets:
        try:
            mae_em = pd.read_csv(os.path.join(DATA_DIR, f"EM_{clf}_{dataset}.dataframe"))["mae"]
        except FileNotFoundError:
            print(f"[!] Missing EMQ file for {clf} / {dataset}. Skipping.")
            continue

        for method in methods:
            if method == "EM":
                continue
            filename = f"{method}_{clf}_{dataset}.dataframe"
            filepath = os.path.join(DATA_DIR, filename)
            if not os.path.exists(filepath):
                print(f"[!] Missing file: {filename}")
                continue

            mae_other = pd.read_csv(filepath)["mae"]
            D = mae_other - mae_em
            D = D.dropna() 
            for d in D:
                deltas.append({
                    "Classifier": clf,
                    "Dataset": dataset,
                    "Heuristic": method,
                    "Delta": d
                })

# === AGGREGATE AND ANALYZE ===
df = pd.DataFrame(deltas)

results = []

for clf in classifiers:
    for method,label in zip(methods,labels):
        if method == "EM":
            continue
        subset = df[(df["Classifier"] == clf) & (df["Heuristic"] == method)]
        D = subset["Delta"]
        mean = D.mean()
        std = D.std()
        win_ratio = (D < 0).mean()
        try:
            pval = wilcoxon(D, alternative='less').pvalue
        except ValueError:
            pval = np.nan

        results.append({
            "Classifier": clf,
            "Heuristic": label,
            "Mean (MAE)": mean,
            "Std": std,
            "% Wins": win_ratio,
            "Wilcoxon p-value": "{:.2e}".format(pval)
        })

# Save final summary
summary_df = pd.DataFrame(results)

# Function to format p-values for LaTeX (bold if < 0.05)
def format_p(p):
    try:
        p = float(p)
        if p < 1e-4:
            return r"$\bm{<0.0001}$"   # always bold
        return f"\\bm{{{p:.4f}}}" if p < 0.05 else f"{p:.4f}"
    except:
        return str(p)

# Round values and format p-values
df_latex = summary_df.copy()
df_latex["Mean (MAE)"] = df_latex["Mean (MAE)"].astype(float).round(5)
df_latex["Std"] = df_latex["Std"].astype(float).round(5)
df_latex["% Wins"] = (df_latex["% Wins"]).round(2).astype(str)
df_latex["Wilcoxon p-value"] = df_latex["Wilcoxon p-value"].apply(format_p)

# Sort for presentation
#df_latex = df_latex.sort_values(by=["Classifier", "Mean Δ (MAE)"])

# Save to LaTeX
latex_table = df_latex.to_latex(index=False, escape=False, column_format="llcccc", float_format="%.5f")
with open("bag_level_summary.tex", "w") as f:
    f.write(latex_table)

