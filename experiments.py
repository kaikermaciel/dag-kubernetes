"""
experiments.py — Executa todos os experimentos e gera os gráficos para o artigo.

Instâncias:
  |V| ∈ {5, 10, 20, 50}  ×  density ∈ {1.0, 1.5, 2.0}

Saídas:
  results/tabela_resultados.txt
  results/fig_convergencia.png
  results/fig_comparativo.png
  results/fig_melhoria.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from dag import generate_dag, instability_window
from algorithms import kahn_ordering, random_baseline, simulated_annealing

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Configuração dos experimentos ───────────────────────────────────────────

SIZES    = [5, 10, 20, 50]
DENSITIES = [1.0, 1.5, 2.0]
SA_PARAMS = dict(T0=1000.0, alpha=0.995, max_iter=5000, seed=42)
N_RANDOM  = 1000
SEED_DAG  = 7


# ─── Execução ────────────────────────────────────────────────────────────────

def run_all():
    rows = []  # para a tabela

    # instância representativa para curva de convergência
    conv_result = None
    conv_kahn   = None
    conv_label  = None

    print(f"{'|V|':>4}  {'density':>7}  {'Kahn':>10}  {'Rand(mean)':>12}  "
          f"{'Rand(best)':>12}  {'SA':>10}  {'Δ(%)':>8}  {'time(s)':>8}")
    print("-" * 80)

    for n in SIZES:
        for density in DENSITIES:
            G = generate_dag(n, density, seed=SEED_DAG)

            # Baseline Kahn
            kahn_order = kahn_ordering(G)
            kahn_score = instability_window(G, kahn_order)

            # Baseline Aleatório
            rand_res = random_baseline(G, n_samples=N_RANDOM, seed=42)

            # SA
            sa_res = simulated_annealing(G, **SA_PARAMS)

            # melhoria relativa em relação ao Kahn
            delta = (kahn_score - sa_res["best_score"]) / kahn_score * 100 \
                    if kahn_score > 0 else 0.0

            print(f"{n:>4}  {density:>7.1f}  {kahn_score:>10.1f}  "
                  f"{rand_res['mean']:>12.1f}  {rand_res['best_score']:>12.1f}  "
                  f"{sa_res['best_score']:>10.1f}  {delta:>7.1f}%  "
                  f"{sa_res['elapsed_time']:>8.2f}s")

            rows.append({
                "n": n, "density": density,
                "kahn": kahn_score,
                "rand_mean": rand_res["mean"],
                "rand_best": rand_res["best_score"],
                "sa": sa_res["best_score"],
                "delta": delta,
                "time": sa_res["elapsed_time"],
                "sa_history": sa_res["best_history"],
                "sa_conv": sa_res["history"],
                "kahn_score": kahn_score,
            })

            # guarda instância de n=20, density=1.5 para curva de convergência
            if n == 20 and density == 1.5:
                conv_result = sa_res
                conv_kahn   = kahn_score
                conv_label  = f"|V|=20, density=1.5"

    _save_table(rows)
    _plot_convergencia(conv_result, conv_kahn, conv_label)
    _plot_comparativo(rows)
    _plot_melhoria(rows)

    print(f"\nResultados salvos em: {RESULTS_DIR}/")


# ─── Tabela texto ─────────────────────────────────────────────────────────────

def _save_table(rows):
    path = os.path.join(RESULTS_DIR, "tabela_resultados.txt")
    header = (f"{'|V|':>4}  {'density':>7}  {'Kahn(s)':>10}  "
              f"{'Rand_mean(s)':>14}  {'Rand_best(s)':>14}  "
              f"{'SA(s)':>10}  {'Δ_kahn(%)':>10}  {'SA_time(s)':>10}\n")
    sep = "-" * 85 + "\n"
    with open(path, "w") as f:
        f.write(header)
        f.write(sep)
        for r in rows:
            f.write(
                f"{r['n']:>4}  {r['density']:>7.1f}  {r['kahn']:>10.1f}  "
                f"{r['rand_mean']:>14.1f}  {r['rand_best']:>14.1f}  "
                f"{r['sa']:>10.1f}  {r['delta']:>9.1f}%  {r['time']:>10.2f}\n"
            )


# ─── Figura 1: Curva de convergência ─────────────────────────────────────────

def _plot_convergencia(sa_res, kahn_score, label):
    fig, ax = plt.subplots(figsize=(7, 4))

    iters = range(len(sa_res["best_history"]))
    ax.plot(iters, sa_res["history"],
            color="#AAAACC", linewidth=0.8, alpha=0.7, label="SA (solução atual)")
    ax.plot(iters, sa_res["best_history"],
            color="#3C3489", linewidth=2.0, label="SA (melhor encontrado)")
    ax.axhline(kahn_score, color="#D85A30", linewidth=1.5,
               linestyle="--", label=f"Kahn (baseline) = {kahn_score:.0f}s")

    ax.set_xlabel("Iteração", fontsize=11)
    ax.set_ylabel("Janela de instabilidade (s)", fontsize=11)
    ax.set_title(f"Curva de Convergência do SA — {label}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig_convergencia.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [salvo] {path}")


# ─── Figura 2: Comparativo de algoritmos por tamanho ─────────────────────────

def _plot_comparativo(rows):
    # agrupa por tamanho, média sobre densidades
    by_size = {}
    for r in rows:
        n = r["n"]
        if n not in by_size:
            by_size[n] = {"kahn": [], "rand_mean": [], "sa": []}
        by_size[n]["kahn"].append(r["kahn"])
        by_size[n]["rand_mean"].append(r["rand_mean"])
        by_size[n]["sa"].append(r["sa"])

    sizes = sorted(by_size.keys())
    kahn_vals = [np.mean(by_size[n]["kahn"]) for n in sizes]
    rand_vals = [np.mean(by_size[n]["rand_mean"]) for n in sizes]
    sa_vals   = [np.mean(by_size[n]["sa"]) for n in sizes]

    x = np.arange(len(sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, kahn_vals, width, label="Kahn",     color="#D85A30", alpha=0.85)
    ax.bar(x,         rand_vals, width, label="Aleatório (média)", color="#AAAACC", alpha=0.85)
    ax.bar(x + width, sa_vals,   width, label="SA (proposto)",     color="#3C3489", alpha=0.9)

    ax.set_xlabel("Número de serviços |V|", fontsize=11)
    ax.set_ylabel("Janela de instabilidade média (s)", fontsize=11)
    ax.set_title("Comparativo entre Algoritmos — Média sobre densidades", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig_comparativo.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [salvo] {path}")


# ─── Figura 3: Melhoria relativa (%) por configuração ────────────────────────

def _plot_melhoria(rows):
    labels = [f"|V|={r['n']}\nd={r['density']}" for r in rows]
    deltas = [r["delta"] for r in rows]
    colors = ["#1D9E75" if d > 0 else "#D85A30" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(range(len(labels)), deltas, color=colors, alpha=0.85, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)

    for bar, val in zip(bars, deltas):
        ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 1.2
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Melhoria relativa ao Kahn (%)", fontsize=11)
    ax.set_title("Redução da Janela de Instabilidade: SA vs. Kahn", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig_melhoria.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [salvo] {path}")


if __name__ == "__main__":
    run_all()
