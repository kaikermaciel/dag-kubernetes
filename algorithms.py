"""
algorithms.py — Baselines e Simulated Annealing sobre DAGs

Baselines:
  - kahn_ordering   : ordenação topológica padrão (O(V+E))
  - random_ordering : amostragem aleatória de extensões lineares válidas

Proposto:
  - simulated_annealing : SA com vizinhança ciente do DAG
"""

import math
import random
import time
from collections import deque

import networkx as nx

from dag import instability_window, is_valid_ordering, startup_time


# ─────────────────────────────────────────────
# Baseline 1: Kahn
# ─────────────────────────────────────────────

def kahn_ordering(G: nx.DiGraph) -> list:
    """
    Ordenação topológica de Kahn.
    Quando há empate (múltiplos nós sem predecessores prontos),
    escolhe o de menor índice — comportamento determinístico e representativo
    do que um operador faria manualmente.
    """
    in_degree = {v: G.in_degree(v) for v in G.nodes()}
    queue = deque(sorted([v for v, d in in_degree.items() if d == 0]))
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for successor in sorted(G.successors(node)):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)
                queue = deque(sorted(queue))

    return order


# ─────────────────────────────────────────────
# Baseline 2: Aleatório
# ─────────────────────────────────────────────

def random_ordering(G: nx.DiGraph, seed: int = None) -> list:
    """
    Gera uma extensão linear aleatória do DAG usando o algoritmo de
    amostragem uniforme de Kahn com desempate aleatório.
    """
    rng = random.Random(seed)
    in_degree = {v: G.in_degree(v) for v in G.nodes()}
    available = [v for v, d in in_degree.items() if d == 0]
    order = []

    while available:
        node = rng.choice(available)
        available.remove(node)
        order.append(node)
        for successor in G.successors(node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                available.append(successor)

    return order


def random_baseline(G: nx.DiGraph, n_samples: int = 1000, seed: int = 42) -> dict:
    """
    Amostra n_samples ordenações aleatórias válidas e retorna estatísticas.
    """
    scores = []
    best_order = None
    best_score = float("inf")

    for i in range(n_samples):
        order = random_ordering(G, seed=seed + i)
        score = instability_window(G, order)
        scores.append(score)
        if score < best_score:
            best_score = score
            best_order = order

    return {
        "best_order": best_order,
        "best_score": best_score,
        "mean": sum(scores) / len(scores),
        "std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5,
        "all_scores": scores,
    }


# ─────────────────────────────────────────────
# Vizinhança ciente do DAG
# ─────────────────────────────────────────────

def _is_topologically_valid(G: nx.DiGraph, ordering: list) -> bool:
    """Verifica validade topológica em O(V+E)."""
    pos = {v: idx for idx, v in enumerate(ordering)}
    return all(pos[u] < pos[v] for u, v in G.edges())


def _dag_neighbor(G: nx.DiGraph, ordering: list,
                  rng: random.Random, max_attempts: int = 100) -> list:
    """
    Gera um vizinho válido por swap de dois elementos escolhidos
    aleatoriamente. Verifica validade completa após cada tentativa.
    """
    n = len(ordering)
    for _ in range(max_attempts):
        i, j = sorted(rng.sample(range(n), 2))
        neighbor = ordering[:]
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        if _is_topologically_valid(G, neighbor):
            return neighbor
    # fallback: retorna o próprio estado sem modificação
    return ordering[:]


# ─────────────────────────────────────────────
# Simulated Annealing
# ─────────────────────────────────────────────

def simulated_annealing(
    G: nx.DiGraph,
    T0: float = 1000.0,
    alpha: float = 0.995,
    max_iter: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Simulated Annealing para minimizar a janela de instabilidade
    sobre o espaço de extensões lineares do DAG.

    Parâmetros
    ----------
    G        : DAG de dependências
    T0       : temperatura inicial
    alpha    : fator de resfriamento (T ← T * alpha a cada iteração)
    max_iter : número máximo de iterações
    seed     : semente aleatória para reprodutibilidade

    Retorna
    -------
    dict com: best_order, best_score, history (curva de convergência),
              elapsed_time, n_accepted, n_rejected
    """
    rng = random.Random(seed)

    # solução inicial via Kahn
    current = kahn_ordering(G)
    current_score = instability_window(G, current)

    best = current[:]
    best_score = current_score

    T = T0
    history = [current_score]   # curva de convergência
    best_history = [best_score]

    n_accepted = 0
    n_rejected = 0

    t0 = time.time()

    for iteration in range(max_iter):
        neighbor = _dag_neighbor(G, current, rng)
        neighbor_score = instability_window(G, neighbor)

        delta = neighbor_score - current_score

        if delta < 0:
            # melhora: aceita sempre
            current = neighbor
            current_score = neighbor_score
            n_accepted += 1
        else:
            # piora: aceita com probabilidade de Boltzmann
            prob = math.exp(-delta / T) if T > 1e-10 else 0.0
            if rng.random() < prob:
                current = neighbor
                current_score = neighbor_score
                n_accepted += 1
            else:
                n_rejected += 1

        if current_score < best_score:
            best = current[:]
            best_score = current_score

        history.append(current_score)
        best_history.append(best_score)

        T *= alpha

    elapsed = time.time() - t0

    return {
        "best_order": best,
        "best_score": best_score,
        "history": history,
        "best_history": best_history,
        "elapsed_time": elapsed,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "final_temp": T,
    }
