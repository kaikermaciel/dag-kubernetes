"""
dag.py — Modelagem do grafo de dependências de deployment

G = (V, E, w)
  V : microsserviços
  E : (v_i, v_j) significa "v_i deve subir antes de v_j"
  w : tempo de startup de cada serviço (segundos)
"""

import random
import networkx as nx


# ─────────────────────────────────────────────
# Geração de instâncias sintéticas
# ─────────────────────────────────────────────

def generate_dag(n_services: int, edge_density: float, seed: int = 42) -> nx.DiGraph:
    """
    Gera um DAG sintético com n_services vértices e densidade aproximada
    de arestas edge_density * n_services.

    Garante aciclicidade construindo arestas apenas de índices menores
    para maiores (ordem topológica implícita na construção).
    """
    rng = random.Random(seed)
    G = nx.DiGraph()

    # vértices com tempo de startup entre 5s e 120s
    for i in range(n_services):
        G.add_node(i, weight=rng.uniform(5, 120), label=f"svc-{i}")

    # arestas: para cada par (i, j) com i < j, adiciona com probabilidade p
    target_edges = int(edge_density * n_services)
    candidate_edges = [(i, j) for i in range(n_services)
                       for j in range(i + 1, n_services)]
    rng.shuffle(candidate_edges)
    for u, v in candidate_edges[:target_edges]:
        G.add_edge(u, v)

    return G


def startup_time(G: nx.DiGraph, node: int) -> float:
    return G.nodes[node]["weight"]


# ─────────────────────────────────────────────
# Função objetivo
# ─────────────────────────────────────────────

def instability_window(G: nx.DiGraph, ordering: list) -> float:
    """
    Calcula a janela de instabilidade total f(π).

    Para cada aresta (v_i, v_j) ∈ E:
      inst(v_i, v_j) = t_start(v_j) - t_end(v_i)
    onde t_end(v_i) = Σ w(v_k) para k até posição de v_i (inclusive).

    f(π) = Σ inst sobre todas as arestas.
    """
    # tempo de início acumulado de cada serviço
    pos = {v: idx for idx, v in enumerate(ordering)}
    t_start = {}
    acc = 0.0
    for v in ordering:
        t_start[v] = acc
        acc += startup_time(G, v)

    total = 0.0
    for (u, v) in G.edges():
        t_end_u = t_start[u] + startup_time(G, u)
        total += t_start[v] - t_end_u  # tempo que v_j espera após v_i terminar

    return total


def is_valid_ordering(G: nx.DiGraph, ordering: list) -> bool:
    """Verifica se a ordering respeita todas as dependências do DAG."""
    pos = {v: idx for idx, v in enumerate(ordering)}
    for (u, v) in G.edges():
        if pos[u] >= pos[v]:
            return False
    return True
