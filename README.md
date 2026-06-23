# DAG Deployment Ordering — SBSE sobre Grafos de Dependência

Trabalho Final — ICC041 Introdução à Teoria dos Grafos / PPGINF539 — UFAM 2026/01

## Problema

Em clusters Kubernetes com microsserviços interdependentes, a **ordem de deployment** durante um *rolling update* influencia diretamente a **janela de instabilidade** do sistema — o intervalo de tempo em que serviços em versões distintas coexistem.

O problema é modelado como:
- **Grafo:** G = (V, E, w) — DAG onde vértices são serviços, arestas são dependências e pesos são tempos de startup
- **Objetivo:** encontrar a permutação π ∈ L(G) (extensão linear do DAG) que minimiza f(π) = Σ instabilidade sobre todas as arestas
- **Complexidade:** NP-hard (redução do problema de escalonamento em DAGs)

## Estrutura

```
dag_deployment/
├── dag.py          # modelagem do grafo, função objetivo
├── algorithms.py   # Kahn (baseline), Aleatório, Simulated Annealing
├── experiments.py  # executa todos os cenários e gera gráficos
├── results/        # gerado automaticamente ao rodar experiments.py
│   ├── tabela_resultados.txt
│   ├── fig_convergencia.png
│   ├── fig_comparativo.png
│   └── fig_melhoria.png
└── README.md
```

## Instalação

```bash
pip install networkx matplotlib numpy
```

## Execução

```bash
# Roda todos os experimentos e salva os gráficos em results/
python experiments.py
```

## Algoritmo Proposto: Simulated Annealing

```
Entrada: DAG G=(V,E,w), T0, alpha, max_iter
Saída:   melhor ordenação π*

π ← kahn_ordering(G)          # solução inicial válida
T ← T0

para i = 1 até max_iter:
    π' ← vizinho_válido(π, G)  # swap que preserva ordem topológica
    Δ  ← f(π') - f(π)

    se Δ < 0:
        π ← π'                 # aceita melhora sempre
    senão:
        aceitar π' com prob. exp(-Δ/T)

    T ← T × alpha              # resfriamento

retornar π*
```

A **função de vizinhança** seleciona dois serviços aleatoriamente e os troca de posição somente se a permutação resultante for topologicamente válida (verificado em O(V+E)).

## Resultados

| \|V\| | Density | Kahn (s) | SA (s) | Melhoria |
|------:|--------:|---------:|-------:|---------:|
|     5 |     1.0 |    124.6 |  122.1 |    2.0%  |
|    10 |     1.0 |    881.8 |  390.7 |   55.7%  |
|    20 |     1.0 |   4960.3 | 2216.4 |   55.3%  |
|    50 |     1.0 |  43599.2 |16512.5 |   62.1%  |

O SA supera Kahn em todas as instâncias com |V| ≥ 10, com ganhos crescentes conforme o grafo cresce — evidenciando que o espaço de extensões lineares exploráveis aumenta com o tamanho do DAG.

## Referências

- Harman & Jones (2001). *Search-Based Software Engineering*. IST.
- Harman et al. (2012). *SBSE: Trends, Techniques and Applications*. ACM CSUR.
- Brogi et al. (2019). *Optimal and Automated Deployment for Microservices*. arXiv:1901.09782.
- He et al. (2022). *Online Deployment Algorithms for Microservice Systems*. IEEE TCC.
- Guerrero et al. (2024). *GA for Multi-Objective Optimization of Container Allocation*. arXiv:2401.12698.
- Roh et al. (2024). *OOSP: Pod Deployment Optimization*. Sensors.
- Ullman (1975). *NP-complete Scheduling Problems*. JCSS.
- Kahn (1962). *Topological Sorting of Large Networks*. CACM.
