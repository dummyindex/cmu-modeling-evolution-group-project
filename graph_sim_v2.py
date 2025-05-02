# Re-import libraries after kernel reset
import json
from pathlib import Path
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import multiprocessing

import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Graph-based simulation of cultural evolution")
parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
parser.add_argument('--graph_N', type=int, default=1000, help='Number of nodes in the graph')
parser.add_argument('--total_trials', type=int, default=2000, help='Total number of trials to run')
parser.add_argument('--k_order', type=int, default=4, help='Order of neighborhood for learning')
parser.add_argument('--k_order_interval', type=int, default=1, help='Interval for k_order')
parser.add_argument('--stable_criterion', type=int, default=500, help='Number of iterations for stability check')
parser.add_argument('--generations', type=int, default=1000, help='Number of generations to simulate')
parser.add_argument('--stable_trial_threshold', type=int, default=500, help='Number of trials for stability check')
args = parser.parse_args()


# Define ModelState and ModelParameters classes
class ModelState:
    def __init__(self, x1, x2, x3, x4):
        self.x = np.array([x1, x2, x3, x4])
        self.normalize()

    def normalize(self):
        total = np.sum(self.x)
        if total > 0:
            self.x = self.x / total

    def clone(self):
        return ModelState(*self.x)

class ModelParameters:
    def __init__(self, mu_M, mu_m, p_o, mode, n_tutors, s1, s2, k_order):
        self.innovation_rate_M = mu_M
        self.innovation_rate_m = mu_m
        self.oblique_learning_prob = p_o
        self.learning_mode = mode
        self.n_tutors = n_tutors
        self.s1 = s1
        self.s2 = s2
        self.k_order = k_order

# Define graph generation functions
def make_small_world_graph(): return nx.watts_strogatz_graph(args.graph_N, 4, 0.1)
def make_erdos_renyi_graph(): return nx.erdos_renyi_graph(args.graph_N, 0.1)
def make_barabasi_albert_graph(): return nx.barabasi_albert_graph(args.graph_N, 2)
def make_complete_graph(): return nx.complete_graph(args.graph_N)
def make_cycle_graph(): return nx.cycle_graph(args.graph_N)

# Helper functions for graph neighborhood
# We assume the neighborhood never changes
cached_neighbors = {}
def get_k_order_neighbors(graph, node, k, include_self=True):
    if (graph, node, k) in cached_neighbors:
        return cached_neighbors[(graph, node, k, include_self)]
    if include_self:
        res = list(nx.single_source_shortest_path_length(graph, node, cutoff=k).keys())
        cached_neighbors[(graph, node, k, include_self)] = res
        return res
    else:
        # Get all nodes within k distance
        neighbors = list(nx.single_source_shortest_path_length(graph, node, cutoff=k).keys())
        # Remove the node itself
        neighbors.remove(node)
        res = neighbors
        cached_neighbors[(graph, node, k, include_self)] = res
        return neighbors

# # Learning mechanisms
def conformist_learning_k(graph, node, state_v, all_states, p_o, n, k):
    neighbors = get_k_order_neighbors(graph, node, k)
    if not neighbors:
        return state_v

    # Aggregate neighbor frequencies
    freq_R = np.mean([all_states[n].x[0] + all_states[n].x[1] for n in neighbors])
    freq_r = np.mean([all_states[n].x[2] + all_states[n].x[3] for n in neighbors])

    def prob_majority(freq):
        return sum(math.comb(n, m) * (freq ** m) * ((1 - freq) ** (n - m)) for m in range(n // 2 + 1, n + 1))

    f_R = prob_majority(freq_R)
    f_r = prob_majority(freq_r)

    # Apply vertical + oblique update
    x1_v, x2_v, x3_v, x4_v = state_v.x

    new_x1 = (1 - p_o) * x1_v + p_o * (x1_v + x3_v) * f_R
    new_x2 = (1 - p_o) * x2_v + p_o * (x2_v + x4_v) * f_R
    new_x3 = (1 - p_o) * x3_v + p_o * (x1_v + x3_v) * f_r
    new_x4 = (1 - p_o) * x4_v + p_o * (x2_v + x4_v) * f_r

    return ModelState(new_x1, new_x2, new_x3, new_x4)

def random_oblique_learning_k(graph, node, state_v, all_states, p_o, k):
    neighbors = get_k_order_neighbors(graph, node, k)
    if not neighbors:
        return state_v
    R_freq = np.mean([all_states[n].x[0] + all_states[n].x[1] for n in neighbors])
    r_freq = np.mean([all_states[n].x[2] + all_states[n].x[3] for n in neighbors])
    x1, x2, x3, x4 = state_v.x
    return ModelState(
        (1 - p_o) * x1 + p_o * (x1 + x3) * R_freq,
        (1 - p_o) * x2 + p_o * (x2 + x4) * R_freq,
        (1 - p_o) * x3 + p_o * (x1 + x3) * r_freq,
        (1 - p_o) * x4 + p_o * (x2 + x4) * r_freq
    )

def success_biased_learning_k(graph, node, state_v, all_states, p_o, n, k, env):
    neighbors = get_k_order_neighbors(graph, node, k)
    if not neighbors:
        return state_v
    R_freq = np.mean([all_states[n].x[0] + all_states[n].x[1] for n in neighbors])
    r_freq = np.mean([all_states[n].x[2] + all_states[n].x[3] for n in neighbors])
    def prob_at_least_one(freq): return 1 - (1 - freq) ** n
    def prob_all(freq): return freq ** n
    x1, x2, x3, x4 = state_v.x
    if env == 0:
        return ModelState(
            (1 - p_o) * x1 + p_o * (x1 + x3) * prob_all(R_freq),
            (1 - p_o) * x2 + p_o * (x2 + x4) * prob_all(R_freq),
            (1 - p_o) * x3 + p_o * (x1 + x3) * prob_at_least_one(r_freq),
            (1 - p_o) * x4 + p_o * (x2 + x4) * prob_at_least_one(r_freq)
        )
    else:
        return ModelState(
            (1 - p_o) * x1 + p_o * (x1 + x3) * prob_at_least_one(R_freq),
            (1 - p_o) * x2 + p_o * (x2 + x4) * prob_at_least_one(R_freq),
            (1 - p_o) * x3 + p_o * (x1 + x3) * prob_all(r_freq),
            (1 - p_o) * x4 + p_o * (x2 + x4) * prob_all(r_freq)
        )

# Innovation and selection
def innovation(state, params):
    x1, x2, x3, x4 = state.x
    mu_M, mu_m = params.innovation_rate_M, params.innovation_rate_m
    return ModelState(
        (1 - mu_M) * x1 + mu_M * x3,
        (1 - mu_m) * x2 + mu_m * x4,
        (1 - mu_M) * x3 + mu_M * x1,
        (1 - mu_m) * x4 + mu_m * x2
    )

def selection(state, environment, params):
    x1, x2, x3, x4 = state.x
    if environment == 0:
        w_bar = x1 + x2 + (x3 + x4) * (1 + params.s1)
        return ModelState(x1/w_bar, x2/w_bar, x3*(1+params.s1)/w_bar, x4*(1+params.s1)/w_bar)
    else:
        w_bar = (x1 + x2)*(1 + params.s2) + x3 + x4
        return ModelState(x1*(1+params.s2)/w_bar, x2*(1+params.s2)/w_bar, x3/w_bar, x4/w_bar)


def mating(state):
    """Implement random mating according to Table 2"""
    x1, x2, x3, x4 = state.x
    
    # Calculate offspring frequencies after mating
    new_x1 = x1**2 + (x1*x2 + x1*x3 + x1*x4/2 + x2*x3/2)
    new_x2 = x2**2 + (x1*x2 + x2*x4 + x1*x4/2 + x2*x3/2)
    new_x3 = x3**2 + (x1*x3 + x3*x4 + x1*x4/2 + x2*x3/2)
    new_x4 = x4**2 + (x2*x4 + x3*x4 + x1*x4/2 + x2*x3/2)
    
    new_state = ModelState(new_x1, new_x2, new_x3, new_x4)
    new_state.normalize()
    return new_state

def vertical_learning(state):
    """Implement vertical learning (incorporated into mating)"""
    return mating(state)

# Continue from setup: define parallel simulation and run across parameter combinations
def simulate_parameter_combo(args, stable_criterion=args.stable_criterion, generations=args.generations, stable_trial_threshold=args.stable_trial_threshold):
    graph_func, graph_name, k_order, learning_mode, c, total_trials = args
    G = graph_func()
    mu_init = np.random.uniform(0.01, 0.1)
    mu = mu_init
    stable_trial_count = 0 # According to Carja et al. [35] 
    stable = False
    trial_count = 0
    while not stable and trial_count < total_trials:
        trial_count += 1
        mu_mut = mu * np.random.exponential(1.0)
        mu_mut = min(0.999, max(0.0001, mu_mut))
        params = ModelParameters(mu, mu_mut, 0.4, learning_mode, 5, 1.0, 1.0, k_order)

        for node in G.nodes():
            G.nodes[node]['state'] = ModelState(0.5 - 0.001 / 2, 0.001 / 2,
                                                0.5 - 0.001 / 2, 0.001 / 2)
        env = 0
        for gen in range(1000):
            if gen % c == 0:
                env = 1 - env
            prev_states = {n: G.nodes[n]['state'] for n in G.nodes()}
            pref_states_after_vertical = {n: vertical_learning(prev_states[n]) for n in G.nodes()}
            next_states = {}
            for node in G.nodes():
                sv = prev_states[node]
                if learning_mode == 'conformist':
                    learned = conformist_learning_k(G, node, sv, pref_states_after_vertical, 0.4, 5, k_order)
                elif learning_mode == 'random':
                    learned = random_oblique_learning_k(G, node, sv, pref_states_after_vertical, 0.4, k_order)
                elif learning_mode == 'success':
                    learned = success_biased_learning_k(G, node, sv, pref_states_after_vertical, 0.4, 5, k_order, env)
                else:
                    learned = sv
                innovated = innovation(learned, params)
                selected = selection(innovated, env, params)
                next_states[node] = selected
            for node in G.nodes():
                G.nodes[node]['state'] = next_states[node]

        m_freq = np.mean([G.nodes[n]['state'].x[1] + G.nodes[n]['state'].x[3] for n in G.nodes()])
        if m_freq > 0.05:
            mu = mu_mut
            stable_trial_count = 0
            break
        else:
            stable_trial_count += 1
            if stable_trial_count >= stable_trial_threshold:
                stable = True
                return {
                    "graph_type": graph_name,
                    "learning_mode": learning_mode,
                    "env_period": c,
                    "init_rate": mu_init,
                    "optimal_rate": mu,
                    "k_order": k_order,
                    "stable": stable,
                    "trial_count": trial_count,
                }
    return {
        "graph_type": graph_name,
        "learning_mode": learning_mode,
        "env_period": c,
        "init_rate": mu_init,
        "optimal_rate": mu,
        "k_order": k_order,
        "stable": stable,
        "trial_count": trial_count,
    }

# Setup parameters
graph_types = {
    "Watts-Strogatz": make_small_world_graph,
    "Erdős-Rényi": make_erdos_renyi_graph,
    "Barabási-Albert": make_barabasi_albert_graph,
    "Complete": make_complete_graph,
    "Cycle": make_cycle_graph
}
learning_modes = ['conformist', 'random', 'success']
env_periods = list(range(5, 41, 5))

param_grid = []
for graph_name, graph_func in graph_types.items():
    for mode in learning_modes:
        for c in env_periods:
            for k in range(1, args.k_order + 1, args.k_order_interval):
                param_grid.append((graph_func, graph_name, k, mode, c, args.total_trials))

# Run in parallel
with multiprocessing.Pool() as pool:
    results_parallel = list(tqdm(pool.imap(simulate_parameter_combo, param_grid), total=len(param_grid)))

res_out_dir = Path(args.output_dir) / f"graph_gpt_v2_N-{args.graph_N}_k-{args.k_order}_{args.k_order_interval}_trials-{args.total_trials}_stable-{args.stable_criterion}_gen-{args.generations}_stable_trials-{args.stable_trial_threshold}"
res_out_dir.mkdir(parents=True, exist_ok=True)

# Write down hyperparameters as json in args
with open(res_out_dir / "hyperparameters.json", "w") as f:
    json.dump(vars(args), f, indent=4)


df_parallel = pd.DataFrame([r for r in results_parallel if r is not None])
df_parallel.to_csv(res_out_dir / "results.csv", index=False)

# Plot results
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_parallel, x="env_period", y="optimal_rate", hue="learning_mode", style="graph_type", marker="o")
plt.title("Optimal Innovation Rate vs Environmental Stability (Graph-Based Learning Modes)")
plt.xlabel("Environmental Stability (period between changes)")
plt.ylabel("Optimal Innovation Rate")
plt.grid(True)
plt.tight_layout()

plt.savefig(res_out_dir / "optimal_innovation_rate.png")