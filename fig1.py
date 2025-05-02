import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import time
import pandas as pd
import os
import math

# Model state representing population frequencies
class ModelState:
    def __init__(self, x1, x2, x3, x4):
        # x1: frequency of RM
        # x2: frequency of Rm
        # x3: frequency of rM
        # x4: frequency of rm
        self.x = np.array([x1, x2, x3, x4])
    
    def normalize(self):
        self.x = self.x / np.sum(self.x)
        
    def clone(self):
        return ModelState(self.x[0], self.x[1], self.x[2], self.x[3])

# Model parameters
class ModelParameters:
    def __init__(self, 
                 innovation_rate_M,       # μM
                 innovation_rate_m,       # μm
                 environmental_period,    # c
                 oblique_learning_prob,   # P_O
                 learning_mode,           # "random", "success", or "conformist"
                 n_tutors,                # n - number of tutors sampled 
                 s1, s2):                 # Selection coefficients
        self.innovation_rate_M = innovation_rate_M
        self.innovation_rate_m = innovation_rate_m
        self.environmental_period = environmental_period
        self.oblique_learning_prob = oblique_learning_prob
        self.learning_mode = learning_mode
        self.n_tutors = n_tutors
        self.s1 = s1
        self.s2 = s2

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

def random_oblique_learning(state_after_vertical, original_state, p_o):
    """Implement random oblique transmission"""
    x1_v, x2_v, x3_v, x4_v = state_after_vertical.x
    x1, x2, x3, x4 = original_state.x
    
    # Repertoire frequencies
    R_freq = x1 + x2
    r_freq = x3 + x4
    
    # Calculate frequencies after oblique learning
    new_x1 = (1 - p_o) * x1_v + p_o * (x1_v + x3_v) * R_freq
    new_x2 = (1 - p_o) * x2_v + p_o * (x2_v + x4_v) * R_freq
    new_x3 = (1 - p_o) * x3_v + p_o * (x1_v + x3_v) * r_freq
    new_x4 = (1 - p_o) * x4_v + p_o * (x2_v + x4_v) * r_freq
    
    new_state = ModelState(new_x1, new_x2, new_x3, new_x4)
    new_state.normalize()
    return new_state

def success_biased_learning(state_after_vertical, original_state, p_o, n, environment):
    """Implement success-biased transmission"""
    x1_v, x2_v, x3_v, x4_v = state_after_vertical.x
    x1, x2, x3, x4 = original_state.x
    
    # Repertoire frequencies
    R_freq = x1 + x2
    r_freq = x3 + x4
    
    # f(m > 1:n, freq) - probability of at least one individual having the trait
    def prob_at_least_one(freq):
        return 1 - (1 - freq)**n
    
    # f(m = n:n, freq) - probability of all individuals having the trait
    def prob_all(freq):
        return freq**n
    
    if environment == 0:  # Environment 1 favors r
        # Calculate frequencies after success-biased learning
        new_x1 = (1 - p_o) * x1_v + p_o * (x1_v + x3_v) * prob_all(R_freq)
        new_x2 = (1 - p_o) * x2_v + p_o * (x2_v + x4_v) * prob_all(R_freq)
        new_x3 = (1 - p_o) * x3_v + p_o * (x1_v + x3_v) * prob_at_least_one(r_freq)
        new_x4 = (1 - p_o) * x4_v + p_o * (x2_v + x4_v) * prob_at_least_one(r_freq)
    else:  # Environment 2 favors R
        # Calculate frequencies after success-biased learning
        new_x1 = (1 - p_o) * x1_v + p_o * (x1_v + x3_v) * prob_at_least_one(R_freq)
        new_x2 = (1 - p_o) * x2_v + p_o * (x2_v + x4_v) * prob_at_least_one(R_freq)
        new_x3 = (1 - p_o) * x3_v + p_o * (x1_v + x3_v) * prob_all(r_freq)
        new_x4 = (1 - p_o) * x4_v + p_o * (x2_v + x4_v) * prob_all(r_freq)
    
    new_state = ModelState(new_x1, new_x2, new_x3, new_x4)
    new_state.normalize()
    return new_state

def conformist_learning(state_after_vertical, original_state, p_o, n):
    """Implement conformist transmission according to paper's equations"""
    x1_v, x2_v, x3_v, x4_v = state_after_vertical.x
    x1, x2, x3, x4 = original_state.x
    
    # Repertoire frequencies
    R_freq = x1 + x2
    r_freq = x3 + x4
    
    # Calculate probability that more than half of sampled individuals have trait R
    # f(m > n/2:n, (x_1 + x_2))
    prob_R_majority = 0
    for m in range(n//2 + 1, n + 1):
        prob_R_majority += math.comb(n, m) * (R_freq**m) * (r_freq**(n-m))
    
    # Calculate probability that more than half of sampled individuals have trait r
    # f(m > n/2:n, (x_3 + x_4))
    prob_r_majority = 0
    for m in range(n//2 + 1, n + 1):
        prob_r_majority += math.comb(n, m) * (r_freq**m) * (R_freq**(n-m))
    
    # Calculate frequencies after conformist learning - directly from paper equations
    new_x1 = (1 - p_o) * x1_v + p_o * (x1_v + x3_v) * prob_R_majority
    new_x2 = (1 - p_o) * x2_v + p_o * (x2_v + x4_v) * prob_R_majority
    new_x3 = (1 - p_o) * x3_v + p_o * (x1_v + x3_v) * prob_r_majority
    new_x4 = (1 - p_o) * x4_v + p_o * (x2_v + x4_v) * prob_r_majority
    
    new_state = ModelState(new_x1, new_x2, new_x3, new_x4)
    new_state.normalize()
    return new_state

def innovation(state, params):
    """Implement innovation according to the model"""
    x1_o, x2_o, x3_o, x4_o = state.x
    
    # Innovation rates for M and m modifiers
    mu_M = params.innovation_rate_M
    mu_m = params.innovation_rate_m
    
    # Calculate frequencies after innovation
    new_x1 = (1 - mu_M) * x1_o + mu_M * x3_o
    new_x2 = (1 - mu_m) * x2_o + mu_m * x4_o
    new_x3 = (1 - mu_M) * x3_o + mu_M * x1_o
    new_x4 = (1 - mu_m) * x4_o + mu_m * x2_o
    
    new_state = ModelState(new_x1, new_x2, new_x3, new_x4)
    new_state.normalize()
    return new_state

def selection(state, environment, params):
    """Implement selection according to the environment"""
    x1_i, x2_i, x3_i, x4_i = state.x
    
    if environment == 0:  # Environment 1 favors r
        # Calculate mean fitness
        w_bar = x1_i + x2_i + (x3_i + x4_i) * (1 + params.s1)
        
        # Calculate new frequencies
        new_x1 = x1_i / w_bar
        new_x2 = x2_i / w_bar
        new_x3 = x3_i * (1 + params.s1) / w_bar
        new_x4 = x4_i * (1 + params.s1) / w_bar
    else:  # Environment 2 favors R
        # Calculate mean fitness
        w_bar = (x1_i + x2_i) * (1 + params.s2) + x3_i + x4_i
        
        # Calculate new frequencies
        new_x1 = x1_i * (1 + params.s2) / w_bar
        new_x2 = x2_i * (1 + params.s2) / w_bar
        new_x3 = x3_i / w_bar
        new_x4 = x4_i / w_bar
    
    new_state = ModelState(new_x1, new_x2, new_x3, new_x4)
    new_state.normalize()
    return new_state

def update_generation(state, environment, params):
    """Update the model for one generation"""
    # 1. Vertical learning (combined with mating)
    state_after_vertical = vertical_learning(state)
    
    # 2. Oblique learning based on the specified mode
    if params.oblique_learning_prob > 0:
        if params.learning_mode == "random":
            state_after_learning = random_oblique_learning(
                state_after_vertical, state, params.oblique_learning_prob)
        elif params.learning_mode == "success":
            state_after_learning = success_biased_learning(
                state_after_vertical, state, params.oblique_learning_prob, 
                params.n_tutors, environment)
        elif params.learning_mode == "conformist":
            state_after_learning = conformist_learning(
                state_after_vertical, state, params.oblique_learning_prob, 
                params.n_tutors)
        else:
            # Default to no oblique learning if mode is unrecognized
            state_after_learning = state_after_vertical
    else:
        # No oblique learning
        state_after_learning = state_after_vertical
    
    # 3. Innovation
    state_after_innovation = innovation(state_after_learning, params)
    
    # 4. Selection
    state_after_selection = selection(state_after_innovation, environment, params)
    
    return state_after_selection

def find_evolutionary_stable_rate(environmental_period, oblique_learning_prob, 
                                 learning_mode, n_tutors, s1, s2, trial_index):
    """
    Find the evolutionarily stable rate of innovation following the paper's method.
    """
    # Set a consistent random seed for each trial
    np.random.seed(trial_index * 1000 + int(environmental_period))
    
    # Parameters based on the paper
    max_non_invasion_trials = 500
    invasion_threshold = 0.05
    num_generations = 1000  # Run long enough to reach equilibrium
    
    # Start with a random innovation rate between 0 and 1
    resident_rate = np.random.uniform(0, 1)
    
    non_invasion_count = 0
    total_trials = 0
    
    while non_invasion_count < max_non_invasion_trials and total_trials < 2000:
        # Generate mutant rate as described in the paper - product of resident rate 
        # and a random number from an exponential distribution with mean 1
        mutant_rate = resident_rate * np.random.exponential(1.0)
        
        # Ensure mutant_rate is between 0 and 1
        mutant_rate = min(0.9999, max(0.0001, mutant_rate))
        
        # Set up parameters
        params = ModelParameters(
            innovation_rate_M=resident_rate,
            innovation_rate_m=mutant_rate,
            environmental_period=environmental_period,
            oblique_learning_prob=oblique_learning_prob,
            learning_mode=learning_mode,
            n_tutors=n_tutors,
            s1=s1, s2=s2
        )
        
        # Initialize with population fixed on M modifier trait with low m frequency
        initial_m_freq = 0.001  # Very low initial frequency for mutant
        initial_state = ModelState(0.5 - initial_m_freq/2, initial_m_freq/2, 
                                  0.5 - initial_m_freq/2, initial_m_freq/2)
        
        state = initial_state.clone()
        environment = 0  # Start with environment 0
        
        # Run the simulation for enough generations to reach equilibrium
        for generation in range(num_generations):
            # Update the environment according to periodicity
            if generation % environmental_period == 0:
                environment = 1 - environment
            
            # Perform one generation of the model
            state = update_generation(state, environment, params)
        
        # Check if mutant has invaded - frequency should exceed threshold
        m_allele_freq = state.x[1] + state.x[3]
        
        if m_allele_freq > invasion_threshold:
            # Mutant has invaded - update resident rate
            resident_rate = mutant_rate
            non_invasion_count = 0
        else:
            # Mutant did not invade - count towards stability
            non_invasion_count += 1
        
        total_trials += 1
    
    return resident_rate

def run_simulation_for_params(args):
    """Wrapper for multiprocessing"""
    (environmental_period, oblique_learning_prob, learning_mode, n_tutors, s1, s2, trial_index) = args
    
    try:
        optimal_rate = find_evolutionary_stable_rate(
            environmental_period=environmental_period,
            oblique_learning_prob=oblique_learning_prob,
            learning_mode=learning_mode,
            n_tutors=n_tutors,
            s1=s1, s2=s2, trial_index=trial_index
        )
        
        return (environmental_period, oblique_learning_prob, learning_mode, n_tutors, optimal_rate, trial_index)
    except Exception as e:
        print(f"Error in simulation: {e}")
        return (environmental_period, oblique_learning_prob, learning_mode, n_tutors, None, trial_index)

def replicate_figure_1():
    """Replicate Figure 1 from the paper and save raw results to CSV"""
    # Parameters from the paper
    s1 = s2 = 1.0  # Equal selection coefficients as mentioned
    
    # Environmental periods to test
    environmental_periods = list(range(5, 41, 1))
    
    # Different learning scenarios to test
    scenarios = [
        {"label": "Vertical only", "p_o": 0, "mode": "random", "n": 5, "color": "k"},
        {"label": "Random oblique (p=0.1)", "p_o": 0.1, "mode": "random", "n": 5, "color": "r"},
        {"label": "Success-biased (p=0.1, n=5)", "p_o": 0.1, "mode": "success", "n": 5, "color": "b"},
        {"label": "Success-biased (p=0.1, n=50)", "p_o": 0.1, "mode": "success", "n": 50, "color": "orange"},
        {"label": "Success-biased (p=0.4, n=5)", "p_o": 0.4, "mode": "success", "n": 5, "color": "y"},
        {"label": "Conformist (p=0.1, n=5)", "p_o": 0.1, "mode": "conformist", "n": 5, "color": "b"},
        {"label": "Conformist (p=0.1, n=50)", "p_o": 0.1, "mode": "conformist", "n": 50, "color": "orange"},
        {"label": "Conformist (p=0.4, n=5)", "p_o": 0.4, "mode": "conformist", "n": 5, "color": "y"},
    ]
    
    # Number of repeats for each parameter combination
    num_repeats = 5
    
    # Prepare parameter combinations
    param_combinations = []
    for environmental_period in environmental_periods:
        for scenario in scenarios:
            for trial in range(num_repeats):
                param_combinations.append(
                    (environmental_period, scenario["p_o"], scenario["mode"], 
                     scenario["n"], s1, s2, trial)
                )
    
    # Run simulations in parallel
    print(f"Running {len(param_combinations)} simulations in parallel...")
    start_time = time.time()
    
    with Pool() as pool:
        results = list(tqdm(
            pool.imap(run_simulation_for_params, param_combinations),
            total=len(param_combinations)
        ))
    
    end_time = time.time()
    print(f"Simulations completed in {end_time - start_time:.2f} seconds.")
    
    # Convert results to DataFrame for CSV export
    results_data = []
    for environmental_period, p_o, mode, n, optimal_rate, trial in results:
        if optimal_rate is not None:  # Filter out any failed runs
            results_data.append({
                'environmental_period': environmental_period,
                'oblique_learning_prob': p_o,
                'learning_mode': mode,
                'n_tutors': n,
                'optimal_rate': optimal_rate,
                'trial': trial
            })
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results_data)
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save raw results to CSV
    results_df.to_csv('results/figure_1_raw_data.csv', index=False)
    print(f"Raw results saved to 'results/figure_1_raw_data.csv'")
    
    # Process results for plotting
    # Group by scenario and environmental period
    plot_data = {}
    for environmental_period, p_o, mode, n, optimal_rate, trial in results:
        if optimal_rate is not None:  # Filter out any failed runs
            key = (p_o, mode, n)
            if key not in plot_data:
                plot_data[key] = {}
            
            if environmental_period not in plot_data[key]:
                plot_data[key][environmental_period] = []
            
            plot_data[key][environmental_period].append(optimal_rate)
    
    # Calculate summary statistics for CSV export
    summary_data = []
    for (p_o, mode, n), periods in plot_data.items():
        for period, rates in periods.items():
            mean_rate = np.mean(rates)
            std_error = np.std(rates) / np.sqrt(len(rates))
            scenario_label = next((s["label"] for s in scenarios if s["p_o"] == p_o and s["mode"] == mode and s["n"] == n), "Unknown")
            
            summary_data.append({
                'environmental_period': period,
                'oblique_learning_prob': p_o,
                'learning_mode': mode,
                'n_tutors': n,
                'mean_optimal_rate': mean_rate,
                'std_error': std_error,
                'scenario_label': scenario_label
            })
    
    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/figure_1_summary_data.csv', index=False)
    print(f"Summary results saved to 'results/figure_1_summary_data.csv'")
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Figure 1a (non-conformist learning)
    non_conformist_scenarios = [s for s in scenarios if s["mode"] != "conformist"]
    for scenario in non_conformist_scenarios:
        key = (scenario["p_o"], scenario["mode"], scenario["n"])
        if key in plot_data:
            env_periods = []
            mean_rates = []
            std_errors = []
            
            for period in sorted(plot_data[key].keys()):
                rates = plot_data[key][period]
                env_periods.append(period)
                mean_rates.append(np.mean(rates))
                std_errors.append(np.std(rates) / np.sqrt(len(rates)))
            
            ax1.errorbar(env_periods, mean_rates, yerr=std_errors, 
                        fmt=f'-o', color=scenario["color"], label=scenario["label"])
    
    # Plot Figure 1b (conformist learning)
    conformist_scenarios = [s for s in scenarios if s["mode"] == "conformist"]
    for scenario in conformist_scenarios:
        key = (scenario["p_o"], scenario["mode"], scenario["n"])
        if key in plot_data:
            env_periods = []
            mean_rates = []
            std_errors = []
            
            for period in sorted(plot_data[key].keys()):
                rates = plot_data[key][period]
                env_periods.append(period)
                mean_rates.append(np.mean(rates))
                std_errors.append(np.std(rates) / np.sqrt(len(rates)))
            
            ax2.errorbar(env_periods, mean_rates, yerr=std_errors, 
                        fmt=f'-o', color=scenario["color"], label=scenario["label"])
    
    # Add vertical learning to both panels for comparison
    key = (0, "random", 5)  # Vertical only
    if key in plot_data:
        env_periods = []
        mean_rates = []
        std_errors = []
        
        for period in sorted(plot_data[key].keys()):
            rates = plot_data[key][period]
            env_periods.append(period)
            mean_rates.append(np.mean(rates))
            std_errors.append(np.std(rates) / np.sqrt(len(rates)))
        
        ax2.errorbar(env_periods, mean_rates, yerr=std_errors, 
                    fmt=f'-o', color="k", label="Vertical only")
    
    # Set labels and titles
    ax1.set_xlabel('Environmental stability (time steps between environmental changes)')
    ax1.set_ylabel('Optimal rate of innovation')
    ax1.set_title('(a) Non-conformist learning')
    ax1.legend()
    ax1.set_ylim(0, 0.2)
    ax1.set_yticks(np.arange(0, 0.22, 0.02))
    ax1.set_xlim(5, 40)
    
    ax2.set_xlabel('Environmental stability (time steps between environmental changes)')
    ax2.set_ylabel('Optimal rate of innovation')
    ax2.set_title('(b) Conformist learning')
    ax2.legend()
    ax2.set_ylim(0, 0.2)
    ax2.set_yticks(np.arange(0, 0.22, 0.02))
    ax2.set_xlim(5, 40)
    
    plt.tight_layout()
    plt.savefig('results/figure_1_replication.png', dpi=300)
    print(f"Figure saved to 'results/figure_1_replication.png'")
    plt.show()

if __name__ == "__main__":
    replicate_figure_1()