import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

# Model state representing population frequencies
class ModelState:
    def __init__(self, x1: float, x2: float, x3: float) -> None:
        # x1: frequency of R (environment 2 specialists)
        # x2: frequency of Rr (generalists with large repertoire)
        # x3: frequency of r (environment 1 specialists)
        self.x: np.ndarray = np.array([x1, x2, x3])
    
    def normalize(self) -> None:
        self.x = self.x / np.sum(self.x)

# Model parameters
class ModelParameters:
    def __init__(self, 
                 innovation_rate: float,         # μ
                 environmental_period: int,      # c
                 oblique_learning_prob: float,   # P_O
                 s1: float, s2: float,          # Selection coefficients
                 delta_c: float,                # Cost of maintaining two repertoires
                 num_generations: int) -> None: # Total generations to simulate
        self.innovation_rate: float = innovation_rate
        self.environmental_period: int = environmental_period
        self.oblique_learning_prob: float = oblique_learning_prob
        self.s1: float = s1
        self.s2: float = s2
        self.delta_c: float = delta_c
        self.num_generations: int = num_generations

def mating(state: ModelState) -> ModelState:
    """Implement random mating according to Table 3"""
    x1, x2, x3 = state.x
    
    # Calculate offspring frequencies after mating
    new_x1 = x1**2 + x1*x2 + x2**2/4
    new_x2 = x2**2/2 + 2*x1*x3 + x1*x2 + x2*x3
    new_x3 = x3**2 + x2*x3 + x2**2/4
    
    new_state = ModelState(new_x1, new_x2, new_x3)
    new_state.normalize()
    return new_state

def oblique_learning(state_after_vertical: ModelState, original_state: ModelState, p_o: float) -> ModelState:
    """Implement random oblique learning"""
    if p_o == 0:
        # No oblique learning, return state after vertical learning
        return state_after_vertical
    
    x1_v, x2_v, x3_v = state_after_vertical.x
    x1, x2, x3 = original_state.x
    
    # Calculate frequencies after oblique learning
    new_x1 = (1 - p_o) * x1_v + p_o * (x1_v + x2_v) * (x1 + x2/2)
    new_x2 = (1 - p_o) * x2_v + p_o * (x1_v * (x3 + x2/2) + x3_v * (x1 + x2/2) + x2_v * (x1 + x2/2))
    new_x3 = (1 - p_o) * x3_v + p_o * (x3_v + x2_v) * (x3 + x2/2)
    
    new_state = ModelState(new_x1, new_x2, new_x3)
    new_state.normalize()
    return new_state

def innovation(state: ModelState, innovation_rate: float) -> ModelState:
    """Implement innovation according to the model"""
    x1_o, x2_o, x3_o = state.x
    
    # Calculate frequencies after innovation
    new_x1 = (1 - innovation_rate) * x1_o + innovation_rate * (x2_o/2)
    new_x2 = (1 - innovation_rate) * x2_o + innovation_rate * (x1_o + x3_o)
    new_x3 = (1 - innovation_rate) * x3_o + innovation_rate * (x2_o/2)
    
    new_state = ModelState(new_x1, new_x2, new_x3)
    new_state.normalize()
    return new_state

def selection(state: ModelState, environment: int, params: ModelParameters) -> ModelState:
    """Implement selection according to the environment"""
    x1_i, x2_i, x3_i = state.x
    
    if environment == 0:  # Environment 1 favors r
        # Calculate mean fitness
        w_bar = x1_i + x2_i * (1 - params.delta_c) * (1 + params.s1) + x3_i * (1 + params.s1)
        
        # Calculate new frequencies
        new_x1 = x1_i / w_bar
        new_x2 = x2_i * (1 - params.delta_c) * (1 + params.s1) / w_bar
        new_x3 = x3_i * (1 + params.s1) / w_bar
    else:  # Environment 2 favors R
        # Calculate mean fitness
        w_bar = x1_i * (1 + params.s2) + x2_i * (1 - params.delta_c) * (1 + params.s2) + x3_i
        
        # Calculate new frequencies
        new_x1 = x1_i * (1 + params.s2) / w_bar
        new_x2 = x2_i * (1 - params.delta_c) * (1 + params.s2) / w_bar
        new_x3 = x3_i / w_bar
    
    new_state = ModelState(new_x1, new_x2, new_x3)
    new_state.normalize()
    return new_state

def update_generation(state: ModelState, environment: int, params: ModelParameters) -> ModelState:
    """Update the model for one generation"""
    # 1. Mating (includes vertical learning)
    state_after_mating = mating(state)
    
    # 2. Oblique learning
    state_after_learning = oblique_learning(state_after_mating, state, params.oblique_learning_prob)
    
    # 3. Innovation
    state_after_innovation = innovation(state_after_learning, params.innovation_rate)
    
    # 4. Selection
    state_after_selection = selection(state_after_innovation, environment, params)
    
    return state_after_selection

def run_simulation(params: ModelParameters, initial_state: ModelState | None = None) -> float:
    """Run simulation and return the frequency of the generalist repertoire"""
    if initial_state is None:
        # Initialize with equal frequencies
        state = ModelState(1/3, 1/3, 1/3)
    else:
        state = initial_state
    
    # Current environment (0 for E1, 1 for E2)
    environment = 0
    
    # Run the simulation for the specified number of generations
    for generation in range(params.num_generations):
        # Update the environment
        if generation % params.environmental_period == 0:
            environment = 1 - environment  # Toggle between 0 and 1
        
        # Perform one generation of the model
        state = update_generation(state, environment, params)
    
    # Return the frequency of the generalist repertoire
    return state.x[1]  # x2 is the frequency of Rr

def run_simulation_for_params(args: tuple[float, int, float, bool, float, float, float, int]) -> tuple[float, int, float, bool, float]:
    """Wrapper for multiprocessing"""
    (innovation_rate, environmental_period, oblique_learning_prob, 
     is_vertical_only, s1, s2, delta_c, num_generations) = args
    
    # If vertical-only transmission, set P_O to 0
    effective_p_o = 0 if is_vertical_only else oblique_learning_prob
    
    params = ModelParameters(
        innovation_rate=innovation_rate,
        environmental_period=environmental_period,
        oblique_learning_prob=effective_p_o,
        s1=s1, s2=s2,
        delta_c=delta_c,
        num_generations=num_generations
    )
    
    generalist_frequency = run_simulation(params)
    
    return (innovation_rate, environmental_period, oblique_learning_prob, is_vertical_only, generalist_frequency)

def replicate_figure_2() -> None:
    """Replicate Figure 2 from the paper"""
    # Parameters from the paper
    s1: float = 1.0
    s2: float = 1.0
    delta_c: float = 0.01
    num_generations: int = 15000
    
    # Innovation rates to test
    innovation_rates: np.ndarray = np.linspace(0.01, 0.5, 20)
    
    # Environmental periods
    environmental_periods: list[int] = [1, 50]  # c values from paper
    
    # Oblique learning probabilities
    oblique_learning_probs: list[float] = [0.1, 0.6]  # For Figure 2a and 2b
    
    # Transmission types
    is_vertical_only_options: list[bool] = [True, False]  # For dashed and solid lines
    
    # Prepare parameter combinations
    param_combinations: list[tuple[float, int, float, bool, float, float, float, int]] = [
        (innovation_rate, environmental_period, oblique_learning_prob, is_vertical_only, s1, s2, delta_c, num_generations)
        for innovation_rate in innovation_rates
        for environmental_period in environmental_periods
        for oblique_learning_prob in oblique_learning_probs
        for is_vertical_only in is_vertical_only_options
    ]
    
    # Run simulations in parallel
    print(f"Running {len(param_combinations)} simulations in parallel...")
    start_time = time.time()
    
    with Pool() as pool:
        results: list[tuple[float, int, float, bool, float]] = pool.map(run_simulation_for_params, param_combinations)
    
    end_time = time.time()
    print(f"Simulations completed in {end_time - start_time:.2f} seconds.")
    
    # Process results for plotting
    plot_data: dict[tuple[float, int, bool], list[tuple[float, float]]] = {}
    for innovation_rate, environmental_period, oblique_learning_prob, is_vertical_only, generalist_frequency in results:
        key = (oblique_learning_prob, environmental_period, is_vertical_only)
        if key not in plot_data:
            plot_data[key] = []
        plot_data[key].append((innovation_rate, generalist_frequency))
    
    # Sort data points by innovation rate
    for key in plot_data:
        plot_data[key].sort(key=lambda x: x[0])
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Figure 2a (P_O = 0.1)
    for environmental_period in environmental_periods:
        for is_vertical_only in is_vertical_only_options:
            key = (0.1, environmental_period, is_vertical_only)
            if key in plot_data:
                x_values, y_values = zip(*plot_data[key])
                
                color = 'k' if environmental_period == 1 else 'r'
                linestyle = '--' if is_vertical_only else '-'
                
                label = f'c = {environmental_period}, {"vertical" if is_vertical_only else "random"}'
                ax1.plot(x_values, y_values, color + linestyle, label=label)
    
    # Plot Figure 2b (P_O = 0.6)
    for environmental_period in environmental_periods:
        for is_vertical_only in is_vertical_only_options:
            key = (0.6, environmental_period, is_vertical_only)
            if key in plot_data:
                x_values, y_values = zip(*plot_data[key])
                
                color = 'k' if environmental_period == 1 else 'r'
                linestyle = '--' if is_vertical_only else '-'
                
                label = f'c = {environmental_period}, {"vertical" if is_vertical_only else "random"}'
                ax2.plot(x_values, y_values, color + linestyle, label=label)
    
    # Set labels and titles
    ax1.set_xlabel('Rate of innovation, μ')
    ax1.set_ylabel('Frequency of large repertoire')
    ax1.set_title('(a) P₀ = 0.1 (low reliance on social learning)')
    ax1.legend()
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(0, 0.7)
    
    ax2.set_xlabel('Rate of innovation, μ')
    ax2.set_ylabel('Frequency of large repertoire')
    ax2.set_title('(b) P₀ = 0.6 (high reliance on social learning)')
    ax2.legend()
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(0, 0.7)
    
    plt.tight_layout()
    plt.savefig('figure_2_replication.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    replicate_figure_2()