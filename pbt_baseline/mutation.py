import copy
import random
import math


def mutate_float(x, change_min=1.1, change_max=1.5):
    """Mutate a float value by a random factor."""
    perturb_amount = random.uniform(change_min, change_max)
    
    # mutation direction
    new_value = x / perturb_amount if random.random() < 0.5 else x * perturb_amount
    return new_value


def mutate_float_min_1(x, **kwargs):
    """Mutate float with minimum value of 1.0."""
    new_value = mutate_float(x, **kwargs)
    new_value = max(1.0, new_value)
    return new_value


def mutate_learning_rate(x, **kwargs):
    """Special mutation for learning rates."""
    new_value = mutate_float(x, **kwargs)
    new_value = max(1e-6, new_value)  # Prevent too small learning rates
    new_value = min(1e-2, new_value)  # Prevent too large learning rates
    return new_value


def mutate_batch_size(x, **kwargs):
    """Mutate batch size (keep as power of 2)."""
    # Convert to log2, mutate, then convert back
    log_val = random.uniform(-0.5, 0.5)
    new_value = x * (2 ** log_val)
    new_value = max(16, int(new_value))  # Minimum batch size
    new_value = min(32768, new_value)    # Maximum batch size
    # Round to nearest power of 2
    new_value = 2 ** round(math.log2(new_value))
    return new_value


def mutate_discount(x, **kwargs):
    """Special mutation func for parameters such as gamma (discount factor)."""
    inv_x = 1.0 - x
    # very conservative, large changes in gamma can lead to very different value estimates
    new_inv_x = mutate_float(inv_x, change_min=1.1, change_max=1.2)
    new_value = 1.0 - new_inv_x
    new_value = max(0.9, new_value)  # Minimum discount factor
    new_value = min(0.999, new_value)  # Maximum discount factor
    return new_value


def mutate_entropy_coeff(x, **kwargs):
    """Mutate entropy coefficient for PPO."""
    new_value = mutate_float(x, **kwargs)
    new_value = max(1e-6, new_value)
    new_value = min(0.1, new_value)
    return new_value


def mutate_clip_coeff(x, **kwargs):
    """Mutate PPO clip coefficient."""
    new_value = mutate_float(x, change_min=1.05, change_max=1.2)
    new_value = max(0.05, new_value)
    new_value = min(0.5, new_value)
    return new_value


def mutate_gae_lambda(x, **kwargs):
    """Mutate GAE lambda parameter."""
    new_value = mutate_float(x, change_min=1.1, change_max=1.3)
    new_value = max(0.8, new_value)
    new_value = min(0.99, new_value)
    return new_value


def mutate_mini_epochs(x, **kwargs):
    """Mutate number of mini epochs."""
    change_amount = 1
    new_value = x + change_amount if random.random() < 0.5 else x - change_amount
    new_value = max(1, new_value)
    new_value = min(10, new_value)
    return new_value


def mutate_hidden_dim(x, **kwargs):
    """Mutate hidden dimension (keep as power of 2)."""
    log_val = random.uniform(-0.3, 0.3)
    new_value = x * (2 ** log_val)
    new_value = max(64, int(new_value))
    new_value = min(2048, new_value)
    # Round to nearest power of 2
    new_value = 2 ** round(math.log2(new_value))
    return new_value


def get_mutation_func(mutation_func_name):
    """Get mutation function by name."""
    try:
        func = eval(mutation_func_name)
    except Exception as exc:
        print(f'Exception {exc} while trying to find the mutation func {mutation_func_name}.')
        raise Exception(f'Could not find mutation func {mutation_func_name}')
    
    return func


def mutate(params, mutations, mutation_rate, pbt_change_min, pbt_change_max):
    """
    Mutate parameters based on mutation configuration.
    
    Args:
        params: Dictionary of current parameter values
        mutations: Dictionary mapping parameter names to mutation function names
        mutation_rate: Probability of mutating each parameter
        pbt_change_min: Minimum change factor for mutations
        pbt_change_max: Maximum change factor for mutations
    
    Returns:
        Dictionary of mutated parameters
    """
    mutated_params = copy.deepcopy(params)
    
    for param, param_value in params.items():
        # toss a coin whether we perturb the parameter at all
        if random.random() > mutation_rate:
            continue
            
        if param not in mutations:
            print(f'Parameter {param} not in mutations config, skipping')
            continue
            
        mutation_func_name = mutations[param]
        mutation_func = get_mutation_func(mutation_func_name)
        
        mutated_value = mutation_func(param_value, change_min=pbt_change_min, change_max=pbt_change_max)
        mutated_params[param] = mutated_value
        
        print(f'Param {param} mutated from {param_value} to {mutated_value}')
    
    return mutated_params