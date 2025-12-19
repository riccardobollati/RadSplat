import torch

def normalize_density_to_opacity(
    densities: torch.Tensor, 
    target_min: float = 0.1, 
    target_max: float = 0.6
) -> torch.Tensor:
    """
    Linearly scales input densities so the lowest value becomes target_min
    and the highest value becomes target_max.
    
    Args:
        densities (torch.Tensor): Raw density values.
        target_min (float): The lower bound for output opacity (0.1).
        target_max (float): The upper bound for output opacity (0.6).
        
    Returns:
        torch.Tensor: Normalized opacities with shape (N, 1).
    """
    # 1. Find the min and max of the current data
    current_min = densities.min()
    current_max = densities.max()

    # 2. Handle edge case: if all densities are identical (div by zero risk)
    if current_min == current_max:
        # If all points are the same, return the target_min for all of them
        return torch.full((densities.numel(), 1), target_min, device=densities.device)

    # 3. Normalize to [0, 1]
    # (x - min) / (max - min)
    normalized = (densities - current_min) / (current_max - current_min)

    # 4. Scale to [target_min, target_max]
    # scaled * (range) + start
    opacities = normalized * (target_max - target_min) + target_min

    # 5. Force shape to (N, 1)
    return opacities
