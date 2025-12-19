import torch

class Initializer:

    def __init__(self, weights, ray_samples) -> None:
        
        self.weights = weights
        self.ray_samples = ray_samples

    def compute_transmittance(self):

        cum_inc = torch.cumsum(self.weights, dim=1)
        cum_exc = cum_inc - self.weights
        T_per_sample = 1.0 - cum_exc

        return T_per_sample

    def compute_inital_positions(self, T_per_sample, treshold):

        positions = self.ray_samples.frustums.get_positions()
        T = T_per_sample.squeeze(-1)
        mask = (T > treshold)
        count = mask.sum(dim=1)
        last_idx = (count - 1).clamp(min=0)

        depths = get_points_depth(self.ray_samples, last_idx)

        initial_gaussians_position = positions[torch.arange(len(positions), device=positions.device), last_idx, :]

        return initial_gaussians_position, depths

def get_points_depth(ray_samples, indexes):
    starts = ray_samples.frustums.starts
    ends   = ray_samples.frustums.ends

    R = starts.shape[0]
    batch_ids = torch.arange(R, device=indexes.device)

    start = starts[batch_ids, indexes, 0]
    end   = ends[batch_ids, indexes, 0]
    depth = 0.5 * (start + end)

    return depth

def create_ray_bins_for_sampling(depths: torch.Tensor, n: int, delta=1.0e-3):
    """
    Generate multiple sample coordinates within centers ± delta

    Params:
        @depths:    -> center of the sampling bins
        @n:         -> number of points to sample
        @delta:     -> max sample ± deviation from sample
    """

    device = depths.device

    sample_depths = depths.unsqueeze(-1)
    for _ in range(n):
        deltas = delta * torch.randn(len(sample_depths), 1, device=device)
        sample_depths = torch.cat([sample_depths, sample_depths + deltas], dim=-1)

    return sample_depths
