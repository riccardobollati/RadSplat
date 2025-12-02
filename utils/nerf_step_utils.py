import torch

def generate_sample_coordinates(centers : torch.Tensor, n: int,  delta = 1.e-2):
    """
    Generate multiple sample coordinates within centers ± delta

    Params:
        @centers: tensor
    """



