import importlib
import torch

try:
    # Try importing directly first (if installed as package)
    ext_module = importlib.import_module('geotransformer.ext')
except ImportError:
    try:
        # Try importing from local path if in development
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../extensions'))
        # This might fail if .so/.pyd is not built
        ext_module = None 
    except ImportError:
        ext_module = None


def grid_subsample(points, lengths, voxel_size):
    """Grid subsampling in stack mode.

    This function is implemented on CPU.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    if ext_module is not None and hasattr(ext_module, 'grid_subsampling'):
        s_points, s_lengths = ext_module.grid_subsampling(points, lengths, voxel_size)
        return s_points, s_lengths
    
    # Pure Python/PyTorch implementation
    s_points_list = []
    s_lengths_list = []
    
    start = 0
    # Ensure lengths is CPU for iteration
    lengths_cpu = lengths.cpu().tolist()
    
    for length in lengths_cpu:
        length = int(length)
        end = start + length
        batch_points = points[start:end]
        
        if length == 0:
            s_lengths_list.append(torch.tensor(0, device=points.device))
            start = end
            continue

        # Voxel grid subsampling using inverse_unique
        # Quantize points
        quantized_points = torch.floor(batch_points / voxel_size).long() # use long for indices
        
        # Unique points in quantized space
        # dim=0 is crucial for coordinate uniqueness
        unique_voxels, unique_indices = torch.unique(quantized_points, dim=0, return_inverse=True)
        
        # Compute centroids
        num_voxels = unique_voxels.shape[0]
        
        # Prepare output for this batch
        voxel_sum = torch.zeros((num_voxels, 3), device=points.device, dtype=points.dtype)
        voxel_count = torch.zeros((num_voxels, 1), device=points.device, dtype=points.dtype)
        
        # Accumulate
        voxel_sum.index_add_(0, unique_indices, batch_points)
        voxel_count.index_add_(0, unique_indices, torch.ones((batch_points.shape[0], 1), device=points.device))
        
        # Average
        subsampled_points = voxel_sum / voxel_count
        
        s_points_list.append(subsampled_points)
        s_lengths_list.append(torch.tensor(subsampled_points.shape[0], device=points.device))
        
        start = end
        
    if len(s_points_list) > 0:
        return torch.cat(s_points_list, dim=0), torch.stack(s_lengths_list)
    else:
        return torch.empty((0, 3), device=points.device), torch.stack(s_lengths_list)

