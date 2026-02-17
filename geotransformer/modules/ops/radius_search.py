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


def radius_search(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit):
    r"""Computes neighbors for a batch of q_points and s_points, apply radius search (in stack mode).

    This function is implemented on CPU.

    Args:
        q_points (Tensor): the query points (N, 3)
        s_points (Tensor): the support points (M, 3)
        q_lengths (Tensor): the list of lengths of batch elements in q_points
        s_lengths (Tensor): the list of lengths of batch elements in s_points
        radius (float): maximum distance of neighbors
        neighbor_limit (int): maximum number of neighbors

    Returns:
        neighbors (Tensor): the k nearest neighbors of q_points in s_points (N, k).
            Filled with M if there are less than k neighbors.
    """
    if ext_module is not None and hasattr(ext_module, 'radius_neighbors'):
        neighbor_indices = ext_module.radius_neighbors(q_points, s_points, q_lengths, s_lengths, radius)
        if neighbor_limit > 0:
            neighbor_indices = neighbor_indices[:, :neighbor_limit]
        return neighbor_indices
        
    # Pure Python/PyTorch implementation
    # This is much slower than C++ implementation but works without compilation
    
    neighbor_indices_list = []
    
    q_start = 0
    s_start = 0
    
    # Ensure lengths are CPU for iteration
    q_lengths_cpu = q_lengths.cpu().tolist()
    s_lengths_cpu = s_lengths.cpu().tolist()
    
    # M is the total number of support points. However, the C++ implementation likely uses
    # batch-relative indices or global indices.
    # Typically, output indices are indices into s_points (global indices).
    # But "Filled with M" suggests M is a sentinel value.
    # If s_points has size (Total_M, 3), then M = Total_M.
    
    total_M = s_points.shape[0]
    
    for i, (q_len, s_len) in enumerate(zip(q_lengths_cpu, s_lengths_cpu)):
        q_len = int(q_len)
        s_len = int(s_len)
        q_end = q_start + q_len
        s_end = s_start + s_len
        
        if q_len == 0:
            q_start = q_end
            s_start = s_end
            continue
            
        if s_len == 0:
            # No support points, fill with M
            neighbors = torch.full((q_len, neighbor_limit), total_M, dtype=torch.long, device=q_points.device)
            neighbor_indices_list.append(neighbors)
            q_start = q_end
            s_start = s_end
            continue
            
        batch_q = q_points[q_start:q_end] # (N_i, 3)
        batch_s = s_points[s_start:s_end] # (M_i, 3)
        
        # Compute pairwise distance
        # (N_i, M_i)
        dist = torch.cdist(batch_q, batch_s)
        
        # Find neighbors within radius
        # This is tricky because different points have different number of neighbors.
        # We need to pad to neighbor_limit.
        
        # Mask for valid neighbors
        mask = dist <= radius
        
        # Get indices
        # We want to prioritize closest neighbors? The C++ implementation usually does not guarantee order
        # unless specified. But usually radius search returns arbitrary neighbors within radius.
        # topk largest=False gives smallest distances
        
        # We can use topk to find closest k neighbors
        # Set distance of invalid neighbors to infinity
        dist_masked = dist.clone()
        dist_masked[~mask] = float('inf')
        
        # Get top k (smallest distance)
        # If neighbor_limit > M_i, we take all M_i
        k = min(neighbor_limit, s_len)
        
        vals, inds = torch.topk(dist_masked, k=k, dim=1, largest=False)
        
        # Check validity (inf means no neighbor found)
        valid_mask = vals != float('inf')
        
        # We need to map local indices (0 to s_len-1) to global indices (s_start to s_end-1)
        global_inds = inds + s_start
        
        # Prepare output tensor filled with M (sentinel)
        out_neighbors = torch.full((q_len, neighbor_limit), total_M, dtype=torch.long, device=q_points.device)
        
        # Fill valid neighbors
        # We only fill up to k columns
        # And only where valid_mask is true
        
        # Since we can't easily vectorize the filling of variable length valid neighbors if we use boolean mask,
        # but here we used topk so we have fixed shape (N_i, k).
        
        # We copy global_inds into out_neighbors[:, :k] where valid_mask is True
        # Actually topk returns values for all rows.
        # If a row has fewer than k neighbors, the rest will be inf.
        
        # Just copy valid indices
        # We iterate columns? Or use scatter?
        # Simple assignment works:
        
        # out_neighbors[:, :k] = global_inds # This copies even invalid ones (which point to s_start + local_ind)
        # We need to reset invalid ones to M
        
        # Create a temporary tensor for this batch's neighbors
        batch_out = torch.full((q_len, neighbor_limit), total_M, dtype=torch.long, device=q_points.device)
        
        # We can just assign and then mask
        batch_out[:, :k] = global_inds
        
        # Mask out invalid ones
        # valid_mask is (N_i, k)
        # batch_out[:, :k][~valid_mask] = total_M # This syntax might not work for assignment
        
        # Correct way:
        mask_k = batch_out[:, :k].clone()
        mask_k[~valid_mask] = total_M
        batch_out[:, :k] = mask_k
        
        neighbor_indices_list.append(batch_out)
        
        q_start = q_end
        s_start = s_end
        
    if len(neighbor_indices_list) > 0:
        return torch.cat(neighbor_indices_list, dim=0)
    else:
        return torch.empty((0, neighbor_limit), dtype=torch.long, device=q_points.device)
