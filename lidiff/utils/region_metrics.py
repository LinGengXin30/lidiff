import torch
import torch.nn as nn
import numpy as np

class RegionAwareMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        self.cd_overlap = []
        self.cd_non_overlap = []
        self.cd_global = []
        self.f1_global = []

    def reset(self):
        self.cd_overlap = []
        self.cd_non_overlap = []
        self.cd_global = []
        self.f1_global = []

    def compute_pairwise_distance(self, src, tgt):
        """
        Compute pairwise distance between source and target point clouds.
        src: [B, N, 3]
        tgt: [B, M, 3]
        Returns: [B, N, M]
        """
        return torch.cdist(src, tgt, p=2)

    def compute_overlap_mask(self, pc_A, pc_B, threshold=0.05):
        """
        Compute overlap mask for pc_A with respect to pc_B.
        Points in pc_A are considered 'overlap' if min_dist(pc_A, pc_B) < threshold.
        """
        dist_mat = self.compute_pairwise_distance(pc_A, pc_B) # [B, N, M]
        min_dist, _ = torch.min(dist_mat, dim=2) # [B, N]
        return min_dist < threshold

    def update(self, pred, gt, source, threshold=0.05):
        """
        Update metrics for a batch.
        pred: [B, N, 3]
        gt: [B, M, 3]
        source: [B, K, 3] (The partial input)
        """
        # Ensure tensors
        if not torch.is_tensor(pred): pred = torch.tensor(pred, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(gt): gt = torch.tensor(gt, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(source): source = torch.tensor(source, device=self.device, dtype=torch.float32)

        batch_size = pred.shape[0]

        # 1. Compute Distances for Chamfer (Global)
        # dist_pred_gt: [B, N] (min dist from pred to gt)
        # dist_gt_pred: [B, M] (min dist from gt to pred)
        d_mat_pg = self.compute_pairwise_distance(pred, gt)
        dist_pred_gt, idx_pred_gt = torch.min(d_mat_pg, dim=2) # [B, N]
        dist_gt_pred, _ = torch.min(d_mat_pg, dim=1)           # [B, M]

        # Global CD
        cd_g = (torch.mean(dist_pred_gt, dim=1) + torch.mean(dist_gt_pred, dim=1)) / 2
        self.cd_global.extend(cd_g.cpu().tolist())

        # Global F1 (at threshold)
        precision = (dist_pred_gt < threshold).float().mean(dim=1)
        recall = (dist_gt_pred < threshold).float().mean(dim=1)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        self.f1_global.extend(f1.cpu().tolist())

        # 2. Compute Overlap Masks
        # Mask for GT: Which GT points are close to Source?
        mask_gt_overlap = self.compute_overlap_mask(gt, source, threshold) # [B, M]

        # Mask for Pred: Which Pred points map to Overlap GT points?
        # We use the nearest neighbor indices (idx_pred_gt) to gather the mask.
        # mask_gt_overlap is [B, M], idx_pred_gt is [B, N].
        # gather requires index to have same dim as input, so we expand mask? No, gather dim=1.
        mask_pred_overlap = torch.gather(mask_gt_overlap, 1, idx_pred_gt) # [B, N]

        # 3. Compute Split CD
        for b in range(batch_size):
            # Overlap Region
            d1_ov = dist_pred_gt[b][mask_pred_overlap[b]]
            d2_ov = dist_gt_pred[b][mask_gt_overlap[b]]

            # Calculate CD_overlap if there is any ground truth overlap region
            # If d1_ov is empty (no pred points mapped to overlap), we rely on d2_ov
            ov_components = []
            if len(d1_ov) > 0:
                ov_components.append(d1_ov.mean())
            if len(d2_ov) > 0:
                ov_components.append(d2_ov.mean())
            
            if len(ov_components) > 0:
                # If both exist: (mean1 + mean2) / 2
                # If only one exists: mean1 / 1 (maintains scale)
                cd_ov = sum(ov_components) / len(ov_components)
                self.cd_overlap.append(cd_ov.item())
            
            # Non-Overlap Region
            d1_nov = dist_pred_gt[b][~mask_pred_overlap[b]]
            d2_nov = dist_gt_pred[b][~mask_gt_overlap[b]]

            nov_components = []
            if len(d1_nov) > 0:
                nov_components.append(d1_nov.mean())
            if len(d2_nov) > 0:
                nov_components.append(d2_nov.mean())

            if len(nov_components) > 0:
                cd_nov = sum(nov_components) / len(nov_components)
                self.cd_non_overlap.append(cd_nov.item())

    def update_single(self, pred, gt, source, threshold=0.05):
        """
        Update metrics for a single sample (unbatched inputs).
        pred: [N, 3]
        gt: [M, 3]
        source: [K, 3]
        """
        # Ensure tensors and add batch dim
        if not torch.is_tensor(pred): pred = torch.tensor(pred, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(gt): gt = torch.tensor(gt, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(source): source = torch.tensor(source, device=self.device, dtype=torch.float32)
        
        # Add batch dim [1, N, 3]
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
        source = source.unsqueeze(0)
        
        self.update(pred, gt, source, threshold)

    def compute(self):
        return {
            'CD_global': np.mean(self.cd_global) if self.cd_global else 0.0,
            'CD_overlap': np.mean(self.cd_overlap) if self.cd_overlap else 0.0,
            'CD_non_overlap': np.mean(self.cd_non_overlap) if self.cd_non_overlap else 0.0,
            'F1_global': np.mean(self.f1_global) if self.f1_global else 0.0
        }
