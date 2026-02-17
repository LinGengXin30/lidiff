
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import numpy as np
from lidiff.models.minkunet import MinkUNet

class SiameseFeatureExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Using MinkUNet as the backbone to ensure we get point-wise features 
        # that align with the input resolution (stride 1).
        # We explicitly set in_channels to 3 (assuming input is coords/colors)
        backbone_kwargs = kwargs.copy()
        backbone_kwargs['in_channels'] = 3
        # out_channels is set by the caller (feature_dim)
        self.backbone = MinkUNet(**backbone_kwargs)

    def forward(self, src_sparse, ref_sparse):
        # Shared weights
        F_src = self.backbone(src_sparse)
        F_ref = self.backbone(ref_sparse)
        return F_src, F_ref

class GatedAttentionFusion(nn.Module):
    def __init__(self, feature_dim, n_head=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=n_head, batch_first=True, dropout=dropout)
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
    def sparse_to_dense(self, sparse_tensor):
        # Decomposition returns a list of tensors [N1, C], [N2, C], ...
        batch_list = sparse_tensor.decomposition()
        max_n = max([t.shape[0] for t in batch_list])
        batch_size = len(batch_list)
        feature_dim = batch_list[0].shape[1]
        device = batch_list[0].device
        
        # Initialize dense tensor and mask
        # We use 0 padding for features
        dense_tensor = torch.zeros(batch_size, max_n, feature_dim, device=device)
        # Mask: True indicates the value should be IGNORED (padding)
        mask = torch.ones(batch_size, max_n, dtype=torch.bool, device=device)
        
        lengths = []
        for i, t in enumerate(batch_list):
            n = t.shape[0]
            lengths.append(n)
            dense_tensor[i, :n, :] = t
            mask[i, :n] = False 
            
        return dense_tensor, mask, lengths

    def forward(self, F_src, F_ref):
        # 1. Sparse -> Dense
        D_src, mask_src, src_lens = self.sparse_to_dense(F_src)
        D_ref, mask_ref, ref_lens = self.sparse_to_dense(F_ref)
        
        # 2. Cross Attention
        # Q = D_src, K = D_ref, V = D_ref
        # key_padding_mask: True for ignored positions
        D_aligned, _ = self.attn(D_src, D_ref, D_ref, key_padding_mask=mask_ref)
        
        # 3. Overlap Gating
        gate_score = self.gate_mlp(D_aligned) # (B, Max_N, 1)
        
        # 4. Residual Fusion
        D_out = D_src + gate_score * D_aligned
        
        # 5. Dense -> Sparse
        # Map back to original coordinates
        flattened_feats = []
        for i in range(len(src_lens)):
            n = src_lens[i]
            # Extract valid features, ignoring padding
            flattened_feats.append(D_out[i, :n, :])
            
        flattened_feats = torch.cat(flattened_feats, dim=0)
        
        # Create output SparseTensor with same coordinates as F_src
        F_condition = ME.SparseTensor(
            features=flattened_feats,
            coordinate_map_key=F_src.coordinate_map_key,
            coordinate_manager=F_src.coordinate_manager
        )
        
        return F_condition, gate_score

class LidiffGatedCompletion(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        feature_dim = kwargs.get('feature_dim', 32) # Default output dim of backbone
        
        # Backbone: SiameseFeatureExtractor
        # We configure it to output 'feature_dim' channels
        backbone_kwargs = kwargs.copy()
        backbone_kwargs['out_channels'] = feature_dim
        self.extractor = SiameseFeatureExtractor(**backbone_kwargs)
        
        # Fusion Module
        self.fusion = GatedAttentionFusion(feature_dim=feature_dim)
        
        # Decoder: Standard MinkUNet
        # Input channels = x_t (3) + condition (feature_dim) + time_emb (feature_dim)
        # We will project time embedding to feature_dim and concat
        decoder_in_channels = 3 + feature_dim + feature_dim
        
        decoder_kwargs = kwargs.copy()
        decoder_kwargs['in_channels'] = decoder_in_channels
        decoder_kwargs['out_channels'] = 3 # Predict noise/displacement
        self.decoder = MinkUNet(**decoder_kwargs)
        
        # Time Embedding Projection
        self.time_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.embed_dim = feature_dim

    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        Sinusoidal embeddings for time steps
        """
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(timesteps.device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        return emb

    def forward(self, src, ref, x_t, t):
        # 1. Extract Features (Siamese)
        # src and ref should be ME.SparseTensor
        F_src, F_ref = self.extractor(src, ref)
        
        # 2. Fuse Features
        # F_condition aligns with src coordinates
        F_condition, gate_scores = self.fusion(F_src, F_ref)
        
        # 3. Prepare Decoder Input
        # x_t is the noisy input (ME.SparseTensor or TensorField)
        if isinstance(x_t, ME.TensorField):
            x_t_sparse = x_t.sparse()
        else:
            x_t_sparse = x_t
            
        # Time Embedding
        t_emb = self.get_timestep_embedding(t, self.embed_dim)
        t_emb = self.time_proj(t_emb) # (B, C)
        
        # Expand time embedding to per-point features
        # We need to map (B, C) to (Total_N, C) based on batch indices
        batch_indices = x_t_sparse.C[:, 0].long()
        t_emb_expanded = t_emb[batch_indices]
        
        t_emb_sparse = ME.SparseTensor(
            features=t_emb_expanded,
            coordinate_map_key=x_t_sparse.coordinate_map_key,
            coordinate_manager=x_t_sparse.coordinate_manager
        )
        
        # Concatenate: x_t + F_condition + t_emb
        # Ensure F_condition aligns with x_t
        # If x_t corresponds to src, they should align.
        decoder_input = ME.cat(x_t_sparse, F_condition, t_emb_sparse)
        
        # 4. Decode
        noise_pred = self.decoder(decoder_input)
        
        return noise_pred, gate_scores
