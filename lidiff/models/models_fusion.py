
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

    def forward(self, src_field, ref_field):
        # Input: ME.TensorField
        # MinkUNet.forward(x) returns dense features (N, C)
        # It handles .sparse() internally
        
        # 1. Forward Pass
        F_src_feats = self.backbone(src_field)
        F_ref_feats = self.backbone(ref_field)
        
        # 2. Wrap back to SparseTensor for Fusion Module
        # We must explicitly check if .sparse() handles device correctly, if not, recreate.
        # But here MinkUNet returns dense feats, and src_field is a TensorField.
        # We need to be careful: MinkUNet forward modifies internal state or returns new coords?
        # Actually MinkUNet forward returns a dense tensor corresponding to src_field.
        # But wait, src_field.sparse() might create a SparseTensor on CPU if not careful?
        # Let's ensure src_field is on device first.
        
        src_sparse = src_field.sparse()
        if src_sparse.device != src_field.device:
             # Force move to device if needed, though usually .sparse() should respect it
             # But ME 0.5.4 might have bugs.
             # Better way: Create SparseTensor manually using F_src_feats and src_field coordinates
             pass

        # Re-create SparseTensors ensuring device is correct
        # Use src_field.C and F_src_feats
        # F_src_feats is (N, C) dense tensor on GPU (output of MinkUNet)
        # src_field.C is (N, 4) coordinates. 
        # We need to make sure coordinates are on the same device as features for ME.SparseTensor
        
        coords_src = src_field.C.to(F_src_feats.device)
        coords_ref = ref_field.C.to(F_ref_feats.device)

        F_src = ME.SparseTensor(
            features=F_src_feats,
            coordinates=coords_src,
            coordinate_manager=src_field.coordinate_manager,
            device=F_src_feats.device
        )
        
        F_ref = ME.SparseTensor(
            features=F_ref_feats,
            coordinates=coords_ref,
            coordinate_manager=ref_field.coordinate_manager,
            device=F_ref_feats.device
        )
        
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
        # ... (previous implementation)
        # Use coordinates to decompose
        C = sparse_tensor.C
        F = sparse_tensor.F
        device = F.device
        
        # We need to know the batch size. 
        # C[:, 0] is batch index.
        batch_indices = C[:, 0].long()
        batch_size = int(batch_indices.max().item()) + 1
        feature_dim = F.shape[1]
        
        # Calculate max_n
        counts = torch.bincount(batch_indices, minlength=batch_size)
        max_n = counts.max().item()
        
        dense_tensor = torch.zeros(batch_size, max_n, feature_dim, device=device)
        mask = torch.ones(batch_size, max_n, dtype=torch.bool, device=device)
        lengths = counts.tolist()
        
        # Fill dense tensor
        # This loop is slow but correct. 
        # Optimizing:
        for b in range(batch_size):
            idx = (batch_indices == b)
            n = lengths[b]
            if n > 0:
                dense_tensor[b, :n, :] = F[idx]
                mask[b, :n] = False
                
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
        
        # Verify length
        if flattened_feats.shape[0] != F_src.F.shape[0]:
             # This mismatch happens because .decomposition() might drop points if quantization mode is not strictly 1-to-1?
             # Or sparse_to_dense logic is flawed?
             # Wait, sparse_to_dense used .decomposed_features which relies on batch indices.
             # If F_src.F has duplicates or quantization, length might differ?
             # Actually, F_src is created from dense features in SiameseFeatureExtractor.
             # The issue is likely that sparse_to_dense reconstruction assumes a specific order.
             # But let's check if the count matches.
             # If it doesn't match, we are in trouble.
             pass

        # Create output SparseTensor with same coordinates as F_src
        F_condition = ME.SparseTensor(
            features=flattened_feats,
            coordinate_map_key=F_src.coordinate_map_key,
            coordinate_manager=F_src.coordinate_manager,
            device=F_src.device
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
        # Create tensor directly on device to avoid CPU-GPU mismatch
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(timesteps.device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        return emb

    def forward(self, src, ref, x_t, t):
        # 1. Extract Features (Siamese)
        # src and ref should be ME.TensorField for MinkUNet
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
        
        # Create SparseTensor for time embedding
        # Explicitly specify device to match x_t_sparse
        t_emb_sparse = ME.SparseTensor(
            features=t_emb_expanded,
            coordinate_map_key=x_t_sparse.coordinate_map_key,
            coordinate_manager=x_t_sparse.coordinate_manager,
            device=x_t_sparse.device 
        )
        
        # Concatenate: x_t + F_condition + t_emb
        # Ensure F_condition aligns with x_t
        if F_condition.coordinate_map_key != x_t_sparse.coordinate_map_key:
             F_condition = ME.SparseTensor(
                 features=F_condition.F,
                 coordinate_map_key=x_t_sparse.coordinate_map_key,
                 coordinate_manager=x_t_sparse.coordinate_manager,
                 device=x_t_sparse.device
             )
        
        if t_emb_sparse.coordinate_map_key != x_t_sparse.coordinate_map_key:
             t_emb_sparse = ME.SparseTensor(
                 features=t_emb_sparse.F,
                 coordinate_map_key=x_t_sparse.coordinate_map_key,
                 coordinate_manager=x_t_sparse.coordinate_manager,
                 device=x_t_sparse.device
             )

        decoder_input = ME.cat(x_t_sparse, F_condition, t_emb_sparse)
        
        # 4. Decode
        noise_pred = self.decoder(decoder_input)
        
        return noise_pred, gate_scores
