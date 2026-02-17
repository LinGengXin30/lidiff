import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
from lidiff.models.minkunet import MinkUNetDiff

# Attempt to import GeoTransformer. 
# If the module is missing in the environment, we provide a placeholder to ensure the code structure is valid.
# The user is expected to have 'geotransformer' package installed or in the python path.
try:
    from geotransformer.modules.geotransformer import GeometricTransformer
except ImportError:
    print("Warning: GeoTransformer not found. Using a placeholder for GeometricEncoderWrapper.")
    class GeometricTransformer(nn.Module):
        def __init__(self, input_dim=3, output_dim=256, **kwargs):
            super().__init__()
            self.output_dim = output_dim
        def forward(self, ref_pcd, src_pcd):
            # Placeholder forward
            # Return random features for testing structure
            B, N, _ = src_pcd.shape
            M = ref_pcd.shape[1]
            return {
                'src_feats': torch.zeros(B, N, self.output_dim, device=src_pcd.device),
                'ref_feats': torch.zeros(B, M, self.output_dim, device=ref_pcd.device)
            }

class GeometricEncoderWrapper(nn.Module):
    """
    Step 1: The Backbone - SE(3) Invariant Feature Extraction
    Wraps GeoTransformer to extract dense, point-wise features.
    """
    def __init__(self, feature_dim=256, **kwargs):
        super().__init__()
        # Initialize GeometricTransformer
        # We assume standard usage. Adjust arguments based on actual GeoTransformer API.
        self.encoder = GeometricTransformer(
            input_dim=3,
            output_dim=feature_dim,
            **kwargs
        )
        self.feature_dim = feature_dim

    def forward(self, src_pcd, ref_pcd):
        """
        Input:
            src_pcd: (Batch, N, 3)
            ref_pcd: (Batch, M, 3)
        Output:
            src_feats: (Batch, N, D_model)
            ref_feats: (Batch, M, D_model)
        """
        # GeoTransformer typically handles batching internally or expects list of tensors.
        # Here we assume it handles (B, N, 3) input or we might need to collate.
        # For this implementation, we assume the wrapper handles the API call.
        
        # Note: Ensure src_pcd and ref_pcd are in the format GeoTransformer expects.
        # Often GeoTransformer takes a dictionary or separate arguments.
        output = self.encoder(ref_pcd, src_pcd)
        
        # Extract features from the output dictionary
        # We target the Fine-level Transformer output for dense features.
        if isinstance(output, dict):
             # Adjust keys based on actual GeoTransformer output
            src_feats = output.get('src_feats', output.get('src_points_feature', None))
            ref_feats = output.get('ref_feats', output.get('ref_points_feature', None))
        else:
            # Fallback if output is tuple
            src_feats, ref_feats = output
            
        return src_feats, ref_feats

class GatedFeatureFusion(nn.Module):
    """
    Step 2: The Core - "Overlap-Gated" Fusion Module
    Aligns Source and Reference features and suppresses non-overlapping regions.
    """
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        
        # Cross-Attention Unit
        # Query = src, Key/Value = ref
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        
        # Overlap Gating Network
        # Input: aligned_feats concatenated with src_feats (2 * d_model)
        # Layers: Linear(2D) -> ReLU -> Linear(D) -> Linear(1) -> Sigmoid
        # Note: User specified Linear(D) -> ReLU -> Linear(1) -> Sigmoid.
        # But input is 2*D (cat). So first layer is 2D -> D.
        self.gating_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, src_feats, ref_feats):
        """
        Input:
            src_feats: (Batch, N, D)
            ref_feats: (Batch, M, D)
        Output:
            condition_feats: (Batch, N, D) - F_condition
            gate_score: (Batch, N, 1) - w
        """
        # 1. Cross-Attention
        # Finds the "most similar structure" in the Reference for each Source point.
        # attn_output: (Batch, N, D)
        aligned_feats, _ = self.cross_attn(query=src_feats, key=ref_feats, value=ref_feats)
        
        # 2. Overlap Gating Network
        # Concatenate src and aligned features
        combined = torch.cat([src_feats, aligned_feats], dim=-1) # (Batch, N, 2*D)
        
        # Predict gate score w
        gate_score = self.gating_net(combined) # (Batch, N, 1)
        
        # 3. Residual Fusion
        # F_condition = F_src + w * F_aligned
        # Explanation: If w=0 (no overlap), condition degrades to F_src.
        condition_feats = src_feats + gate_score * aligned_feats
        
        return condition_feats, gate_score

class MinkUNetGeo(MinkUNetDiff):
    """
    Adapted MinkUNetDiff for Geometry-Aware Diffusion.
    1. Removes explicit 'part_feats' matching (since we use Input Concatenation).
    2. Supports SparseTensor input/output directly (fixing .slice() issue).
    3. Retains Time Embedding injection (modulating features with time).
    """
    def forward(self, x_sparse, t):
        # We assume x_sparse is a ME.SparseTensor containing [x_t, condition].
        
        temp_emb = self.get_timestep_embedding(t)

        x0 = self.stem(x_sparse)
        
        # Helper to compute modulation weights w
        # mode: 'stage' (p, t) or 'up' (t, p)
        def get_w(idx, x_curr, mode='stage'):
            # Get time embedding projection
            # e.g. stage1_temp or up1_temp
            prefix = f'stage{idx}' if mode == 'stage' else f'up{idx}'
            temp_proj = getattr(self, f'{prefix}_temp')(temp_emb)
            
            # Expand time embedding to match batch size of sparse tensor
            # x_curr.C[:, 0] is the batch index
            batch_indices = x_curr.C[:, 0]
            # We need to repeat t_emb for each point based on its batch index
            # t_emb is (B, D). x_curr is (N, C).
            # We gather t_emb using batch_indices.
            # batch_indices is long tensor.
            t_expanded = temp_proj[batch_indices.long()]
            
            # Get part embedding (dummy zeros)
            # latent_{prefix} is the projection for part features.
            # We access the first layer to get input dimension.
            latent_layer = getattr(self, f'latent_{prefix}')
            if isinstance(latent_layer, nn.Sequential):
                in_dim = latent_layer[0].in_features
            else:
                in_dim = latent_layer.in_features
                
            p_emb = torch.zeros(x_curr.F.shape[0], in_dim, device=x_curr.device)
            
            # Combine p and t
            # latemp_{prefix} is the fusion layer
            latemp_layer = getattr(self, f'latemp_{prefix}')
            
            if mode == 'stage':
                # (p, t)
                w = latemp_layer(torch.cat((p_emb, t_expanded), -1))
            else:
                # (t, p) for up layers
                w = latemp_layer(torch.cat((t_expanded, p_emb), -1))
                
            return w

        # Encoder
        w0 = get_w(1, x0, 'stage')
        x1 = self.stage1(x0 * w0)
        
        w1 = get_w(2, x1, 'stage')
        x2 = self.stage2(x1 * w1)
        
        w2 = get_w(3, x2, 'stage')
        x3 = self.stage3(x2 * w2)
        
        w3 = get_w(4, x3, 'stage')
        x4 = self.stage4(x3 * w3)
        
        # Decoder
        # Up1 (corresponds to stage4 output -> up1)
        w4 = get_w(1, x4, 'up')
        y1 = self.up1[0](x4 * w4)
        y1 = ME.cat(y1, x3)
        y1 = self.up1[1](y1)
        
        w5 = get_w(2, y1, 'up')
        y2 = self.up2[0](y1 * w5)
        y2 = ME.cat(y2, x2)
        y2 = self.up2[1](y2)
        
        w6 = get_w(3, y2, 'up')
        y3 = self.up3[0](y2 * w6)
        y3 = ME.cat(y3, x1)
        y3 = self.up3[1](y3)
        
        w7 = get_w(4, y3, 'up')
        y4 = self.up4[0](y3 * w7)
        y4 = ME.cat(y4, x0)
        y4 = self.up4[1](y4)
        
        return self.last(y4.F)

class DiffusionDecoder(nn.Module):
    """
    Step 3: Diffusion Decoder Integration
    Injects conditioned features into the Denoising Network (MinkUNet).
    """
    def __init__(self, d_model=256, d_diffusion_input=32, out_dim=3, **kwargs):
        super().__init__()
        
        # Refinement 3: Projection Layer
        # Compress GeoTransformer features (D_model) to Diffusion input dim (D_diffusion_input)
        self.d_diffusion_input = d_diffusion_input
        self.proj = nn.Linear(d_model, d_diffusion_input)
        
        # Initialize MinkUNetDiff
        # We increase in_channels to accommodate the conditioned features
        # in_channels = 3 (xyz) + d_diffusion_input
        # Use MinkUNetGeo instead of MinkUNetDiff
        self.model = MinkUNetGeo(in_channels=3 + d_diffusion_input, out_channels=out_dim, **kwargs)

    def forward(self, x_t, t, condition_feats):
        """
        Input:
            x_t: (Batch, N, 3) - Noisy points
            t: (Batch,) - Timestep
            condition_feats: (Batch, N, D_model)
        """
        # 1. Project condition features
        cond_proj = self.proj(condition_feats) # (Batch, N, 32)
        
        # 2. Concatenate inputs (Refinement 1)
        # input_feats: (Batch, N, 3 + 32)
        input_feats = torch.cat([x_t, cond_proj], dim=-1)
        
        # 3. Create SparseTensor
        # Coordinates: x_t (Batch, N, 3) -> need to batchify
        # We assume x_t is a tensor. We need to create batch indices.
        # ME.utils.batched_coordinates expects a list of tensors/arrays.
        
        batch_size = x_t.shape[0]
        coords_list = [x_t[i] for i in range(batch_size)]
        
        # Create batched coordinates (N_total, 4)
        # Note: We must ensure device compatibility
        coords = ME.utils.batched_coordinates(coords_list, dtype=torch.float32, device=x_t.device)
        
        # Flatten features (N_total, C)
        feats_flat = input_feats.view(-1, input_feats.shape[-1])
        
        # Create input SparseTensor
        x_in = ME.SparseTensor(
            features=feats_flat,
            coordinates=coords,
            device=x_t.device
        )
        
        # 4. Forward Pass through MinkUNetGeo
        # Only pass x_in and t. MinkUNetGeo handles the rest.
        noise_pred = self.model(x_in, t)
        
        # Output is (N_total, 3)
        # We need to reshape back to (Batch, N, 3)
        
        return noise_pred.view(batch_size, -1, 3)

class LidiffGeoCompletion(nn.Module):
    """
    Step 4: The Forward Pass & Loss
    Assembles the full pipeline.
    """
    def __init__(self, backbone_cfg, fusion_cfg, decoder_cfg):
        super().__init__()
        self.backbone = GeometricEncoderWrapper(**backbone_cfg)
        self.fusion_module = GatedFeatureFusion(**fusion_cfg)
        self.decoder = DiffusionDecoder(**decoder_cfg)
        
    def forward(self, src, ref, x_t, t):
        """
        Input:
            src: (Batch, N, 3)
            ref: (Batch, M, 3)
            x_t: (Batch, N, 3) - Noisy src
            t: (Batch,)
        Output:
            noise_pred: (Batch, N, 3)
            gate_score: (Batch, N, 1) - For visualization/aux loss
        """
        # 1. Extract Invariant Features (Backbone)
        # src_feats: (Batch, N, D_model)
        # ref_feats: (Batch, M, D_model)
        src_feats, ref_feats = self.backbone(src, ref)
        
        # 2. Gated Fusion (Innovation)
        # condition_feats: (Batch, N, D_model)
        # gate_score: (Batch, N, 1)
        condition_feats, gate_score = self.fusion_module(src_feats, ref_feats)
        
        # 3. Denoise (Decoder)
        # noise_pred: (Batch, N, 3)
        noise_pred = self.decoder(x_t, t, condition_feats)
        
        return noise_pred, gate_score
