# Plan: Implement Gated Attention Fusion Completion Network

We will implement a new completion network `LidiffGatedCompletion` in a new file `lidiff/models/models_fusion.py`. This network uses a Siamese `MinkResNet` (specifically `MinkGlobalEnc` from the existing codebase) for feature extraction, a custom `GatedAttentionFusion` module for fusing source and reference features, and a standard `MinkUNet` decoder for the final completion.

## File Structure

- **New File**: `lidiff/models/models_fusion.py`

## Implementation Steps

### 1. The Backbone - Siamese MinkResNet
- **Class**: `SiameseFeatureExtractor`
- **Location**: `lidiff/models/models_fusion.py`
- **Logic**:
  - Initialize a single `MinkGlobalEnc` instance (from `lidiff.models.minkunet`) to act as the shared backbone.
  - Implement `forward(src_pcd, ref_pcd)`:
    - Pass `src_pcd` (ME.SparseTensor) through the backbone to get `F_src_sparse`.
    - Pass `ref_pcd` (ME.SparseTensor) through the backbone to get `F_ref_sparse`.
    - Return both sparse feature tensors.
  - **Note**: `MinkGlobalEnc` returns features at the last stage (stride 16), which satisfies the requirement for "features at specific stride".

### 2. The Innovation - Dense Attention & Gating
- **Class**: `GatedAttentionFusion`
- **Location**: `lidiff/models/models_fusion.py`
- **Logic**:
  - **Sparse to Dense**: Implement a helper to convert `ME.SparseTensor` (batch of sparse points) to padded dense tensors `(B, N_max, D)` and masks.
  - **Cross-Attention**:
    - Query = `F_src_dense`
    - Key, Value = `F_ref_dense`
    - Compute attention to get `aligned_feats`.
  - **Overlap Gating**:
    - Apply MLP (Linear -> ReLU -> Linear -> Sigmoid) on `aligned_feats` to get `gate_score`.
  - **Residual Fusion**:
    - `F_fused = F_src_dense + gate_score * aligned_feats`.
  - **Dense to Sparse**:
    - Map `F_fused` back to the original sparse coordinates of the source (`F_src_sparse.C`) to create `F_condition_sparse` (ME.SparseTensor).
  - Return `F_condition_sparse` and `gate_score`.

### 3. Integration - The Main Pipeline
- **Class**: `LidiffGatedCompletion`
- **Location**: `lidiff/models/models_fusion.py`
- **Inheritance**: `nn.Module` (or `pl.LightningModule` if needed, but `nn.Module` is standard for internal models).
- **Components**:
  - `self.backbone`: `SiameseFeatureExtractor`.
  - `self.fusion_module`: `GatedAttentionFusion`.
  - `self.decoder`: `MinkUNetDiff` (from `lidiff.models.minkunet`).
- **Forward Pass**:
  - `forward(src, ref, x_t, t)`:
    1.  **Extract**: `F_src, F_ref = self.backbone(x_t, ref)`
    2.  **Fuse**: `F_cond, gates = self.fusion_module(F_src, F_ref)`
    3.  **Decode**: `noise_pred = self.decoder(x_t, t, F_cond)`
    4.  Return `noise_pred, gates`.

## Verification
- We will verify the implementation by checking:
  - Import success.
  - Shape consistency in `forward` pass (mock inputs).
  - Correct sparse-dense-sparse conversion.
