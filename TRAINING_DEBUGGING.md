# Training Debugging Guide

## Common Issues and Solutions

### 1. Training Hanging After Epoch Completion

**Problem:** Training appears to hang after epoch 0 completes and before epoch 1 starts.

**Root Causes and Solutions:**

#### A. Memory Management Issues
- **Issue**: Excessive calls to `torch.cuda.empty_cache()` can cause performance bottlenecks
- **Solution**: Removed unnecessary `torch.cuda.empty_cache()` calls from model forward passes and training/validation steps. PyTorch Lightning handles memory management automatically in most cases.

#### B. Checkpoint Saving Configuration
- **Issue**: Default checkpoint configuration may not save intermediate epochs properly
- **Solution**: Updated ModelCheckpoint callback with explicit configuration:
  ```python
  ModelCheckpoint(
      dirpath=f'experiments/{cfg["experiment"]["id"]}/checkpoints',
      filename=cfg['experiment']['id']+'_epoch_{epoch:02d}',
      save_top_k=-1,
      save_last=True,
      every_n_epochs=1,
      verbose=True
  )
  ```

#### C. Validation Frequency
- **Issue**: Heavy validation steps running too frequently
- **Solution**: Consider adjusting validation frequency with `check_val_every_n_epoch` parameter

### 2. Missing Checkpoint Files

**Problem:** No checkpoint file found after epoch 0

**Solutions:**
- Checkpoints are now saved to: `experiments/{experiment_id}/checkpoints/`
- Both `{experiment_id}_epoch_{epoch:02d}.ckpt` and `last.ckpt` files are created
- Use `save_last=True` to ensure latest checkpoint is always available

### 3. Performance Optimizations Applied

1. **Reduced Memory Management Overhead**: Removed redundant `torch.cuda.empty_cache()` calls
2. **Explicit Checkpoint Path**: Defined clear checkpoint saving location
3. **Better Checkpoint Options**: Enabled `save_last` and `verbose` options for better tracking

### 4. Monitoring Training Progress

To monitor your training progress:

```bash
# Monitor checkpoints being created
watch -n 1 'find experiments/ -name "*.ckpt" -type f -newer /tmp/start_time'

# View TensorBoard logs
tensorboard --logdir=experiments/

# Check training logs
tail -f experiments/{experiment_id}/lightning_logs/version_*/events.out.tfevents.*
```

### 5. Recommended Training Configuration

For stable training, consider these parameters:

- Use single GPU initially (`n_gpus: 1`) to avoid DDP-related issues
- Reduce `num_workers` if experiencing data loading issues
- Enable mixed precision training if supported (`precision=16`)
- Monitor GPU memory usage: `watch -n 1 nvidia-smi`

### 6. Troubleshooting Steps

If training still hangs:

1. **Check system resources**:
   ```bash
   free -h  # Check RAM
   nvidia-smi  # Check GPU
   df -h  # Check disk space
   ```

2. **Reduce batch size** or number of workers temporarily

3. **Verify data loading** by testing with a smaller dataset

4. **Monitor the process** with `htop` or `ps aux | grep python`

5. **Check for deadlocks** in multiprocessing by reducing `num_workers` to 0

### 7. Recovery from Stuck Training

If training gets stuck:

1. Use Ctrl+C to interrupt
2. Resume from last checkpoint using `--checkpoint` flag:
   