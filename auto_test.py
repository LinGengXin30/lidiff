
import os
import yaml
import torch
import click
import pandas as pd
from pytorch_lightning import Trainer
from lidiff.models.models import DiffusionPoints
from lidiff.datasets.datasets import dataloaders
from os.path import join, dirname, abspath
import MinkowskiEngine as ME

@click.command()
@click.option('--exp_id', type=str, required=True, help='Experiment ID (e.g., prob10_5p0reg)')
@click.option('--ckpt_dir', type=str, default=None, help='Path to checkpoints directory. If None, inferred from exp_id.')
@click.option('--uncond_w_list', type=str, default="0.0,2.0,4.0,6.0,8.0,10.0", help='Comma separated list of unconditional weights (guidance scales) to test. Default: 0.0,2.0,4.0,6.0,8.0,10.0')
@click.option('--limit_batches', type=int, default=10, help='Number of batches to test per checkpoint.')
@click.option('--save_pcd', is_flag=True, default=False, help='Whether to save generated point clouds.')
@click.option('--s_steps', type=int, default=10, help='Diffusion steps for fast testing (default 10). Lower is faster.')
def evaluate_grid(exp_id, ckpt_dir, uncond_w_list, limit_batches, save_pcd, s_steps):
    """
    Evaluate checkpoints with multiple weights per pass.
    """
    
    root_dir = dirname(dirname(abspath(__file__)))
    if ckpt_dir is None:
        ckpt_dir = join(root_dir, 'experiments', exp_id, 'checkpoints')
    
    if not os.path.exists(ckpt_dir):
        print(f"Error: {ckpt_dir} not found.")
        return

    # Find checkpoints
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    try:
        ckpts.sort(key=lambda x: int(x.split('epoch_')[1].split('.')[0]))
    except:
        ckpts.sort()
        
    print(f"Found {len(ckpts)} checkpoints.")
    
    uncond_weights = [float(x) for x in uncond_w_list.split(',')]
    
    # Iterate over checkpoints
    for ckpt_name in ckpts:
        ckpt_path = join(ckpt_dir, ckpt_name)
        print(f"\n{'='*50}")
        print(f"Evaluating Checkpoint: {ckpt_name}")
        print(f"{'='*50}")
        
        try:
            # Load Model
            model = DiffusionPoints.load_from_checkpoint(ckpt_path)
            
            # Global config override for fast testing
            model.hparams['experiment']['save_test_pcd'] = save_pcd
            model.hparams['diff']['s_steps'] = s_steps
            model.dpm_scheduler.set_timesteps(s_steps)
            
            # Setup Data (only once per checkpoint)
            data_cfg = model.hparams
            data_module = dataloaders[data_cfg['data']['dataloader']](data_cfg)
            
            if torch.cuda.device_count() > 1:
                 model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

            # --- EXTERNAL LOOP OVER PARAMETERS ---
            # We run the trainer.test() multiple times for the same checkpoint
            # This avoids modifying models.py, while still testing all weights.
            
            for w in uncond_weights:
                print(f"\n  Testing with uncond_w = {w} ...")
                
                # Update model parameter
                model.w_uncond = w
                model.hparams['train']['uncond_w'] = w # Keep consistent
                
                # We need a new Trainer instance for each run because PL closes loops
                trainer = Trainer(
                    gpus=1,
                    logger=False,
                    limit_test_batches=limit_batches,
                    accelerator='gpu' if torch.cuda.is_available() else 'cpu'
                )
                
                # Run Test
                # metrics is usually a list of dicts, but limit_test_batches returns accumulated results
                # in a single dict if using a single dataloader.
                # However, PL might return tensors still on GPU in that dict.
                test_out = trainer.test(model, dataloaders=data_module, verbose=False)
                
                # Extract the dict from the list
                metrics = test_out[0] if isinstance(test_out, list) and len(test_out) > 0 else {}

                # Print Result
                # Metrics from PL might be tensors on GPU
                cd = metrics.get('test/cd_mean', -1)
                f1 = metrics.get('test/fscore', -1)
                
                # Robustly handle tensor/gpu conversion
                try:
                    if hasattr(cd, 'cpu'): cd = cd.cpu()
                    if hasattr(cd, 'numpy'): cd = cd.numpy()
                    if hasattr(cd, 'item'): cd = cd.item()
                except: pass

                try:
                    if hasattr(f1, 'cpu'): f1 = f1.cpu()
                    if hasattr(f1, 'numpy'): f1 = f1.numpy()
                    if hasattr(f1, 'item'): f1 = f1.item()
                except: pass
                
                print(f"  -> Result [w={w}]: CD={cd:.4f}, F1={f1:.4f}")

        except Exception as e:
            print(f"Error evaluating {ckpt_name}: {e}")

if __name__ == "__main__":
    evaluate_grid()
