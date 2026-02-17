
import os
import json
import io
import contextlib
import torch
import click
from pytorch_lightning import Trainer
from lidiff.models.models import DiffusionPoints
import lidiff.models.models as models_mod
from lidiff.datasets.datasets import dataloaders
from os.path import join, dirname, abspath
from typing import Any, Dict

_orig_tensor_numpy = torch.Tensor.numpy

def _patched_tensor_numpy(self, *args, **kwargs):
    if getattr(self, "is_cuda", False):
        return _orig_tensor_numpy(self.detach().cpu(), *args, **kwargs)
    return _orig_tensor_numpy(self, *args, **kwargs)

torch.Tensor.numpy = _patched_tensor_numpy

models_mod.tqdm = lambda x: x

def _silence_pl_logs():
    try:
        from pytorch_lightning.utilities import rank_zero
        rank_zero.rank_zero_info = lambda *a, **k: None
        rank_zero.rank_zero_debug = lambda *a, **k: None
        rank_zero.rank_zero_warn = lambda *a, **k: None
    except Exception:
        pass

_silence_pl_logs()

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
    
    root_dir = dirname(abspath(__file__))
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
    
    def safe_float(x):
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
        try:
            return float(x)
        except Exception:
            return None

    def get_metric(metrics, keys):
        for k in keys:
            if k in metrics:
                return metrics[k]
        return None

    results = []

    # Iterate over checkpoints
    for ckpt_name in ckpts:
        ckpt_path = join(ckpt_dir, ckpt_name)
        print(f"Evaluating: {ckpt_name}")
        
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

            # --- EXTERNAL LOOP OVER PARAMETERS ---
            # We run the trainer.test() multiple times for the same checkpoint
            # This avoids modifying models.py, while still testing all weights.
            
            for w in uncond_weights:
                print(f"  w={w}: ", end="", flush=True)

                # Update model parameter
                model.w_uncond = w
                model.hparams['train']['uncond_w'] = w # Keep consistent
                
                # We need a new Trainer instance for each run because PL closes loops
                # Also silence hardware/progress outputs during construction
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        trainer = Trainer(
                            gpus=1,
                            logger=False,
                            limit_test_batches=limit_batches,
                            enable_checkpointing=False,
                            enable_progress_bar=False,
                            enable_model_summary=False,
                        )
                    except TypeError:
                        try:
                            trainer = Trainer(
                                gpus=1,
                                logger=False,
                                limit_test_batches=limit_batches,
                                checkpoint_callback=False,
                                progress_bar_refresh_rate=0,
                                weights_summary=None,
                            )
                        except TypeError:
                            trainer = Trainer(
                                gpus=1,
                                logger=False,
                                limit_test_batches=limit_batches,
                            )
                
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    test_out = trainer.test(model, dataloaders=data_module, verbose=False)
                
                # Extract the dict from the list
                metrics: Dict[str, Any] = test_out[0] if isinstance(test_out, list) and len(test_out) > 0 else {}
                if not isinstance(metrics, dict) or len(metrics) == 0:
                    cm = getattr(trainer, "callback_metrics", {})
                    if hasattr(cm, "items"):
                        try:
                            metrics = {k: (v.detach().cpu().item() if torch.is_tensor(v) else v) for k, v in cm.items()}
                        except Exception:
                            metrics = dict(cm)

                cd_val = get_metric(metrics, ['test/cd_mean', 'test_cd_mean', 'test/cd_mean_epoch', 'test_cd_mean_epoch', 'cd_mean'])
                f1_val = get_metric(metrics, ['test/fscore', 'test_fscore', 'test/fscore_epoch', 'test_fscore_epoch', 'fscore'])
                cd = safe_float(cd_val) if cd_val is not None else None
                f1 = safe_float(f1_val) if f1_val is not None else None

                results.append({
                    'checkpoint': ckpt_name,
                    'checkpoint_path': ckpt_path,
                    'uncond_w': w,
                    'cd_mean': cd,
                    'f1': f1,
                })

                if cd is None or f1 is None:
                    print("no-metrics")
                else:
                    print(f"CD={cd:.6f} F1={f1:.6f}")

        except Exception as e:
            print(f"Error evaluating {ckpt_name}: {e}")

    valid = [r for r in results if isinstance(r.get('f1'), float)]
    best = None
    if valid:
        best = sorted(valid, key=lambda r: (-r['f1'], r['cd_mean'] if r['cd_mean'] is not None else 1e9))[0]

    out_dir = join(root_dir, 'experiments', exp_id)
    os.makedirs(out_dir, exist_ok=True)
    out_json = join(out_dir, 'evaluation_results.json')
    out_txt = join(out_dir, 'evaluation_results.txt')
    payload = {'exp_id': exp_id, 'limit_batches': limit_batches, 's_steps': s_steps, 'results': results, 'best': best}
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(out_txt, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f"{r['checkpoint']}\tw={r['uncond_w']}\tCD={r['cd_mean']}\tF1={r['f1']}\n")
        f.write("\n")
        f.write(f"BEST\t{best}\n")

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_txt}")
    if best is not None:
        print(f"Best: {best['checkpoint']} w={best['uncond_w']} CD={best['cd_mean']} F1={best['f1']}")

if __name__ == "__main__":
    evaluate_grid()
