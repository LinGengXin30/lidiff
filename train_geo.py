import os

import click
from os.path import join, dirname, abspath
from os import environ, makedirs
import subprocess
from pytorch_lightning import Trainer, Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import torch
import yaml
import MinkowskiEngine as ME

import lidiff.datasets.datasets as datasets
import lidiff.models.models_geo as models_geo
from lidiff.models.models import DiffusionPoints
from lidiff.utils.scheduling import beta_func
from lidiff.utils.collations import *
from lidiff.utils.metrics import ChamferDistance, PrecisionRecall
from diffusers import DPMSolverMultistepScheduler
import torch.nn.functional as F
import open3d as o3d
from tqdm import tqdm

def set_deterministic():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

class CheckpointDebugCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        print(f"\n[DEBUG] Triggering checkpoint save at epoch {trainer.current_epoch}")

class DiffusionPointsGeo(DiffusionPoints):
    def __init__(self, hparams:dict, data_module = None):
        # We call super init but we need to override the model initialization
        # So we call super init to setup scheduler and other params
        # But we will replace self.model and self.partial_enc
        super().__init__(hparams, data_module)
        
        # Override the model with our Geo model
        backbone_cfg = {
            'feature_dim': 256,
            'num_heads': 4,
            'blocks': ['self', 'cross', 'self', 'cross', 'self', 'cross'],
            'sigma_d': 0.2,
            'sigma_a': 15,
            'angle_k': 3,
            'hidden_dim': 128
        }
        
        fusion_cfg = {
            'd_model': 256
        }
        
        decoder_cfg = {
            'd_model': 256,
            'd_diffusion_input': 32,
            'out_dim': self.hparams['model']['out_dim'],
            # MinkUNet args
            'cr': 1.0,
            'run_up': True,
            'D': 3
        }
        
        self.geo_model = models_geo.LidiffGeoCompletion(backbone_cfg, fusion_cfg, decoder_cfg)
        
        # Remove original model parts to save memory/avoid confusion
        del self.partial_enc
        del self.model
        
    def forward(self, x_full, x_part, t):
        # Placeholder for compatibility with LightningModule hooks if needed
        pass

    def training_step(self, batch:dict, batch_idx):
        # Prepare data
        # batch['pcd_full']: (B, N, 3) GT
        # batch['pcd_part']: (B, M, 3) Input Partial
        
        # initial random noise
        noise = torch.randn(batch['pcd_full'].shape, device=self.device)
        
        # sample step t
        t = torch.randint(0, self.t_steps, size=(batch['pcd_full'].shape[0],)).cuda()
        
        # sample q at step t
        # x_t = x_0 + noise
        t_sample = batch['pcd_full'] + self.q_sample(torch.zeros_like(batch['pcd_full']), t, noise)
        
        # LidiffGeoCompletion inputs:
        # src: Partial (batch['pcd_part'])
        # ref: Reference. We will use batch['pcd_part'] as placeholder reference for now.
        # x_t: Noisy Full (t_sample)
        # t: Timestep
        
        src = batch['pcd_part']
        # If the dataloader provides 'pcd_ref', use it. Else use 'pcd_part'.
        ref = batch.get('pcd_ref', batch['pcd_part'])
        
        noise_pred, gate_score = self.geo_model(src, ref, t_sample, t)
        
        # Loss
        loss_mse = F.mse_loss(noise_pred, noise)
        
        # Auxiliary Loss (Optional)
        # If we had ground truth overlap mask, we would add BCE loss here.
        # loss_aux = F.binary_cross_entropy(gate_score, mask_gt)
        
        loss = loss_mse
        
        self.log('train/loss_mse', loss_mse)
        self.log('train/loss', loss)
        self.log('train/gate_mean', gate_score.mean())
        self.log('train/gate_min', gate_score.min())
        self.log('train/gate_max', gate_score.max())
        
        return loss

    def validation_step(self, batch:dict, batch_idx):
        if batch_idx != 0:
            return

        self.geo_model.eval()
        with torch.no_grad():
            gt_pts = batch['pcd_full'].detach().cpu().numpy()
            
            # For inference
            src = batch['pcd_part']
            ref = batch.get('pcd_ref', batch['pcd_part'])
            
            # Initial noisy points (around partial)
            # x_init = batch['pcd_part'].repeat(1,10,1) # 10x points?
            # Simplified: just same number of points as GT for metric calc
            x_init = batch['pcd_part'] # Or random points? 
            # Lidiff logic: sample noise around partial
            x_init = batch['pcd_part']
            # If we want to generate dense points, we might need to upsample src first?
            # For now keep simple.
            
            # We need a sampling loop here.
            # But p_sample_loop in parent class relies on self.model (MinkUNetDiff).
            # We need to override or adapt it.
            
            # For quick verification of "training works", we can skip full generation
            # and just log validation loss.
            
            noise = torch.randn(batch['pcd_full'].shape, device=self.device)
            t = torch.randint(0, self.t_steps, size=(batch['pcd_full'].shape[0],)).cuda()
            t_sample = batch['pcd_full'] + self.q_sample(torch.zeros_like(batch['pcd_full']), t, noise)
            
            noise_pred, gate_score = self.geo_model(src, ref, t_sample, t)
            val_loss = F.mse_loss(noise_pred, noise)
            
            self.log('val/loss', val_loss)
            
        return {'val/loss': val_loss}

@click.command()
@click.option('--config', '-c', type=str, default=join(dirname(abspath(__file__)),'lidiff/config/config.yaml'))
@click.option('--weights', '-w', type=str, default=None)
def main(config, weights):
    set_deterministic()
    cfg = yaml.safe_load(open(config))
    
    # Override batch size for testing
    cfg['train']['batch_size'] = 2
    cfg['train']['num_workers'] = 0
    
    root_dir = dirname(abspath(__file__))
    exp_dir = join(root_dir, 'experiments', 'geo_test')
    ckpt_dir = join(exp_dir, 'checkpoints')
    makedirs(ckpt_dir, exist_ok=True)
    
    # Initialize our new model
    model = DiffusionPointsGeo(cfg)
    
    # Load data
    data = datasets.dataloaders[cfg['data']['dataloader']](cfg)
    
    # Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(dirpath=ckpt_dir, save_last=True)
    
    trainer = Trainer(
        gpus=1,
        logger=pl_loggers.TensorBoardLogger(exp_dir),
        max_epochs=1,
        limit_train_batches=10, # Short run to verify
        limit_val_batches=2,
        callbacks=[lr_monitor, checkpoint_saver],
        accelerator='gpu',
        devices=1
    )
    
    print('STARTING GEO TRAINING VERIFICATION...')
    trainer.fit(model, data)

if __name__ == "__main__":
    main()
