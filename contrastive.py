import sys
import os

import pytorch_lightning as pl

sys.path.append(os.path.abspath('../LanguageBind'))

print(sys.path)

import languagebind

from omegaconf import OmegaConf
from train import instantiate_from_config

import torch

def main():
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")
    
    conf = OmegaConf.load('configs/contrastive_videoonly.yaml')
    model = instantiate_from_config(conf.model)
    data = instantiate_from_config(conf.data)
    data.prepare_data()
    data.setup()
    
    profiler = pl.profilers.AdvancedProfiler(dirpath='.', filename='profiler_report')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=5,
                                                       monitor='hp_metric',
                                                       mode='min',
                                                       filename='{epoch}-{step}-{hp_metric}')
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         #strategy='ddp',
                         precision='16-mixed',
                         log_every_n_steps=10,
                         val_check_interval=0.50,
                         profiler=profiler)
                         
    
    trainer.fit(model, data)

if __name__ == '__main__':
    main()
