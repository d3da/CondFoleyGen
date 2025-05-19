import sys
import pytorch_lightning as pl

sys.path.append('../LanguageBind')

import languagebind

from omegaconf import OmegaConf
from utils import instantiate_from_config

import torch

def main():
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')

    conf = OmegaConf.load('configs/contrastive/encoder_v2.yaml')
    model = instantiate_from_config(conf.model)
    data = instantiate_from_config(conf.data)
    wandb_conf = conf.wandb


    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=5,
                                                       monitor='hp_metric',
                                                       mode='min',
                                                       filename='{epoch}-{step}-{hp_metric}')
    profiler = pl.profilers.AdvancedProfiler(dirpath='.', filename='encoder_profiler_report')
    #logger = pl.loggers.WandbLogger(offline=True,
    #                                save_dir='wandb_logs',
    #                                project='CondFoleyGen',
    #                                log_model=False,
    #                                config=OmegaConf.to_object(conf),
    #                                **wandb_conf)
    #logger.watch(model)

    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         # fast_dev_run=10,
                         profiler=profiler,
                         #logger=logger,
                         log_every_n_steps=1,
                         accumulate_grad_batches=32,
                         callbacks=[checkpoint_callback])

    print('TODO: gradient accumulation')

    trainer.fit(model, data)



if __name__ == '__main__':
    main()