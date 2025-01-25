import pytorch_lightning as pl
from omegaconf import OmegaConf

import torch
import sys
print(sys.path)

#sys.path.insert(0, '/')
sys.path.append('../LanguageBind/')


from utils import instantiate_from_config


def main():
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("high")

    conf = OmegaConf.load('configs/temporal.yaml')
    model = instantiate_from_config(conf.model)
    data = instantiate_from_config(conf.data)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=5,
                                                       monitor='hp_metric',
                                                       mode='min',
                                                       filename='{epoch}-{step}-{hp_metric}')
    # profiler = pl.profilers.AdvancedProfiler(dirpath='.', filename='profiler_repor')
    logger = pl.loggers.WandbLogger(name='CondFoleyGen',
                                    offline=True,
                                    save_dir='wandb_logs',
                                    project='CondFoleyGen',
                                    log_model=False)
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         # fast_dev_run=10,
                         profiler='simple',
                         logger=logger,
                         callbacks=[checkpoint_callback])

    trainer.fit(model, data)


if __name__ == '__main__':
    main()
