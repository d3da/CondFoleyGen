import pytorch_lightning as pl
from omegaconf import OmegaConf

import torch
import sys
print(sys.path)

sys.path.insert(0, '/')
sys.path.append('../LanguageBind/')


from utils import instantiate_from_config


def main():

    conf = OmegaConf.load('configs/temporal.yaml')
    model = instantiate_from_config(conf.model)
    data = instantiate_from_config(conf.data)

    trainer = pl.Trainer(accelerator='cpu', fast_dev_run=10)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()