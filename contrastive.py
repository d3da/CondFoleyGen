import sys
import os

import pytorch_lightning as pl

sys.path.append(os.path.abspath('../LanguageBind'))

print(sys.path)

import languagebind

from omegaconf import OmegaConf
from train import instantiate_from_config

# sys.path.append(os.getcwd())

# conf = OmegaConf.load('configs/contrastive_textbound_av.yaml')
# conf = OmegaConf.load('configs/contrastive_new.yaml')
conf = OmegaConf.load('configs/contrastive_videoonly.yaml')
model = instantiate_from_config(conf.model)
data = instantiate_from_config(conf.data)
data.prepare_data()
data.setup()

x = data.train_dataloader()

# vid = model.video_encoder(['./data/demo_video/chopping.mp4'])
# print(vid)

# aud = model.audio_encoder(['./data/greatesthit/greatesthit_processed/2015-02-16-16-49-06/audio/2015-02-16-16-49-06_denoised.wav'])
# print(aud)

# lbl = model.label_encoder(['The quick brown fox jumps over the lazy dog'])
# print(lbl)

# [print(a) for a in x]

# for batch in x:
    # model.shared_step(batch)

trainer = pl.Trainer(accelerator="cpu")

trainer.fit(model, data)

# import pdb; pdb.set_trace()
