# import numpy as np
import sys
import os
import torch
import json
import pytorch_lightning as pl

CONDFOLEYGEN_SR = 22050

class GreatestHit(torch.utils.data.Dataset):

    def __init__(self,
                 split,
                 data_path,
                 splits_path,
                 metadata_path,
                 duration=2.0,
                 n_frames=30,
                 p_audio_aug=0.5,
                 rand_shift=True,
                 remove_single_hits=False):
        super().__init__()
        self.split = split
        self.data_path = data_path
        self.duration = duration
        self.n_frames = n_frames
        self.p_audio_aug = p_audio_aug
        self.rand_shift = rand_shift
        self.metadata_path = metadata_path
        self.remove_single_hits = remove_single_hits
        
        with open(self.metadata_path, 'r') as meta_file:
            self.greatesthit_meta = json.load(meta_file)
        split_filepath = os.path.join(splits_path, f'greatesthit_{split}.json')
        with open(split_filepath, 'r') as split_file:
            self.split_videos = json.load(split_file)

        self.init_dataset()


    def init_dataset(self):
        self.video2idx = {}
        for video_start_idx in self.split_videos:
            video, start_idx = video_start_idx.split('_')
            start_idx = int(start_idx)
            if video not in self.video2idx.keys():
                self.video2idx[video] = [start_idx]
            else:
                self.video2idx[video].append(start_idx)

        if self.remove_single_hits:
            self.remove_single_hit_videos()

        self.dataset = []
        for video, start_ids in self.video2idx.items():
            for idx in start_ids:
                self.dataset.append((video, idx))

        self.video2label = {(v, int(i)): l
                            for v, i, l in zip(self.greatesthit_meta['video_name'],
                                               self.greatesthit_meta['start_idx'],
                                               self.greatesthit_meta['hit_type'],
                                               strict=True)
                            if (v, int(i)) in self.dataset}


    def remove_single_hit_videos(self):
        for video, idx_list in self.video2idx.items():
            if len(idx_list) == 1:
                self.video2idx.pop(video)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, i):
        """
        TODO video/audio transforms, augmentation, returning the right section of the video/audio
        """
        video, start_idx = self.dataset[i]

        video_path = os.path.join(self.data_path, f'{video}_denoised.mp4')
        audio_path = os.path.join(self.data_path, f'{video}_denoised.wav')

        start_time = self.idx_to_seconds(start_idx)
        end_time = start_time + self.duration

        label = self.video2label[(video, start_idx)]

        return dict(video_path=video_path,
                    audio_path=audio_path,
                    start_time=start_time,
                    end_time=end_time,
                    duration=self.duration,
                    label=label)


    def idx_to_seconds(self, idx: int) -> float:
        return idx / CONDFOLEYGEN_SR

        

class GreatestHitDataModule(pl.LightningDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = GreatestHit('train', *self.args, **self.kwargs)
        self.val_dataset   = GreatestHit('val',   *self.args, **self.kwargs)
        self.test_dataset  = GreatestHit('test',  *self.args, **self.kwargs)

    def train_dataloader(self):
        """TODO batch_size"""
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        """TODO batch_size"""
        return torch.utils.data.DataLoader(self.val_dataset)

    def test_dataloader(self):
        """TODO batch_size"""
        return torch.utils.data.DataLoader(self.test_dataset)
