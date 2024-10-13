# import numpy as np
import os
import torch
import json
import pytorch_lightning as pl
import languagebind as lb

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
                 remove_single_hits=False,
                 remove_none_materials=False,
                 remove_none_actions=False,
                 preprocess_video=False,
                 preprocess_audio=False):
        super().__init__()
        self.split = split
        self.data_path = data_path
        self.duration = duration
        self.n_frames = n_frames
        self.p_audio_aug = p_audio_aug
        self.metadata_path = metadata_path
        self.remove_single_hits = remove_single_hits
        self.remove_none_material = remove_none_materials
        self.remove_none_actions = remove_none_actions

        self.preprocess_video = preprocess_video
        self.preprocess_audio = preprocess_audio
        self.init_preprocessors()

        with open(self.metadata_path, 'r') as meta_file:
            self.greatesthit_meta = json.load(meta_file)
        split_filepath = os.path.join(splits_path, f'greatesthit_{split}.json')
        with open(split_filepath, 'r') as split_file:
            self.split_videos = json.load(split_file)

        self.init_dataset()

    def init_preprocessors(self):
        if self.preprocess_video:
            video_config = lb.LanguageBindVideoConfig.from_pretrained('LanguageBind/LanguageBind_Video_FT')
            video_config.vision_config.video_decode_backend = 'pytorchvideo'
            self.video_preprocessor = lb.LanguageBindVideoProcessor(video_config)
        if self.preprocess_audio:
            audio_config = lb.LanguageBindAudioConfig.from_pretrained('LanguageBind/LanguageBind_Audio_FT')
            self.audio_preprocessor = lb.LanguageBindAudioProcessor(audio_config)


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
        dataset_set = set()  # for efficient lookups 'x in set'
        for video, start_ids in self.video2idx.items():

            # Test if the files exist
            video_path, audio_path = self.get_video_path(video), self.get_audio_path(video)
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f'Error: could not find video at {video_path}')
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(f'Error: could not find audio at {audio_path}')

            for idx in start_ids:
                self.dataset.append((video, idx))
                dataset_set.add((video, idx))

        self.video2label = {(v, int(i)): l
                            for v, i, l in zip(self.greatesthit_meta['video_name'],
                                               self.greatesthit_meta['start_idx'],
                                               self.greatesthit_meta['hit_type'],
                                               strict=True)
                            if (v, int(i)) in dataset_set}

        unique_classes = sorted(list(set(ht for ht in self.greatesthit_meta['hit_type'])))
        self.label2hit_class = {label: i for i, label in enumerate(unique_classes)}

        if self.remove_none_material or self.remove_none_actions:
            self.remove_none_videos()

        print(f'Dataset {self.split} contains {len(self.dataset)} videos')

    def remove_single_hit_videos(self):
        for video, idx_list in self.video2idx.items():
            if len(idx_list) == 1:
                self.video2idx.pop(video)


    def remove_none_videos(self):
        for video, id in self.dataset:
            label = self.video2label[(video, int(id))]
            material, action = label.split(' ')
            if self.remove_none_material and material == 'None' \
                    or self.remove_none_actions and action == 'None':
                self.dataset.remove((video, id))

    def __len__(self):
        return len(self.dataset)


    def get_video_path(self, video):
        return os.path.join(self.data_path, f'{video}_denoised.mp4')

    def get_audio_path(self, video):
        return os.path.join(self.data_path, f'{video}_denoised.wav')


    def __getitem__(self, i):
        video, start_idx = self.dataset[i]

        video_path = self.get_video_path(video)
        audio_path = self.get_audio_path(video)

        start_time = self.idx_to_seconds(start_idx)
        end_time = start_time + self.duration

        label = self.video2label[(video, start_idx)]
        hit_class = self.label2hit_class[label]

        video = self.video_preprocess(video_path, start_time, end_time)
        audio = self.audio_preprocess(audio_path, start_time, end_time)

        return dict(video_path=video_path,
                    audio_path=audio_path,
                    start_time=start_time,
                    end_time=end_time,
                    duration=self.duration,
                    label=label,
                    hit_class=hit_class,
                    video=video,
                    audio=audio)


    def idx_to_seconds(self, idx: int) -> float:
        return idx / CONDFOLEYGEN_SR

    def video_preprocess(self, video_path, start_time, end_time):
        if not self.preprocess_video:
            return []

        pixel_values = self.video_preprocessor.image_processor(video_path,
                                                               self.video_preprocessor.transform,
                                                               video_decode_backend='pytorchvideo',
                                                               clip_start_sec=start_time,
                                                               clip_end_sec=end_time,
                                                               num_frames=None)['video']
        return pixel_values

    def audio_preprocess(self, audio_path, start_time, end_time):
        if not self.preprocess_audio:
            return []

        waveform, sample_rate = lb.audio.processing_audio.torchaudio_loader(audio_path)
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        waveform_sliced = waveform[:, start_frame:end_frame]
        pixel_values = self.audio_preprocessor.transform((waveform_sliced, sample_rate))
        return pixel_values


class GreatestHitDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, num_workers, *args, **kwargs):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.args = args
        self.kwargs = kwargs

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = GreatestHit('train', *self.args, **self.kwargs)
        self.val_dataset   = GreatestHit('val',   *self.args, **self.kwargs)
        self.test_dataset  = GreatestHit('test',  *self.args, **self.kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)
