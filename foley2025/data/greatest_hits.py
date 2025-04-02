# import numpy as np
import itertools
import os
from typing import Iterator

import torch
import json
import pytorch_lightning as pl
import languagebind as lb
import cv2
import av
import numpy as np
import tqdm
import random

from utils import instantiate_from_config

CONDFOLEYGEN_SR = 22050
LANGUAGEBIND_VIDEO_NUM_FRAMES = 8

class GreatestHit(torch.utils.data.Dataset):

    def __init__(self,
                 split,
                 data_path,
                 splits_path,
                 metadata_path,
                 duration=2.0,
                 n_frames=30,
                 # p_audio_aug=0.5,
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
        # self.p_audio_aug = p_audio_aug
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
            video_config = lb.LanguageBindVideoConfig.from_pretrained('LanguageBind/LanguageBind_Video_FT', cache_dir='./cache_dir')
            video_config.vision_config.video_decode_backend = 'opencv'
            self.video_preprocessor = lb.LanguageBindVideoProcessor(video_config)
        if self.preprocess_audio:
            audio_config = lb.LanguageBindAudioConfig.from_pretrained('LanguageBind/LanguageBind_Audio_FT', cache_dir='./cache_dir')
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

        video = self.video_preprocess(video_path, start_time, end_time, num_frames=LANGUAGEBIND_VIDEO_NUM_FRAMES)
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

    def video_preprocess(self, video_path, start_time, end_time, num_frames):
        if not self.preprocess_video:
            return []
        
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        fps = video_stream.average_rate
        duration = video_stream.duration * video_stream.time_base
        start_time = max(0, start_time)
        end_time = min(duration, end_time)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Generate frame indices for the desired segment
        frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)

        # Seek to the closest keyframe before start_frame
        container.seek(int(start_time / video_stream.time_base))

        video_data = []
        current_frame_idx = start_frame - 1
        target_frame_idx = 0

        for frame in container.decode(video=0):
            current_frame_idx += 1

            # Skip frames until reaching the target frame
            while target_frame_idx < len(frame_id_list) and current_frame_idx < frame_id_list[target_frame_idx]:
                current_frame_idx += 1
                continue

            # Capture frames at the required frame indices
            if target_frame_idx < len(frame_id_list) and current_frame_idx == frame_id_list[target_frame_idx]:
                frame_rgb = frame.to_rgb().to_ndarray()
                video_data.append(torch.from_numpy(frame_rgb).permute(2, 0, 1))  # Format as C x H x W for PyTorch
                target_frame_idx += 1

                # Stop if all required frames are collected
                if target_frame_idx >= len(frame_id_list):
                    break

        # Raise an error if not all frames were captured
        if len(video_data) != len(frame_id_list):
            raise Exception("Did not find all frames in video.")

        container.close()
        
        # Stack frames and apply preprocessing
        video_data_tensor = torch.stack(video_data, dim=1)
        pixel_values = self.video_preprocessor.transform(video_data_tensor)

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

    def __init__(self, batch_size, num_workers, shuffle_every_epoch, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_every_epoch = shuffle_every_epoch

        self.args = args
        self.kwargs = kwargs

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = GreatestHit('train',
                                         *self.args,
                                         **self.kwargs)
        self.val_dataset = GreatestHit('val',
                                       *self.args,
                                       **self.kwargs)
        self.test_dataset = GreatestHit('test',
                                        *self.args,
                                        **self.kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=self.shuffle_every_epoch,
                                           pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False)




class GreatestHitFullClips(torch.utils.data.Dataset):
    def __init__(self,
                 split,
                 splits_path,
                 data_path):
        super().__init__()

        self.split = split
        self.splits_path = splits_path
        self.data_path = data_path

        if split == 'val':
            split = 'valid'
        split_filepath = os.path.join(splits_path, f'greatesthit_video_{split}.json')
        with open(split_filepath, 'r') as split_file:
            self.split_videos = json.load(split_file)

        self.init_dataset()

    def init_dataset(self):
        self.dataset = []
        for clip in self.split_videos:
            # Test if the files exist
            video_path, audio_path = self.get_video_path(clip), self.get_audio_path(clip)
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f'Error: could not find video at {video_path}')
            if not os.path.isfile(audio_path):
                raise FileNotFoundError(f'Error: could not find audio at {audio_path}')

            self.dataset.append({'clip': clip, 'video_path': video_path, 'audio_path': audio_path})

        print(f'Dataset {self.split} contains {len(self.dataset)} videos')

    def get_video_path(self, video):
        return os.path.join(self.data_path, f'{video}_denoised.mp4')

    def get_audio_path(self, video):
        return os.path.join(self.data_path, f'{video}_denoised.wav')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class GreatestHitFullClipsDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, shuffle_every_epoch, num_workers, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle_every_epoch = shuffle_every_epoch
        self.num_workers = num_workers
        self.args = args
        self.kwargs = kwargs
        self.collate_fn = lambda x: x

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = GreatestHitFullClips('train',
                                                  *self.args,
                                                  **self.kwargs)
        self.val_dataset = GreatestHitFullClips('val',
                                                *self.args,
                                                **self.kwargs)
        self.test_dataset = GreatestHitFullClips('test',
                                                 *self.args,
                                                 **self.kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           collate_fn=self.collate_fn,
                                           shuffle=self.shuffle_every_epoch,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           collate_fn=self.collate_fn,
                                           shuffle=False,
                                           num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           collate_fn=self.collate_fn,
                                           shuffle=False,
                                           num_workers=self.num_workers)

class SequenceTooShortError(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class GreatestHitEmbeddingSequence(pl.LightningModule):
    def __init__(self,
                 video_encoder_config,
                 audio_encoder_config,
                 embedding_save_dir,
                 segment_duration_frames,
                 min_shift,
                 max_shift,
                 min_sequence_length,
                 max_sequence_length,
                 encoder_batch_size):
        super().__init__()

        self.video_encoder_config = video_encoder_config
        self.audio_encoder_config = audio_encoder_config
        self._video_encoder = None
        self._audio_encoder = None

        self.embedding_save_dir = embedding_save_dir

        self.segment_duration_frames = segment_duration_frames
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length

        self.encoder_batch_size = encoder_batch_size

        self._video_preprocessor = None
        self._audio_preprocessor = None

        if not os.path.isdir(self.embedding_save_dir):
            os.makedirs(self.embedding_save_dir, exist_ok=True)

    @property
    def video_preprocessor(self):
        if self._video_preprocessor is None:
            video_config = lb.LanguageBindVideoConfig.from_pretrained('LanguageBind/LanguageBind_Video_FT',
                                                                      cache_dir='./cache_dir')
            video_config.vision_config.video_decode_backend = 'opencv'
            self._video_preprocessor = lb.LanguageBindVideoProcessor(video_config)

        return self._video_preprocessor

    @property
    def audio_preprocessor(self):
        if self._audio_preprocessor is None:
            audio_config = lb.LanguageBindAudioConfig.from_pretrained('LanguageBind/LanguageBind_Audio_FT',
                                                                      cache_dir='./cache_dir')
            self._audio_preprocessor = lb.LanguageBindAudioProcessor(audio_config)

        return self._audio_preprocessor

    @property
    def video_encoder(self):
        if self._video_encoder is None:
            self._video_encoder = instantiate_from_config(self.video_encoder_config)
            self._video_encoder.eval()
        return self._video_encoder.to(device=self.device)

    @property
    def audio_encoder(self):
        if self._audio_encoder is None:
            self._audio_encoder = instantiate_from_config(self.audio_encoder_config)
            self._audio_encoder.eval()
        return self._audio_encoder.to(device=self.device)

    def start_end_times(self, num_segments):
        start_ends = [
            (i * self.segment_duration, (i + 1) * self.segment_duration)
            for i in range(num_segments)
        ]
        return start_ends

    def embedding_save_path(self, clip):
        return os.path.join(self.embedding_save_dir, f'{clip}.pt')


    def video_rgb_frames_generator(self, video_path) -> Iterator[tuple[torch.Tensor, float]]:
        print(f'Decoding {video_path}')
        segment_frames = []
        # num_segments = 0

        with (av.open(video_path) as container):
            new_segment = True
            segment_start_time = 0
            for i, frame in tqdm.tqdm(enumerate(container.decode(video=0)), total=container.streams[0].frames):

                if new_segment:
                    segment_start_time = frame.time
                    new_segment = False

                if i % self.segment_duration_frames == 0 and i != 0:
                    yield (self.video_preprocessor.transform(torch.stack(segment_frames, dim=1)),  # C x T x H x W
                           (segment_start_time, frame.time))  # (start_time, end_time) tuple
                    segment_frames = []
                    new_segment = True

                frame_rgb = frame.to_rgb().to_ndarray()
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)  # Format as C x H x W for pytorch
                segment_frames.append(frame_tensor)

        # Drop the last segment
        return


    def batched_embed_videos(self, video_path) -> Iterator[tuple[torch.Tensor, list[tuple[float, float]]]]:
        batch = []
        batch_start_end_times = []
        i = 0
        for segment_frames, (start_time, end_time) in self.video_rgb_frames_generator(video_path):
            if i % self.encoder_batch_size == 0 and i != 0:
                batch_tensor = torch.stack(batch, dim=0).to(device=self.device)
                with torch.inference_mode():
                    embeddings_batch = self.video_encoder(batch_tensor)

                yield embeddings_batch, batch_start_end_times
                batch = []
                batch_start_end_times = []

            batch_start_end_times.append((start_time, end_time))
            batch.append(segment_frames)

            i += 1

        batch_tensor = torch.stack(batch, dim=0).to(device=self.device)
        with torch.inference_mode():
            embeddings_batch = self.video_encoder(batch_tensor)
        yield embeddings_batch, batch_start_end_times

    def preprocess_clip(self, clip_dict):
        video_path = clip_dict['video_path']
        audio_path = clip_dict['audio_path']

        all_video_embeddings = []
        all_audio_embeddings = []
        for video_embeddings_batch, batch_start_end_times in self.batched_embed_videos(video_path):
            audio_embeddings_batch = self.audio_embeddings(audio_path, batch_start_end_times)

            all_video_embeddings.append(video_embeddings_batch)
            all_audio_embeddings.append(audio_embeddings_batch)
        return torch.cat(all_video_embeddings), torch.cat(all_audio_embeddings)


    def _plot_mel_spec(self, mel_specs, audio_path, idx):
        import matplotlib.pyplot as plt
        import librosa
        fig, axs = plt.subplots(mel_specs.shape[0], 1)

        for i, ax in enumerate(axs.flat):
            mel_spec = mel_specs[i]
            ax.set_title(f'Mel-spec {i}')
            ax.imshow(librosa.power_to_db(mel_spec), origin="lower", aspect="auto", interpolation="nearest")

        plt.savefig(f'mel_specs/{idx}.png')
        plt.close(fig)


    def audio_embeddings(self, audio_path, batch_start_end_times):
        waveform, sample_rate = lb.audio.processing_audio.torchaudio_loader(audio_path)

        preprocessed_frames = []
        for i, (start_time, end_time) in enumerate(batch_start_end_times):
            start_frame = int(start_time * sample_rate)
            end_frame = int(end_time * sample_rate)
            waveform_sliced = waveform[:, start_frame:end_frame]
            mel_specs = self.audio_preprocessor.transform((waveform_sliced, sample_rate))
            #
            # import pdb; pdb.set_trace()
            #
            # self._plot_mel_spec(mel_specs, audio_path, i)
            preprocessed_frames.append(mel_specs)

        with torch.inference_mode():
            raw_audio = torch.stack(preprocessed_frames, dim=0).to(device=self.device)
            audio_embeddings = self.audio_encoder(raw_audio)
        return audio_embeddings


    def load_or_preprocess_clip(self, clip_dict) -> tuple[torch.Tensor, torch.Tensor]:
        clip = clip_dict['clip']
        save_path = self.embedding_save_path(clip)
        if os.path.isfile(save_path):
            video_embeddings, audio_embeddings = torch.load(save_path)
            return video_embeddings, audio_embeddings

        video_embeddings, audio_embeddings = self.preprocess_clip(clip_dict)
        torch.save((video_embeddings, audio_embeddings), save_path)
        return video_embeddings, audio_embeddings


    def random_shifted_sequence(self, original_sequence_length):
        shift = random.randint(self.min_shift, self.max_shift)

        remaining = original_sequence_length - shift
        new_sequence_length = min(remaining, self.max_sequence_length)

        if new_sequence_length < self.min_sequence_length:
            #print(f'Warning: Random sequence of length {new_sequence_length} is too short.')
            raise SequenceTooShortError()

        start_a, end_a = 0, new_sequence_length
        start_b, end_b = shift, new_sequence_length + shift

        absolute_shift = random.randint(0, original_sequence_length - new_sequence_length - shift + 1)

        start_a, end_a = start_a + absolute_shift, end_a + absolute_shift
        start_b, end_b = start_b + absolute_shift, end_b + absolute_shift

        if bool(random.getrandbits(1)):
            return (start_a, end_a), (start_b, end_b)
        else:
            return (start_b, end_b), (start_a, end_a)


    def forward(self, batch):
        return self.load_all_embeddings(batch)

    def load_all_embeddings(self, path):
        video_embeddings, audio_embeddings = self.load_or_preprocess_clip(path)

        num_segments = video_embeddings.shape[0]
        assert audio_embeddings.shape[0] == num_segments

        try:
            (orig_start, orig_end), (shifted_start, shifted_end) = self.random_shifted_sequence(num_segments)
        except SequenceTooShortError:
            return None

        orig_video_embeddings = video_embeddings[orig_start:orig_end].to(device=self.device)
        orig_audio_embeddings = audio_embeddings[orig_start:orig_end].to(device=self.device)
        shifted_video_embeddings = video_embeddings[shifted_start:shifted_end].to(device=self.device)
        shifted_audio_embeddings = audio_embeddings[shifted_start:shifted_end].to(device=self.device)

        sliced_video_embeddings = video_embeddings[orig_start:orig_end].to(device=self.device)
        sliced_audio_embeddings = audio_embeddings[shifted_start:shifted_end].to(device=self.device)

        return {
            'unshifted_video': orig_video_embeddings,
            'unshifted_audio': orig_audio_embeddings,
            'shifted_video': shifted_video_embeddings,
            'shifted_audio': shifted_audio_embeddings
        }

        return sliced_video_embeddings, sliced_audio_embeddings


class DummyTensors(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs):
        # inputs: B x C x T x W x H
        batch_size = inputs.shape[0]
        return torch.randn(batch_size, 768, device=inputs.device)

class ZeroTensors(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        return torch.zeros(batch_size, 768, device=inputs.device)


class GreatestHitEmbeddingSequenceDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, shuffle_every_epoch, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.shuffle_every_epoch = shuffle_every_epoch

        self.args = args
        self.kwargs = kwargs
        # print(args)
        # print(kwargs)
        # print('============')

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = GreatestHitEmbeddingSequence('train', *self.args, **self.kwargs)
        self.val_dataset =  GreatestHitEmbeddingSequence('val', *self.args, **self.kwargs)
        self.test_dataset =  GreatestHitEmbeddingSequence('test', *self.args, **self.kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=self.shuffle_every_epoch,
                                           pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False)

if __name__ == '__main__':
    x = GreatestHitEmbeddingSequence(
        split='train',
        splits_path='data',
        video_encoder_config={
            'target': 'specvqgan.models.contrastive_pretraining_clip.LB_VideoEncoder_PartiallyFrozen',
            'params': {
                'n_finetune_layers': 5
            }
        },
        audio_encoder_config={},
        embedding_save_dir='/tmp/embeddings_test',
        segment_duration_frames=LANGUAGEBIND_VIDEO_NUM_FRAMES,
        min_shift_frames=None,
        max_shift_frames=None,
        min_segments=None,
        max_segments=None,
        encoder_batch_size=10,
    )

    print([q for q in x])

    import pdb; pdb.set_trace()
