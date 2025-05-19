import tqdm
import random
import torch
import collections
import pytorch_lightning as pl
if __name__ == '__main__':
    import sys
    sys.path.append('../LanguageBind')
import languagebind as lb
import av
import json

import torchaudio

from utils import instantiate_from_config


CONDFOLEYGEN_SR = 22050


def random_ranges(n, k, tot) -> list[tuple[int, int]]:
    required = n * k
    if required > tot:
        raise ValueError(f'Cannot generate {n} ranges of length {k} for {tot} items.')

    available = tot - required
    gaps = [0] * (n + 1)

    # Distribute 'available' units into gaps
    for _ in range(available):
        idx = random.randint(0, n)  # Includes all n+1 gaps
        gaps[idx] += 1

    # Compute start positions
    starts = []
    current = gaps[0]
    starts.append(current)
    for i in range(1, n):
        current += k + gaps[i]
        starts.append(current)

    return starts

class GreatestHitEmbeddingsSequence_Encoder(pl.LightningModule):
    def __init__(self,
                 segment_duration_frames,
                 metadata_path,
                 segments_per_video):
        super().__init__()
        self.segment_duration_frames = segment_duration_frames

        self.hit_index = None
        self.hit_class_numbers = None
        self.load_hit_index(metadata_path)

        # self.hit_index = self.load_hit_index(metadata_path)
        self.segments_per_video = segments_per_video
        self._video_preprocessor = None
        self._audio_preprocessor = None

    def load_hit_index(self, metadata_path):
        with open(metadata_path, 'r') as meta_file:
            greatesthit_meta = json.load(meta_file)

        self.hit_index = collections.defaultdict(list)
        for v, i, l in zip(greatesthit_meta['video_name'],
                           greatesthit_meta['start_idx'],
                           greatesthit_meta['hit_type'],
                           strict=True):
            self.hit_index[v].append((i, l))

        self.hit_class_numbers = {l: i for i, l in enumerate(sorted(list(set(ht for ht in greatesthit_meta['hit_type']))) + [None])}

    def timestamp_to_hit_label(self, video_name, start_timestamp, end_timestamp) -> str | None:
        # Note: if there are multiple labels within the timeframe, pick one at random
        start_idx = start_timestamp * CONDFOLEYGEN_SR
        end_idx = end_timestamp * CONDFOLEYGEN_SR
        label_candidates = []
        for i, l in self.hit_index[video_name]:
            if start_idx <= i <= end_idx:
                label_candidates.append(l)

        if len(label_candidates) == 0:
            return None
        return random.choice(label_candidates)

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

    def forward(self, batch) -> dict[str, list[torch.Tensor | str]]:
        data = collections.defaultdict(list)
        for clip_dict in batch:
            processed_dict = self.load_clip_dict(clip_dict)
            for k, v in processed_dict.items():
                data[k].append(v)

        for k, v in data.items():
            if 'hit_labels' in k:
                continue  # This can contain None, and is not used as input to any models.

            data[k] = torch.stack(v, dim=0)

        return data


    def load_clip_dict(self, clip_dict) -> dict[str, list[torch.Tensor | str | None]]:
        video_data = []
        audio_data = []
        hit_labels = []
        hit_class_nums = []

        audio_waveform, orig_audio_sr = lb.audio.processing_audio.torchaudio_loader(clip_dict['audio_path'])
        target_sr = self.audio_preprocessor.transform.sample_rate
        if orig_audio_sr != target_sr:
            audio_waveform = torchaudio.functional.resample(audio_waveform, orig_freq=orig_audio_sr, new_freq=target_sr)


        with av.open(clip_dict['video_path']) as container:
            n_video_frames = container.streams[0].frames
            frame_starts = random_ranges(self.segments_per_video,
                                         self.segment_duration_frames,
                                         n_video_frames)
            current_start_frame_idx = 0
            frame_generator = container.decode(video=0)
            i = 0
            while True:
                if i < frame_starts[current_start_frame_idx]:
                    # Drop frames
                    _ = next(frame_generator)
                    i += 1
                    continue

                # Process 8 frames of video
                frame = next(frame_generator)
                start_time = frame.time
                frame_rgb = frame.to_rgb().to_ndarray()
                frame_tensors = [torch.from_numpy(frame_rgb).permute(2, 0, 1)]
                for _ in range(self.segment_duration_frames - 1):
                    i += 1
                    frame = next(frame_generator)
                    frame_rgb = frame.to_rgb().to_ndarray()
                    frame_tensors.append(torch.from_numpy(frame_rgb).permute(2, 0, 1))  # C x H x W
                end_time = frame.time
                frame_tensors = torch.stack(frame_tensors, dim=1).to(device=self.device)  # C x T x H x W
                video_data.append(self.video_preprocessor.transform(frame_tensors))

                # Process audio
                audio_start = int(start_time * target_sr)
                audio_end = int(end_time * target_sr)
                waveform_slice = audio_waveform[:, audio_start:audio_end]
                audio_data.append(self.audio_preprocessor.transform((waveform_slice, target_sr)))

                # Load hit label
                hit_label = self.timestamp_to_hit_label(clip_dict['clip'], start_time, end_time)
                hit_labels.append(hit_label)
                hit_class_nums.append(self.hit_class_numbers[hit_label])

                current_start_frame_idx += 1
                if current_start_frame_idx == len(frame_starts):
                    break

        return {
            'video_data': torch.stack(video_data, dim=0).to(device=self.device),
            'audio_data': torch.stack(audio_data, dim=0).to(device=self.device),
            'hit_class_nums': torch.tensor(hit_class_nums, device=self.device),
            'hit_labels': hit_labels,
        }


if __name__ == '__main__':
    from foley2025.data.greatest_hits import GreatestHitFullClips
    data = GreatestHitFullClips('train', 'data', '/workspace/vis-data')
    x = GreatestHitEmbeddingsSequence_Encoder(8, 'data/info_r2plus1d_dim1024_15fps.json', 3)

    for i in data:
        print(i)
        print(x([i])[0]['hit_labels'])
        print(x([i])[0]['hit_class_nums'])
    import pdb; pdb.set_trace()