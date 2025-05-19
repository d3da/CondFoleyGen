import torch
import json
from collections import OrderedDict

import sys
sys.path.insert(0, '../LanguageBind')
from foley2025.contrastive.encoder import LB_LabelEncoder


def map_label_text(label: str | None) -> str:
    print(label)
    if label is None:
        return 'Nothing being hit by a drumstick'

    material, hit_type = label.split(' ')

    if hit_type == 'hit':
        return f'{material.capitalize()} being hit by a drumstick'
    elif hit_type == 'scratch':
        return f'{material.capitalize()} being scratched by a drumstick'
    elif hit_type == 'None':
        return f'{material.capitalize()} not being hit by a drumstick'

    raise ValueError(f'Unknown hit type {hit_type}')


def create_label_embeddings(label_encoder: torch.nn.Module,
                            gh_meta_filepath: str = 'data/info_r2plus1d_dim1024_15fps.json',
                            label_embeddings_path: str = 'data/GH_label_embeddings_v2.pt'):
    with open(gh_meta_filepath, 'r') as meta_file:
        meta = json.load(meta_file)

    hit_text = OrderedDict()

    unique_classes = sorted(list(set(ht for ht in meta['hit_type']))) + [None]
    for i, hit_type in enumerate(unique_classes):
        mapped = map_label_text(hit_type)
        hit_text[hit_type] = mapped

    print(hit_text)

    label_embeddings = label_encoder([text for text in hit_text.values()])
    print(label_embeddings.shape)

    torch.save(dict(label_embeddings=label_embeddings, hit_text=hit_text), label_embeddings_path)
    print('Done')


if __name__ == '__main__':
    label_encoder = LB_LabelEncoder()
    create_label_embeddings(label_encoder)
