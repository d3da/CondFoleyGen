import json
import sys
import torch

sys.path.insert(0, '../LanguageBind')

from specvqgan.data.gh2 import GreatestHit
from specvqgan.models.contrastive_pretraining_clip import LB_LabelEncoder


def create_label_embeddings(gh_meta_filepath: str = 'data/info_r2plus1d_dim1024_15fps.json',
                            label_embeddings_path: str = 'data/GH_label_embeddings.pt'):
    """
    TODO: Transform the text of the labels to something more useful
    """
    label_encoder = LB_LabelEncoder()

    with open(gh_meta_filepath, 'r') as meta_file:
        meta = json.load(meta_file)

    unique_classes = sorted(list(set(ht for ht in meta['hit_type'])))
    #label2hit_class = {label: i for i, label in enumerate(unique_classes)}

    none_mask = []
    for label in unique_classes:
        material, action = label.split(' ')
        if material == 'None' or action == 'None':
            none_mask.append(0)
        else:
            none_mask.append(1)
        

    label_embeddings = label_encoder(unique_classes)
    none_mask = torch.tensor(none_mask)
    torch.save(dict(label_embeddings=label_embeddings, none_mask=none_mask), label_embeddings_path)


if __name__ == '__main__':
    create_label_embeddings()
