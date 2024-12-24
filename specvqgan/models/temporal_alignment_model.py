

import pytorch_lightning as pl
import torch


class LSTMAlignmentModel(pl.LightningModule):
    def __init__(self, input_size, output_size, n_layers, dropout=0.2):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=output_size,
                                  num_layers=n_layers,
                                  dropout=dropout,
                                  bidirectional=True,
                                  device=self.device,
                                  batch_first=True)

    def forward(self, x):
        return self.lstm(x)[0]



