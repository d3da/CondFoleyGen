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
        self.save_hyperparameters()

    def forward(self, x):
        return self.lstm(x)[0]




class TransformerAlignmentModel(pl.LightningModule):
    """
    Encoder-only transformer model with learned positional embeddings.
    """
    def __init__(self, input_size, output_size, n_layers, sequence_length, num_heads, dropout=0.2):
        super().__init__()

        self.embed_dim = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length

        self.positional_embeddings = torch.nn.Embedding(self.sequence_length, self.embed_dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.embed_dim,
                                                         nhead=num_heads,
                                                         dim_feedforward=4 * self.embed_dim,
                                                         dropout=dropout,
                                                         batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        self.output_projection = torch.nn.Linear(self.embed_dim, self.output_size)

        self.save_hyperparameters()

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        assert x.shape[1] == self.sequence_length
        assert x.shape[2] == self.embed_dim
        batch_size = x.shape[0]

        positions = torch.arange(self.sequence_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.positional_embeddings(positions)
        x = x + pos_embeddings

        x = self.transformer_encoder(x)
        x = self.output_projection(x)
        return x



class FullyConnectedAlignmentModel(pl.LightningModule):
    """
    Simple MLP. Treats every embedding independently! It does not combine information across time.
    """
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout=0.2):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.layers = torch.nn.Sequential()
        if n_layers == 1:
            self.layers.append(torch.nn.Linear(self.input_size, self.output_size))
            self.layers.append(torch.nn.ReLU())
        else:
            self.layers.append(torch.nn.Linear(self.input_size, self.hidden_size))
            self.layers.append(torch.nn.ReLU())
            for n in range(n_layers - 2):
                self.layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size))
                self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Linear(self.hidden_size, self.output_size))
            self.layers.append(torch.nn.ReLU())

        self.save_hyperparameters()

    def forward(self, x):
        return self.layers(x)

