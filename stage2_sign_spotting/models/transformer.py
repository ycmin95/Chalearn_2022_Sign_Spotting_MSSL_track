import torch
import pdb
import math
import torch.nn as nn


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class transformer_model(nn.Module):
    def __init__(self, input_size, dropout_ratio):
        super().__init__()
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=8,
            dropout=dropout_ratio
        )
        self.encoder = nn.TransformerEncoder(transformer_encoder_layer, 2)
        self.pos_encoder = PositionalEncoding(input_size, dropout_ratio)
        # self.classifier = nn.Sequential(
        #     nn.Linear(kargs['input_size'], kargs['hidden_size']),
        #     nn.BatchNorm1d(kargs['hidden_size']),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(kargs['hidden_size'], kargs['num_classes'])
        # )
        # self.dim_decrease = nn.Sequential(
        #     nn.Linear(kargs['input_size'], kargs['hidden_size']),
        #     nn.BatchNorm1d(kargs['hidden_size']),
        #     nn.ReLU(inplace=True)
        # )

    def generate_mask(self, inputs, len_x, max_len):
        batch = len(len_x)
        mask = torch.ones((batch, max_len), device=inputs.device)
        for i in range(batch):
            mask[i, 0:len_x[i]] = 0
        mask = (mask == 1)
        return mask
        
    def forward(self, x, len_x, ret_type='classify'):
        # x [b,t,c] len_x [b,]
        b, t, c = x.size()
        padded_mask = self.generate_mask(x, len_x, torch.max(len_x))  # [b,t]
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)  # [t,b,c]
        output = self.encoder(x, src_key_padding_mask=padded_mask)  # [t,b,c]
        output = output.permute(1, 0, 2).contiguous()  # [b,t,c]
        if ret_type == 'classify':
            logits = self.classifier(output.view(b * t, c)).view(b, t, -1)
            return {
                'transformer_classify_logits': logits
            }
        else:
            # features = self.dim_decrease(output.view(b*t,c)).view(b,t,-1)
            return {
                'transformer_feat': output
            }


if __name__ == '__main__':
    kargs = {'input_size': 512, 'hidden_size': 512, 'dropout': 0.3, 'num_classes': 61}
    model = transformer_model(**kargs)
    inputs = torch.rand(2, 100, 512)
    len_x = torch.tensor([60, 100])
    model(inputs, len_x)
