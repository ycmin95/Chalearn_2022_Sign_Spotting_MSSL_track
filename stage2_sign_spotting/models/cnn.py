import torch
import torch.nn as nn

class Visual(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type='A'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if conv_type == 'A':
            self.kernel_size = ['K3']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=int(ks[1])//2)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def forward(self, x):
        b,t,c = x.size()
        x = x.permute(0,2,1)
        visual_feat = self.temporal_conv(x) #[b,c,t]
        return visual_feat.permute(0,2,1)