import torch
import torch.nn as nn
import torch.nn.functional as F
        
class RNN(nn.Module):
    def __init__(self, in_ch, hidden_ch, num_classes=2):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch

        self.W_xh = nn.Linear(in_ch, hidden_ch)
        self.W_hh = nn.Linear(hidden_ch, hidden_ch)
        self.fc = nn.Linear(hidden_ch, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.size()
        h_t = torch.zeros(batch_size, self.hidden_ch, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]
            x_t = nn.functional.one_hot(x_t, num_classes=self.in_ch).float()
            h_t = torch.tanh(self.W_xh(x_t) + self.W_hh(h_t))
            outputs.append(h_t.unsqueeze(1))

        h_last = h_t
        out = self.fc(h_last)
        
        return out






