import torch
import torch.nn as nn
import torch.nn.functional as F

class Gate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # [h_{t-1}, x_t] = input_size + hidden_size
        concat_size = input_size + hidden_size
        self.W_z = nn.Linear(concat_size, hidden_size)
        self.W_r = nn.Linear(concat_size, hidden_size)
        self.W_h = nn.Linear(concat_size, hidden_size)

    def update_gate(self, x):
        z = torch.sigmoid(self.W_z(x))
        return z # z_t

    def reset_gate(self, x):
        r = torch.sigmoid(self.W_r(x))
        return r  # r_t
    
    def candidate_hidden(self, h_prev, x, r):
            concat_rh = torch.cat((r * h_prev, x), dim=1)
            h_hat = torch.tanh(self.W_h(concat_rh))
            return h_hat # h_hat_t

class GRU(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gate = Gate(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        embedded = self.embedding(x)  # (batch, seq_len, hidden_size)

        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch, hidden_size)
            combined = torch.cat((h_t, x_t), dim=1)  # (batch, hidden+hidden)

            # Update gate
            z_t = self.gate.update_gate(combined)

            # Reset gate
            r_t = self.gate.reset_gate(combined)

            h_hat_t = self.gate.candidate_hidden(h_t, x_t, r_t)

            # hidden state update
            h_t = (1 - z_t) * h_t + z_t * h_hat_t

        out = self.fc(h_t)
        return out

