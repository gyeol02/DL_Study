import torch
import torch.nn as nn
import torch.nn.functional as F

class Gate(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # [h_{t-1}, x_t] = input_size + hidden_size
        concat_size = input_size + hidden_size
        self.W_f = nn.Linear(concat_size, hidden_size)
        self.W_o = nn.Linear(concat_size, hidden_size)
        self.W_i = nn.Linear(concat_size, hidden_size)
        self.W_C = nn.Linear(concat_size, hidden_size)

    def input_gate(self, x):
        i = torch.sigmoid(self.W_i(x))
        C_hat = torch.tanh(self.W_C(x))
        return i, C_hat  # i_t, C̃_t

    def output_gate(self, x, C):
        o = torch.sigmoid(self.W_o(x))
        h = o * torch.tanh(C)
        return h  # h_t

    def forget_gate(self, x):
        f = torch.sigmoid(self.W_f(x))
        return f  # f_t


class LSTM(nn.Module):
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
        C_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        embedded = self.embedding(x)  # (batch, seq_len, hidden_size)

        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch, hidden_size)
            combined = torch.cat((h_t, x_t), dim=1)  # (batch, hidden+hidden)

            # Forget gate
            f_t = self.gate.forget_gate(combined)

            # Input gate & candidate cell
            i_t, C_hat_t = self.gate.input_gate(combined)

            # Cell state update
            C_t = f_t * C_t + i_t * C_hat_t

            # Output gate & hidden state
            h_t = self.gate.output_gate(combined, C_t)

        out = self.fc(h_t)
        return out
