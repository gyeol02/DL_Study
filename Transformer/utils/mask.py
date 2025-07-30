import torch

def create_padding_mask(seq, pad_token=0):
    """
    Creates a mask to ignore padding tokens in the input sequence
    """
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, seq_len)

def create_look_ahead_mask(size):
    """
    Creates a mask to prevent attention to future tokens (used in decoder)
    """
    return torch.tril(torch.ones(size, size)).bool()  # (tgt_len, tgt_len)

def combine_masks(padding_mask, look_ahead_mask):
    """
    Combines padding mask and look-ahead mask into a single decoder mask
    """
    return padding_mask & look_ahead_mask