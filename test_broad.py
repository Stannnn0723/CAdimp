import torch

num_sequences = 1
shield_active = torch.zeros(num_sequences, dtype=torch.bool)
H_diag_ema = torch.zeros(num_sequences, 64, 22, 22)

try:
    print('Testing 1 sequence:')
    x = shield_active.unsqueeze(-1).unsqueeze(-1).expand_as(H_diag_ema[:1])
    print('Shape 1:', x.shape)
except Exception as e:
    print('Error 1:', e)

num_sequences = 5
shield_active = torch.zeros(num_sequences, dtype=torch.bool)
H_diag_ema = torch.zeros(num_sequences, 64, 22, 22)

try:
    print('Testing 5 sequences:')
    x = shield_active.unsqueeze(-1).unsqueeze(-1).expand_as(H_diag_ema[:1])
    print('Shape 5:', x.shape)
except Exception as e:
    print('Error 5:', e)

