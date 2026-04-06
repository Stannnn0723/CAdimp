import torch
import torch.nn.functional as F

def softmax_reg(x: torch.Tensor, dim, reg=None):
    if reg is None:
        return torch.softmax(x, dim=dim)
    dim %= x.dim()
    if isinstance(reg, (float, int)):
        reg = x.new_tensor([reg])
    reg = reg.expand([1 if d==dim else x.shape[d] for d in range(x.dim())])
    x = torch.cat((x, reg), dim=dim)
    return torch.softmax(x, dim=dim)[[slice(-1) if d==dim else slice(None) for d in range(x.dim())]]


scores = torch.rand(35, 1, 23, 23)
scores_view = scores.view(scores.shape[0], -1)
# reg_val is None or not None?
reg_val = 0.05
scores_softmax = softmax_reg(scores_view, dim=-1, reg=reg_val)
print("scores_softmax shape:", scores_softmax.shape)
scores = scores_softmax.view(scores.shape)
print("scores shape:", scores.shape)
