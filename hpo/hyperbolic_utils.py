import torch

def artanh(x):
    return Artanh.apply(x)

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)
    
def proj_tan0(u, c):
    return u

def logmap0(p, c):
    min_norm = 1e-15
    sqrt_c = c ** 0.5
    p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(min_norm)
    scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
    return scale * p

def euclidean_to_hyperbolic(data, c=1):
    return proj_tan0(logmap0(torch.Tensor(data), c=c), c=c).numpy()