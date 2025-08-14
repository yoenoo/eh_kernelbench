import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class KLLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        return torch.sum(targets * (torch.log(targets) - torch.log(predictions)), dim=-1).mean()

    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        grad_pred = grad_output * (targets / predictions - torch.sum(targets - predictions, dim=-1, keepdim=True)) / predictions.shape[0]
        return grad_pred, None

def custom_kl_div(predictions, targets, reduction='batchmean'):
    return KLLossFunction.apply(predictions, targets)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, predictions, targets):
        return custom_kl_div(predictions, targets, reduction='batchmean')

batch_size = 8192 * 2
input_shape = (8192 * 2,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [(torch.rand(batch_size, *input_shape)*scale).softmax(dim=-1), torch.rand(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []