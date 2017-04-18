import torch

class ReverseGradient(torch.autograd.Function):
    def __inti__(self):
        super(ReverseGradient, self).__init__()
    def forward(self, x):
        return x
    def backward(self, x):
#        return -x
        return x
