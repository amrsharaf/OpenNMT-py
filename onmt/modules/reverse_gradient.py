import torch.nn as nn
import torch.nn.functional as F

class ReverseGradient(nn.Module):
    def __init__(self):
        super(ReverseGradient, self).__init__()

    #def forward(self, x):
    #    return x
    #def backward(self, x):
    #    return -x

    def forward(self, input):
        self.output = input
        return self.output
