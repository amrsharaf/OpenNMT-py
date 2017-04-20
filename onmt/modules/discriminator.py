import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(input_size, 1)

    def forward(self, x):
        output = self.lin1(x)
        output = F.sigmoid(output)
        return output
