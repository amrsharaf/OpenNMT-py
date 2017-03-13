import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(input_size, input_size * 4)
        self.lin2 = nn.Linear(input_size * 4, input_size * 4)
        self.lin3 = nn.Linear(input_size * 4, 1)

    def forward(self, x, training):
        output = self.lin1(x)
        output = F.relu(output)
        # TODO: Dropout test / train?
        output = F.dropout(output, p=0.5, training=training)
        output = self.lin2(output)
        output = F.relu(output)
        output = F.dropout(output, p=0.5, training=training)
        output = self.lin3(output)
        output = F.sigmoid(output)
        return output
