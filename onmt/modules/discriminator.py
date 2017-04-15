import torch.nn as nn
import torch.nn.functional as F

#self.linear1 = torch.nn.Linear(D_in, H)
#self.linear2 = torch.nn.Linear(H, D_out)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(input_size, input_size * 4)
        self.drop = nn.Dropout(p=0.95)
        self.lin2 = nn.Linear(input_size * 4, input_size * 4)
        self.lin3 = nn.Linear(input_size * 4, 1)

    def forward(self, x, training):
        # return F.sigmoid(self.lin1(x))
        # return F.sigmoid(self.drop(self.lin1(x)))
        output = F.sigmoid(self.lin1(x))
        #TODO: Dropout test / train?
        output = F.dropout(output, p=0.95)
        output = self.lin2(output)
        output = F.sigmoid(output)
        output = F.dropout(output, p=0.95)
        output = self.lin3(output)
        output = F.sigmoid(output)
        return output
