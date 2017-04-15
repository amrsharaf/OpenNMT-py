# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from sklearn import datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(input_size, 1)
#        self.lin1 = nn.Linear(input_size, input_size * 4)
        self.lin2 = nn.Linear(input_size * 4, input_size * 4)
        self.lin3 = nn.Linear(input_size * 4, 1)

    def forward(self, x, training):
        output = self.lin1(x)
 #       output = F.relu(output)
        # TODO: Dropout test / train?
#        output = F.dropout(output, p=0.5, training=training)
#        output = self.lin2(output)
#        output = F.relu(output)
#        output = F.dropout(output, p=0.5, training=training)
#        output = self.lin3(output)
        output = F.sigmoid(output)
        return output


#iris = datasets.load_iris()
#inpt = Variable(torch.from_numpy(iris.data))
#inpt = inpt.float()
#target = iris.target
#input_size = inpt.size()[1]
#discriminator = Discriminator(input_size)
#criterion = nn.BCELoss()
#output = discriminator(inpt, training=True)
#print 'output: ', output
#print 'target: ', target
#target[target == 2] = 1
#target = Variable(torch.from_numpy(target), requires_grad=False)
#target = target.float()
#print 'target: ', target
#loss = criterion(output, target)
#print 'loss: ', loss
#
#optimizer = optim.SGD(discriminator.parameters(), lr = 0.01)
## in your training loop:
#for i in range(10000):
#    optimizer.zero_grad() # zero the gradient buffers
#    output = discriminator(inpt, True)
#    loss = criterion(output, target)
#    print 'loss: ', loss
#    loss.backward()
#    optimizer.step() # Does the update