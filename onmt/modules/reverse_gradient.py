import torch


# Code in file autograd/two_layer_net_custom_function.py
import torch
from torch.autograd import Variable

class ReverseGradient(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  def forward(self, input):
    """
    In the forward pass we receive a Tensor containing the input and return a
    Tensor containing the output. You can cache arbitrary Tensors for use in the
    backward pass using the save_for_backward method.
    """
    self.save_for_backward(input)
    return input.clamp(min=0)

  def backward(self, grad_output):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    input, = self.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input < 0] = 0
    return grad_input



#class ReverseGradient(torch.nn.functional):
#    def __inti__(self):
#        super(ReverseGradient, self).__init__()
#    def forward(self, x):
#        return x
#    def backward(self, x):
#        print 'x: ', x.size()
##        return -x
#        return x
