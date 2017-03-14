import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules

class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        inputSize = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(inputSize, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

        # self.rnn.bias_ih_l0.data.div_(2)
        # self.rnn.bias_hh_l0.data.copy_(self.rnn.bias_ih_l0.data)

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.copy_(pretrained)

    def forward(self, inpt, hidden=None):
        batch_size = inpt.size(0) # batch first for multi-gpu compatibility
        emb = self.word_lut(inpt).transpose(0, 1)
        if hidden is None:
            h_size = (self.layers * self.num_directions, batch_size, self.hidden_size)
            h_0 = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)
            c_0 = Variable(emb.data.new(*h_size).zero_(), requires_grad=False)
            hidden = (h_0, c_0)

        outputs, hidden_t = self.rnn(emb, hidden)
        return hidden_t, outputs

class NMTModel(nn.Module):

    def __init__(self, encoder, discriminator):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        # TODO Input to the discriminator should be similar to the generator
        self.discriminator = discriminator

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h
        
    # type(inpt) = list
    # len(inpt) = 2
    # type(inpt[0]) = <class 'torch.autograd.variable.Variable'>
    # type(inpt[1]) = <class 'torch.autograd.variable.Variable'>
    # inpt[0].size() = (batch_size, )
    # inpt[1].size() = (batch_size, )
    # type(enc_hidden) = tuple
    # len(enc_hidden)  = 2
    # type(enc_hidden[0]) = type(enc_hidden[1]) = Variable
    # enc_hidden[0].size() = enc_hidden[1].size() = (layers, batch_size, dim)
    # type(context) = Variable
    # context.size() = (words, batch_size, dim)
    def forward(self, inpt, domainBatch=None):
        src = inpt[0]
        batch_size = src.size()[0]
        enc_hidden, context = self.encoder(src)
        assert(context[-1,0,:10] == enc_hidden[0][-1, 0,:10])
        # Domain adaptation
        old_domain = None
        new_domain = None
        if domainBatch is not None:
            enc_hidden_adapt, _ = self.encoder(domainBatch[0])
            # TODO: training flag
            old_domain = self.discriminator(enc_hidden[1].transpose(0,1).contiguous().view(batch_size,-1), True)
            # This should give a label of 0
            new_domain = self.discriminator(enc_hidden_adapt[1].transpose(0,1).contiguous().view(batch_size,-1), True)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))

        return old_domain, new_domain
#        return out, old_domain, new_domain
