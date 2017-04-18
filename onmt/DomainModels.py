import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, inpt, hidden=None):
        if isinstance(inpt, tuple):
            emb = pack(self.word_lut(inpt[0]), inpt[1])
        else:
            emb = self.word_lut(inpt)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(inpt, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, inpt, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(inpt, (h_0[i], c_0[i]))
            inpt = h_1_i
            if i + 1 != self.num_layers:
                inpt = self.dropout(inpt)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return inpt, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, inpt, hidden, context, init_output):
        emb = self.word_lut(inpt)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, attn = self.attn(output, context.t())
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, discriminator=None, reverse_gradient=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.reverse_gradient = reverse_gradient

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

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
    # enc_hidden[0]: hidden unit states for every word
    # enc_hidden[1]: output of the encoder 
    # type(enc_hidden[0]) = type(enc_hidden[1]) = Variable
    # enc_hidden[0].size() = enc_hidden[1].size() = (layers, batch_size, dim)
    # type(context) = Variable
    # context.size() = (words, batch_size, dim)
    def forward(self, inpt, domain_batch=None):
        src = inpt[0]
        tgt = inpt[1][:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src)
        init_output = self.make_init_decoder_output(context)
        
        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))
        
        out, dec_hidden, _attn = self.decoder(tgt, enc_hidden, context, init_output)
        
        debug = False
        
        if domain_batch is not None:
            # TODO: make sure this is the correct batch_size
            old_batch_size = context.size(1) 
            src_domain_batch = domain_batch[0]
            
            enc_hidden_adapt,context = self.encoder(src_domain_batch)
            new_batch_size = context.size(1)
            enc_hidden_adapt  = (self._fix_enc_hidden(enc_hidden_adapt[0]),
                                 self._fix_enc_hidden(enc_hidden_adapt[1]))
            
            if debug:
                print "enc_hidden_adapt.size(): ", enc_hidden_adapt[1].size()
                print "enc_hidden_adapt: ", enc_hidden_adapt[1].transpose(0,1).contiguous().view(new_batch_size,-1).size()
                
                print "enc_hidden.size(): ", enc_hidden[1].size(), '\n'
                print "enc_hidden: ", enc_hidden[1].transpose(0,1).contiguous().view(old_batch_size,-1).size(), '\n'

            
            # TODO: training flag, and maybe concatenate the two batches!?
            old_domain = self.reverse_gradient(self.discriminator(enc_hidden[1].transpose(0,1).contiguous().view(old_batch_size,-1), True))
            
            # This should give a label of 0
            new_domain = self.reverse_gradient(self.discriminator(enc_hidden_adapt[1].transpose(0,1).contiguous().view(new_batch_size,-1), True))
    
            return out, old_domain, new_domain

        # if not domain_batch
        return out
