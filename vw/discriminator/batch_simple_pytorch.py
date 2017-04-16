from onmt.modules.discriminator import Discriminator
import onmt
import torch.nn as nn
from argparse import ArgumentParser
import torch
from preprocess_vw import text_to_vw
from preprocess_vw import process_sentences
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda
from random import shuffle

dtype = torch.FloatTensor


def parse_arguments():
    ap = ArgumentParser()
    ap.add_argument('--data', '-d', required=True)
    # TODO: refactor this part
    ap.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
    ap.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
    ap.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
    ap.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
    ap.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
    ap.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
    ap.add_argument('-adapt', action='store_true',
                    help='Domain Adaptation')
    
    # GPU
    ap.add_argument('-gpus', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")

    ap.add_argument('-log_interval', type=int, default=50,
                        help="Print stats at this interval.")
    return ap.parse_args()

def load_text_data(data, source, opt):
    if len(opt.gpus) >= 1:
        encoded_data = data[source]['src']
        random.shuffle(encoded_data)
        text_data = map(lambda x:' '.join([str(i) for i in x]) , encoded_data)
    else:
        encoded_data = data[source]['src']
        random.shuffle(encoded_data)
        text_data = map(lambda x:' '.join([str(i) for i in x]) , encoded_data)
    return text_data, encoded_data

from functools import partial

def get_valid_accuracy(valid_old, valid_new, model, opt):
    shuffle(valid_old)
    shuffle(valid_new)
    min_len = min(len(valid_old), len(valid_new))
    valid_old = valid_old[:min_len]
    valid_new = valid_new[:min_len]
    sentence_to_variable_partial = partial(sentence_to_variable, opt=opt)
    valid_old = map(sentence_to_variable_partial, valid_old)
    valid_new = map(sentence_to_variable_partial, valid_new)
    correct = 0
    total   = 0
    for pos_example, neg_example in zip(valid_old, valid_new):
        # Positive example
        total += 2.0
        old_output, new_output = model(pos_example, neg_example)
        if old_output.data[0][0] >= 0.5:
            correct += 1.0
        if new_output.data[0][0] < 0.5:
            correct += 1.0
    return correct / total

def sentence_to_variable(sentence, opt=None):
    if len(opt.gpus) >= 1:
        return Variable(sentence.view(1, -1)).cuda()
    else:
        return Variable(sentence.view(1, -1))
        
class RecurrentModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RecurrentModel, self).__init__()
        self.embedding = nn.Embedding(50000, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.lin = nn.Linear(input_size * num_layers * 2,1)
         
    def forward(self, inpt):
        output = self.embedding(inpt)
        output = output.transpose(0,1)
        _, (h_n, _) = self.lstm(output)
        output = self.lin(h_n.view(1,-1))
        return F.sigmoid(output)
    
def learn_recurrent(old_domain_encoded, new_domain_encoded, opt, dicts, valid_old, valid_new):
    encoder = onmt.Models.Encoder(opt, dicts['src'])
    discriminator = Discriminator(opt.word_vec_size * opt.layers)
    model = onmt.Models.NMTModel(encoder, discriminator)

    if len(opt.gpus) >= 1:
        model.cuda()
    else:
        model.cpu()
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(100):
        print 'Epoch: ', epoch
        loss = 0.0
        correct = 0.0
        i = 0.0
        total = 0.0
        # Create datasets!
        is_cuda = len(opt.gpus) >= 1
        batch_size = 64
        old_domain_dataset = list_to_dataset(old_domain_encoded, batch_size, is_cuda)
        new_domain_dataset = list_to_dataset(new_domain_encoded, batch_size, is_cuda)
        # Length and stuff!
        num_old_batches = len(old_domain_dataset)
        num_new_batches = len(new_domain_dataset)
        min_num_batches = min(num_old_batches, num_new_batches)
        old_indicies = range(num_old_batches)
        new_indicies = range(num_new_batches)
        # Shuffle
        random.shuffle(old_indicies)
        random.shuffle(new_indicies)
        # Truncate
        old_indicies = old_indicies[:min_num_batches]
        new_indicies = new_indicies[:min_num_batches]
        
        # Now zip and loop
        for old_id, new_id in zip(old_indicies, new_indicies):
            old_batch = old_domain_dataset[old_id][0]
            new_batch = new_domain_dataset[new_id][0]
            # batch first!
            old_batch = old_batch.transpose(0, 1)
            new_batch = new_batch.transpose(0, 1)
            # TODO Check the right alignment thing!
            # Forward prop!
            # Maybe we'll need a batch to variable thing
            old_output, new_output = model(old_batch, new_batch)
            print 'type of old output: ', old_output
            print 'type of new output: ', new_output

            exit(0)
            
        
        
           
        for pos_example, neg_example in zip(old_domain_encoded, new_domain_encoded):
            
            # Positive example
            #print total
            total+=1.0
            i += 1.0
            old_output, new_output = model(sentence_to_variable(pos_example, opt), sentence_to_variable(neg_example, opt))

            
            if old_output.data[0][0] >= 0.5:
                correct += 1
                
            if is_cuda:
                tgts_old =  Variable(torch.Tensor([1.0])).cuda()    
            else:
                tgts_old =  Variable(torch.Tensor([1.0]))
                
            loss += criterion(old_output,tgts_old)
            # Negative example
            total += 1.0
            if new_output.data[0][0] < 0.5:
                correct += 1.0
            if total % 100 == 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print 'iter:', i, ' | accuracy: ', correct / total
                loss = 0.0
                
            if is_cuda:
                tgts_new =  Variable(torch.Tensor([0.0])).cuda()    
            else:
                tgts_new =  Variable(torch.Tensor([0.0]))
            
            loss += criterion(new_output, tgts_new)
        # Done with this epoch, do evaluation
        valid_accuracy = get_valid_accuracy(valid_old, valid_new, model, opt)  
        print '\n\nValidation Accuracy: ', valid_accuracy
        print 'total: ', total, ' correct: ', correct
        print 'accuracy: ', correct / total, '\n\n'
        
def lookup_src(x, dicts=None):
    return dicts['src'].idxToLabel[x]

def convert_single_sentence(single_sentence, dicts=None):
    lookup_partial = partial(lookup_src, dicts=dicts)
    return str(" ".join(map(lookup_partial, single_sentence)))

def convert_to_sentence(list_of_sentences,dicts):
    convert_partial = partial(convert_single_sentence, dicts=dicts) 
    return map(convert_partial, list_of_sentences)

def tensor_to_vw_text(source_key, data, dicts, args):
    _, domain_encoded = load_text_data(data, source_key, args)
    domain_txt = convert_to_sentence(domain_encoded, dicts)
    domain_txt = map(text_to_vw, domain_txt)
    return domain_txt, domain_encoded


import math
def list_to_dataset(lst, batch_size, is_cuda):
    print 'list length: ', len(lst)
    print 'batch size: ', batch_size
    print 'then number of batches: ', math.ceil(len(lst)/float(batch_size))
    return onmt.Dataset(lst, None, batch_size, is_cuda)
    
import numpy as np


def count_unique(lst):
    hash_set = set(lst)
    return len(hash_set)


def main():
    random.seed(1234)
    args = parse_arguments()
    
    if torch.cuda.is_available() and not args.gpus:
        print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

    if args.gpus:
        cuda.set_device(args.gpus[0])
    
    print 'Reading data from: ', args.data
    data = torch.load(args.data)
    dicts = data['dicts']
    
    train_old_domain_txt, train_old_domain_encoded = tensor_to_vw_text('train', data, dicts, args) 
    train_new_domain_txt, train_new_domain_encoded = tensor_to_vw_text('domain_train', data, dicts, args) 

    valid_old_domain_txt, valid_old_domain_encoded = tensor_to_vw_text('valid', data, dicts, args) 
    valid_new_domain_txt, valid_new_domain_encoded = tensor_to_vw_text('domain_valid', data, dicts, args) 

    test_old_domain_txt, _ = tensor_to_vw_text('test', data, dicts, args) 
    test_new_domain_txt, _ = tensor_to_vw_text('domain_test', data, dicts, args)
    
#    train_new_domain_encoded_len = map(count_unique, train_new_domain_encoded)
#    train_new_domain_encoded_hist = np.histogram(train_new_domain_encoded_len, bins=50)
#    print train_new_domain_encoded_hist
#    exit(0)
    
    
    # Removing vw special characters!
    # TODO create functions for these stuff
    
    #print 'Generating vw files...'
    #process_sentences(train_old_domain_txt, train_new_domain_txt, 'data/train.vw')
    #process_sentences(valid_old_domain_txt, valid_new_domain_txt, 'data/valid.vw')
    #process_sentences(test_old_domain_txt, test_new_domain_txt, 'data/test.vw')
    #print 'done!'
    #print 'Generating vw files...'
    #process_sentences(train_old_domain_encoded_txt, train_new_domain_encoded_txt, 'data/train_encoded.vw')
    #process_sentences(valid_old_domain_encoded_txt, valid_new_domain_encoded_txt, 'data/valid_encoded.vw')
    
    learn_recurrent(train_old_domain_encoded, train_new_domain_encoded, args, data['dicts'], valid_old_domain_encoded, valid_new_domain_encoded)
#    learn_lstm(train_old_domain_encoded, train_new_domain_encoded, args, data['dicts'], valid_old_domain_encoded, valid_new_domain_encoded)
    
if __name__ == '__main__':
    main()
    
    