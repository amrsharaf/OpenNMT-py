import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import torch.optim as optimizer
from onmt.modules.discriminator import Discriminator

parser = argparse.ArgumentParser(description='train.py')

## Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from',
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

## Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")

## Optimization options

parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.1""")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""Decay learning rate by this much if (i) perplexity
                    does not decrease on the validation set or (ii) epoch has
                    gone past the start_decay_at_limit""")
parser.add_argument('-start_decay_at', default=8,
                    help="Start decay after this epoch")
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")
# parser.add_argument('-seed', type=int, default=3435,
#                     help="Seed for random initialization")

# domain adaptation
parser.add_argument('-adapt', action='store_true',
                    help='Domain Adaptation')

opt = parser.parse_args()
opt.cuda = len(opt.gpus)

print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with -cuda")

if opt.cuda:
    cuda.set_device(opt.gpus[0])

def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.cuda:
        crit.cuda()
    return crit


def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = generator(out_t)
        loss_t = crit(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(onmt.Constants.PAD).data).sum()
        num_correct += num_correct_t
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, num_correct


def eval(model, criterion, data):
    total_loss = 0
    total_words = 0
    total_num_correct = 0

    model.eval()
    for i in range(len(data)):
        batch = data[i][:-1] # exclude original indices
        outputs = model(batch)
        targets = batch[1][1:]  # exclude <s> from targets
        loss, _, num_correct = memoryEfficientLoss(
                outputs, targets, model.generator, criterion, eval=True)
        total_loss += loss
        total_num_correct += num_correct
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    return float(total_loss) / float(total_words),\
           float(total_num_correct) / float(total_words)

def accuracy_eval(model, data_old, data_new):
    model.eval()
    for i in range(min(len(data_old),len(data_new))):
        batch_old = [x.transpose(0, 1) for x in data_old[i]]
        batch_new = data_new[i][0].transpose(0, 1)
        
        _,old_domain, new_domain = model(batch_old,batch_new)   
        tgts = Variable(torch.FloatTensor(len(old_domain) + len(new_domain),), requires_grad=False)   
        if opt.cuda:
            tgts = tgts.cuda()
        
        tgts[:] = 0.0
        tgts[:len(old_domain)] = 1.0
        accuracy = get_accuracy(torch.cat([old_domain, new_domain]).data.squeeze(), tgts.data)  
        print "batch_old: ", batch_old[0].size()
        print "batch_new: ", batch_new.size()
        print 'valid accuracy: ', accuracy, '\n'
    return accuracy

def domain_eval(model, data_old, data_new):
    model.eval()
    accuracy = 0
    total_num_discrim_correct, total_num_discrim_elements = 0, 0
    for i in range(min(len(data_new),len(data_old))):
        batch_old = data_old[i][:-1] # exclude original indices
        batch_new = data_new[i][:-1]
        
        _, old_domain, new_domain = model(batch_old, domain_batch=batch_new)  
        
        tgts = Variable(torch.FloatTensor(len(old_domain) + len(new_domain),), requires_grad=False) 
            
        if opt.cuda:
            tgts = tgts.cuda()

        tgts[:] = 0.0
        tgts[:len(old_domain)] = 1.0
        discrim_correct, num_discrim_elements = get_accuracy(torch.cat([old_domain, new_domain]).data.squeeze(), tgts.data)
        
        # Discriminator counts
        total_num_discrim_correct += discrim_correct
        total_num_discrim_elements += num_discrim_elements
        
    return float(total_num_discrim_correct) / float(total_num_discrim_elements)


def get_accuracy(prediction, truth):
    assert(prediction.nelement() == truth.nelement())
    prediction[prediction < 0.5]  = 0.0
    prediction[prediction >= 0.5] = 1.0
    accuracy = (100.0 * prediction.eq(truth).sum()) / float(prediction.nelement())
    return accuracy
    
def trainModel(model, trainData, validData, domain_train, domain_valid, dataset, optim):
    print(model)
    model.train()

    # define criterion of each GPU
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())

    start_time = time.time()
    def trainEpoch(epoch):
        
        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))
        
        discriminator_criterion = None
        if opt.adapt:
            batchOrderAdapt = torch.randperm(len(domain_train))
            discriminator_criterion = nn.BCELoss()

        total_num_discrim_correct, total_num_discrim_elements = 0, 0
        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
        start = time.time()
        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i   
            batch = trainData[batchIdx][:-1] # exclude original indices
            
            if debug:
                print "trainData batch size: ", trainData.batchSize
                print "batch len : ", len(batch)
                
                print "\n\nbacth size: ", (len(batch[0][1]))
                print "other ---->: ", batch[0][1]
                print "batchIdx: ",batchIdx

            model.zero_grad()
            if opt.adapt:
                batchIdxAdapt = batchOrderAdapt[i] if epoch >= opt.curriculum else i
                batch_len = len(batch[0][1])
                domain_batch = domain_train[batchIdxAdapt][:-1]
                
                if debug:
                    print "domain_train batch size: ", domain_train.batchSize
                    print "domain_batch[0] type: ", type(domain_batch[0])
                    print "domain_batch[0][0] type: ", type(domain_batch[0][0]), '\n'

                outputs, old_domain, new_domain = model(batch, domain_batch=domain_batch)       
                discriminator_targets = Variable(torch.FloatTensor(len(old_domain) + len(new_domain),), requires_grad=False)

                if opt.cuda:
                    discriminator_targets = discriminator_targets.cuda()
                
                discriminator_targets[:] = 0.0
                discriminator_targets[:len(old_domain)] = 1.0
                discrim_correct, num_discrim_elements = get_accuracy(torch.cat([old_domain, new_domain]).data.squeeze(), discriminator_targets.data)
                
                discriminator_loss = discriminator_criterion(torch.cat([old_domain, new_domain]), discriminator_targets)
            else:
                outputs = model(batch)

            targets = batch[1][1:]  # exclude <s> from targets
            loss, gradOutput, num_correct = memoryEfficientLoss(
                    outputs, targets, model.generator, criterion)

            # We do the domain adaptation backward call here
            if opt.adapt:
                outputs.backward(gradOutput, retain_variables=True)
                model.zero_grad()
                discriminator_loss.backward()
            else:
                outputs.backward(gradOutput)
    

            # update the parameters
            optim.step()

            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            report_loss += loss
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += sum(batch[0][1])
            total_loss += loss
            total_num_correct += num_correct
            total_words += num_words
            
            # Discriminator counts
            total_num_discrim_correct += discrim_correct
            total_num_discrim_elements += num_discrim_elements
            
            if i % opt.log_interval == -1 % opt.log_interval:
                print("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed" %
                      (epoch, i+1, len(trainData),
                      report_num_correct / report_tgt_words * 100,
                      math.exp(report_loss / report_tgt_words),
                      report_src_words/(time.time()-start),
                      report_tgt_words/(time.time()-start),
                      time.time()-start_time))
                
                print "discrim_correct: ", discrim_correct
                print "num_discrim_elements: ", num_discrim_elements, '\n'

                report_loss = report_tgt_words = report_src_words = report_num_correct = 0
                start = time.time()

        return float(total_loss) / float(total_words),\
               float(total_num_correct) / float(total_words),\
               float(total_num_discrim_correct) / float(total_num_discrim_elements)

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss, train_acc, train_discrim_acc = trainEpoch(epoch)
        print('Train perplexity: %g' % math.exp(min(train_loss, 100)))
        print('Train accuracy: %g' % (train_acc*100))
        print('Train discriminator accuracy: %g' % (train_discrim_acc * 100))

        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(model, criterion, validData)
        valid_discrim_acc = domain_eval(model, validData, domain_valid)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        print('Validation accuracy: %g' % (valid_acc*100))
        print('Validation discriminator accuracy: %g' % (valid_discrim_acc * 100))

        
        #  (3) update the learning rate
        # optim.updateLearningRate(valid_loss, epoch)

        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(opt.gpus) > 1 else model.generator.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, 100*valid_acc, valid_ppl, epoch))

def main():

    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)

    # type(dataset) = <type 'dict'>
    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.cuda)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.cuda)

    domain_train = None
    domain_valid = None
    if opt.adapt:
        assert('domain_train' in dataset)
        assert('domain_valid' in dataset)
        domain_train = onmt.Dataset(dataset['domain_train']['src'], None, 
                                  opt.batch_size, opt.cuda)
        domain_valid = onmt.Dataset(dataset['domain_valid']['src'], None, 
                                  opt.batch_size, opt.cuda)
        
        
    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    if opt.train_from is None:
        encoder = onmt.Models.Encoder(opt, dicts['src'])
        decoder = onmt.Models.Decoder(opt, dicts['tgt'])
        generator = nn.Sequential(
            nn.Linear(opt.rnn_size, dicts['tgt'].size()),
            nn.LogSoftmax())
        if opt.cuda > 1:
            generator = nn.DataParallel(generator, device_ids=opt.gpus)
        # Domain Adaptation
        discriminator = None
        if opt.adapt:
            discriminator = Discriminator(opt.word_vec_size  * opt.layers)
        model = onmt.Models.NMTModel(encoder, decoder, generator, discriminator)
        if opt.cuda > 1:
            model = nn.DataParallel(model, device_ids=opt.gpus)
        if opt.cuda:
            model.cuda()
        else:
            model.cpu()

        model.generator = generator

        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        optim = onmt.Optim(
            model.parameters(), opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
        #optim = optimizer.SGD(model.parameters(), lr=0.01)
    else:
        print('Loading from checkpoint at %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from)
        model = checkpoint['model']
        if opt.cuda:
            model.cuda()
        else:
            model.cpu()
        optim = checkpoint['optim']
        opt.start_epoch = checkpoint['epoch'] + 1
        
    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict()

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(model, trainData, validData, domain_train, domain_valid, dataset, optim)


if __name__ == "__main__":
    main()
