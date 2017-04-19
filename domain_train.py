from __future__ import division

import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
from onmt.modules.discriminator import Discriminator
from onmt.modules.reverse_gradient import ReverseGradient

parser = argparse.ArgumentParser(description='train.py')

## Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
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
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")

#learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")

#pretrained word vectors

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
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")

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
    return prediction.eq(truth).sum(), float(prediction.nelement())

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

            model.zero_grad()
            if opt.adapt:
                batchIdxAdapt = batchOrderAdapt[i] if epoch >= opt.curriculum else i
                batch_len = len(batch[0][1])
                domain_batch = domain_train[batchIdxAdapt][:-1]

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
                discriminator_loss.backward(retain_variables=True)
                model.encoder.zero_grad()
                model.decoder.zero_grad()
                model.discriminator.zero_grad()
                outputs.backward(gradOutput)
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
            if opt.adapt:
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

                if opt.adapt:
                    print "discrim_correct: ", discrim_correct
                    print "num_discrim_elements: ", num_discrim_elements, '\n'

                report_loss = report_tgt_words = report_src_words = report_num_correct = 0
                start = time.time()

        if opt.adapt:
            return total_loss / total_words, total_num_correct / total_words, total_num_discrim_correct / total_num_discrim_elements
        else:
            return total_loss / total_words, total_num_correct / total_words, 0

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss, train_acc, train_discrim_acc = trainEpoch(epoch)
        print('Train perplexity: %g' % math.exp(min(train_loss, 100)))
        print('Train accuracy: %g' % (train_acc*100))
        print('Train discriminator accuracy: %g' % (train_discrim_acc * 100))

        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(model, criterion, validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        print('Validation accuracy: %g' % (valid_acc*100))
        if opt.adapt:
            valid_discrim_acc = domain_eval(model, validData, domain_valid)
            print('Validation discriminator accuracy: %g' % (valid_discrim_acc * 100))
        #  (3) update the learning rate
        optim.updateLearningRate(valid_loss, epoch)

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

    dict_checkpoint = opt.train_from if opt.train_from else opt.train_from_state_dict
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    # type(dataset) = <type 'dict'>
    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.cuda)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.cuda, volatile=True)

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

    encoder = onmt.DomainModels.Encoder(opt, dicts['src'])
    decoder = onmt.DomainModels.Decoder(opt, dicts['tgt'])

    generator = nn.Sequential(
        nn.Linear(opt.rnn_size, dicts['tgt'].size()),
        nn.LogSoftmax())

    discriminator = None
    if opt.adapt:
        discriminator = Discriminator(opt.word_vec_size  * opt.layers)

    model = onmt.DomainModels.NMTModel(encoder, decoder, discriminator)

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items() if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        print('Loading model from checkpoint at %s' % opt.train_from_state_dict)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator

    if not opt.train_from_state_dict and not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        encoder.load_pretrained_vectors(opt)
        decoder.load_pretrained_vectors(opt)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(model, trainData, validData, domain_train, domain_valid, dataset, optim)

if __name__ == "__main__":
    main()
