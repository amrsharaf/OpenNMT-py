import sys
import torch
from argparse import Namespace

'''
strip the checkpoint of dict, epoch and optim
convert cuda.tensors to tensors
'''

ckp_dir = '../../shifeng/60_percent.pt'
data_dir = 'data/amazon/all_60.bpe.train.pt'
ckp = torch.load(ckp_dir)

new_ckp = dict()
new_ckp['model'] = dict()
for k, v in ckp['model'].items():
    new_ckp['model'][k] = v.cpu()
new_ckp['model']['generator.generator.weight'] = ckp['generator']['0.weight'].cpu()
new_ckp['model']['generator.generator.bias'] = ckp['generator']['0.bias'].cpu()

o = ckp['opt']
opt = Namespace()
opt.data = data_dir
opt.batch_size = o.batch_size
opt.brnn = o.brnn
opt.brnn_merge = o.brnn_merge
opt.curriculum = o.curriculum
opt.dropout = o.dropout
opt.end_epoch = o.epochs
opt.gpus = o.gpus
opt.input_feed = o.input_feed
opt.layers = o.layers
opt.lr = 1e-3
opt.learning_rate_decay = o.learning_rate_decay
opt.log_interval = o.log_interval
opt.max_generator_batches = o.max_generator_batches
opt.max_grad_norm = o.max_grad_norm
opt.optim = o.optim
opt.param_init = o.param_init
opt.pre_word_vecs_dec = o.pre_word_vecs_dec
opt.pre_word_vecs_enc = o.pre_word_vecs_enc
opt.rnn_size = o.rnn_size
opt.save_dir= ckp_dir
opt.start_decay_at = o.start_decay_at
opt.start_epoch = o.start_epoch
opt.word_vec_size = o.word_vec_size

new_ckp['opt'] = opt

torch.save(new_ckp, ckp_dir + '.2rl')
