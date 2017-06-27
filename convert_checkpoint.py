import argparse
import torch
import onmt
import codecs

parser = argparse.ArgumentParser(description="convert_checkpoint.py")

parser.add_argument("-model", type=str, required=True)
parser.add_argument("-src_dict", type=str, default=None)
parser.add_argument("-tgt_dict", type=str, default=None)
parser.add_argument("-src_bpe", type=str, default=None)
parser.add_argument("-tgt_bpe", type=str, default=None)
parser.add_argument("-new_model", type=str, required=True)

args = parser.parse_args()

ckp = torch.load(args.model)
model = dict()
for k, v in ckp['model'].items():
    if not k[0] == '0':
        model[k] = v
model['generator.generator.bias'] = ckp['generator']['0.bias']
model['generator.generator.weight'] = ckp['generator']['0.weight']

dicts = bpes = None

if args.src_dict is not None:
    src_dict = onmt.Dict()
    src_dict.loadFile(args.src_dict)
    tgt_dict = onmt.Dict()
    tgt_dict.loadFile(args.tgt_dict)
    dicts = {'src': src_dict, 'tgt': tgt_dict}

if args.src_bpe is not None:
    from bpe import BPE
    src_bpe = BPE(codecs.open(args.src_bpe))
    tgt_bpe = BPE(codecs.open(args.tgt_bpe))
    bpes = {'src': src_bpe, 'tgt': tgt_bpe}

opt = ckp['opt']
opt.cuda = True
opt.dropout = 0
opt.lr = 1e-4

epoch = ckp["epoch"]

new = {'model': model, "epoch": epoch, 'opt': opt, 'dicts': dicts, "bpe": bpes}
torch.save(new, args.new_model)
