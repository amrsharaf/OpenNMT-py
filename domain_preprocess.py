import onmt

import argparse
import torch

parser = argparse.ArgumentParser(description='preprocess.lua')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")
parser.add_argument('-sort_new_domain', help="Sort new domain", default=True)

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                     help="Path to the validation target data")
parser.add_argument('-test_src', required=False,
                    help="Path to the test source data")
parser.add_argument('-test_tgt', required=False,
                     help="Path to the test target data")
# For domain adaptation
parser.add_argument('-domain_train_src', required=False,
                     help="Path to in-domain source data")
parser.add_argument('-domain_valid_src', required=False,
                     help="Path to in-domain source data")
parser.add_argument('-domain_test_src', required=False,
                     help="Path to in-domain test source data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")


parser.add_argument('-src_seq_length', type=int, default=50,
                    help="Maximum source sequence length")
parser.add_argument('-tgt_seq_length', type=int, default=50,
                    help="Maximum target sequence length")
parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()


def makeVocabulary(filename, size):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD])

    with open(filename) as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab

def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts, sort_by_len=True):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile)
    tgtF = open(tgtFile)

    while True:
        srcWords = srcF.readline()
        tgtWords = tgtF.readline()

        if not srcWords or not tgtWords:
            break

        srcWords = srcWords.split()
        tgtWords = tgtWords.split()

        if (len(srcWords) > 0) and (len(tgtWords) > 0) and  len(srcWords) <= opt.src_seq_length and len(tgtWords) <= opt.tgt_seq_length:

            src += [srcDicts.convertToIdx(srcWords,
                                          onmt.Constants.UNK_WORD)]
            tgt += [tgtDicts.convertToIdx(tgtWords,
                                          onmt.Constants.UNK_WORD,
                                          onmt.Constants.BOS_WORD,
                                          onmt.Constants.EOS_WORD)]

            sizes += [len(srcWords)]
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    if sort_by_len:
        print('... sorting sentences by size')
        _, perm = torch.sort(torch.Tensor(sizes))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or source length > %d or target length > %d)' %
          (len(src), ignored, opt.src_seq_length, opt.tgt_seq_length))

    return src, tgt


def is_not_unk(value):
    return value != onmt.Constants.UNK

def filter_unk_tensor(tensor):
    return torch.LongTensor(filter(is_not_unk, tensor))

def filter_unk(tensor_list):
    filtered_tensor_list = map(filter_unk_tensor, tensor_list)
    return filter(lambda x: len(x) != 0, filtered_tensor_list)

def main():

    dicts = {}
    # type(dicts['src']) = <class 'onmt.Dict.Dict'>
    dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                  opt.src_vocab_size)
    # type(dicts['tgt']) = <class 'onmt.Dict.Dict'>
    dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                  opt.tgt_vocab_size)

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeData(opt.train_src, opt.train_tgt,
                                          dicts['src'], dicts['tgt'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = makeData(opt.valid_src, opt.valid_tgt,
                                    dicts['src'], dicts['tgt'])

    test = {}
    if opt.test_src:
        assert(opt.test_tgt)
        print('Preparing testing ...')
        test['src'], test['tgt'] = makeData(opt.test_src, opt.test_tgt,
                                    dicts['src'], dicts['tgt'])
    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')


    print('Saving data to \'' + opt.save_data + '-train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid,
                 'test': test}

    # domain adaptation source data
    domain_train = {}
    domain_valid = {}
    domain_test  = {}
    if opt.domain_train_src is not None:
        # Tra vbin

        domain_train['src'], _ = makeData(opt.domain_train_src, opt.domain_train_src,
                                       dicts['src'], dicts['tgt'], opt.sort_new_domain)


        domain_train['src'] = filter_unk(domain_train['src'])

        # Validation data
        domain_valid['src'], _ = makeData(opt.domain_valid_src, opt.domain_valid_src,
                                       dicts['src'], dicts['tgt'])

        # Test data
        domain_test = {}
        if opt.domain_test_src:
            domain_test['src'], _ = makeData(opt.domain_test_src, opt.domain_test_src,
                                       dicts['src'], dicts['tgt'])

        # saving!
        save_data['domain_train'] = domain_train
        save_data['domain_valid'] = domain_valid
        save_data['domain_test'] = domain_test


    torch.save(save_data, opt.save_data + '-train.pt')

if __name__ == "__main__":
    main()
