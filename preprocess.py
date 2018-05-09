import os
import onmt
import argparse
import torch
import codecs


parser = argparse.ArgumentParser(description='preprocess.py')

# **Preprocess Options**

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-src_type', default="text",
                    help="Type of the source input. Options are [text|img].")
parser.add_argument('-src_img_dir', default=".",
                    help="Location of source images")

parser.add_argument('-data_dir', required=True,
                    help="Directory to the data")
parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', default="",
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', default="",
                    help="Path to the validation target data")
parser.add_argument('-test_src', required=True,
                    help="Path to the test source data")
parser.add_argument('-test_tgt', required=True,
                    help="Path to the test target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_min_freq', type=int, default=0,
                    help="Minimum frequency of the source vocabulary")
parser.add_argument('-tgt_min_freq', type=int, default=0,
                    help="Minimum frequency of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")

parser.add_argument('-src_seq_length', type=int, default=100,
                    help="Maximum source sequence length")
parser.add_argument('-src_seq_length_trunc', type=int, default=0,
                    help="Truncate source sequence length.")
parser.add_argument('-tgt_seq_length', type=int, default=100,
                    help="Maximum target sequence length to keep.")
parser.add_argument('-tgt_seq_length_trunc', type=int, default=0,
                    help="Truncate target sequence length.")

parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

opt.train_src = os.path.join(opt.data_dir, opt.train_src)
opt.train_tgt = os.path.join(opt.data_dir, opt.train_tgt)
if '' not in (opt.valid_src, opt.valid_tgt):
    opt.valid_src = os.path.join(opt.data_dir, opt.valid_src)
    opt.valid_tgt = os.path.join(opt.data_dir, opt.valid_tgt)
opt.test_src = os.path.join(opt.data_dir, opt.test_src)
opt.test_tgt = os.path.join(opt.data_dir, opt.test_tgt)

opt.save_data = os.path.join(opt.data_dir, opt.save_data)

torch.manual_seed(opt.seed)


def makeVocabulary(filename, size, min_freq):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)

    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune_by_freq(min_freq)
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize, min_freq):
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
        genWordVocab = makeVocabulary(dataFile, vocabSize, min_freq)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts, is_test_split=False):
    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = codecs.open(srcFile, 'r', encoding='utf-8')
    tgtF = codecs.open(tgtFile, 'r', encoding='utf-8')

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: src and tgt do not have the same # of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        srcWords = sline.split()
        tgtWords = tline.split()

        if (len(srcWords) <= opt.src_seq_length and len(tgtWords) <= opt.tgt_seq_length) or is_test_split:
            if not is_test_split:
                # Check truncation condition.
                if opt.src_seq_length_trunc != 0:
                    srcWords = srcWords[:opt.src_seq_length_trunc]
                if opt.tgt_seq_length_trunc != 0:
                    tgtWords = tgtWords[:opt.tgt_seq_length_trunc]

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

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print(('Prepared %d sentences ' +
           '(%d ignored due to length == 0 or src len > %d or tgt len > %d)') %
          (len(src), ignored, opt.src_seq_length, opt.tgt_seq_length))

    return src, tgt


def main():

    dicts = {}
    dicts['src'] = onmt.Dict()
    dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                  opt.src_vocab_size, opt.src_min_freq)

    dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                  opt.tgt_vocab_size, opt.tgt_min_freq)

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeData(opt.train_src, opt.train_tgt,
                                          dicts['src'], dicts['tgt'])

    print('Preparing test ...')
    test = {}
    test['src'], test['tgt'] = makeData(opt.test_src, opt.test_tgt,
                                        dicts['src'], dicts['tgt'], True)

    print('Preparing validation ...')
    if '' not in (opt.valid_src, opt.valid_tgt):
        valid = {}
        valid['src'], valid['tgt'] = makeData(opt.valid_src, opt.valid_tgt,
                                              dicts['src'], dicts['tgt'])
    else:
        print('Empty validation')
        valid = test

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')

    print('Saving data to \'' + opt.save_data + '\'...')
    save_data = {'dicts': dicts,
                 'type':  opt.src_type,
                 'train': train,
                 'valid': valid,
                 'test': test}
    torch.save(save_data, opt.save_data)


if __name__ == "__main__":
    main()
