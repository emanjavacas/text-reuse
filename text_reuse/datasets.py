
import os
import math
import csv
import json
from collections import Counter

from seqmod.misc import Dict, PairedDataset
import seqmod.utils as u


MSRP_PATH = '/home/manjavacas/corpora/MSRParaphraseCorpus/'
QUORA_PATH = '/home/manjavacas/corpora/'
SICK_PATH = '/home/manjavacas/corpora/SICK/'


def msrp_pairs(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        header = next(f).strip().split('\t')
        for line in f:
            row = dict(zip(header, line.strip().split('\t')))
            pair1, pair2 = row['#1 String'], row['#2 String']
            yield (pair1.split(), pair2.split()), row['Quality']


def quora_pairs(path):
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for line in reader:
            _, _, _, q1, q2, label = line
            yield (q1.split(), q2.split()), label


def snli_entailment_pairs(path):
    with open(path, 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj['gold_label'] == 'entailment':
                s1, s2 = obj['sentence1'], obj['sentence2']
                yield (s1.split(), s2.split()), '1'


def encode_sick_label(label, nclass=5):
    """
    Encode score from 1 to nclass into a categorical cross-entropy format:
    
    1:   [1.0, 0,   0, 0, ...]
    1.5: [0.5, 0.5, 0, 0, ...]

    Same format as in Socher's TreeLSTM paper, skipthoughts, etc...
    """
    label = float(label)
    if label < 1:
        raise ValueError("Encoding should be in range [1, nclass]")

    output = [0.0] * nclass
    for i in range(nclass):
        if i + 1 == (math.floor(label) + 1):
            output[i] = label - math.floor(label)
        if i + 1 == (math.floor(label)):
            output[i] = math.floor(label) - label + 1
    return output


def sick_pairs(path):
    with open(path, 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            yield (line[1].split(), line[2].split()), line[3]


def default_pairs(path):
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            try:
                s1, s2, label = line.split('\t')
                yield (s1.split(), s2.split()), label
            except Exception as e:
                print("Error on line {}".format(idx))
                raise e


def load_dataset(loader, path, test=None, valid=None,
                 include_test=True, include_valid=True,
                 gpu=False, batch_size=100,
                 d=None, preprocessing=None, use_vocab=True, **kwargs):

    path = os.path.expanduser(path)

    train_pairs, train_labels = zip(*loader(path))

    if d is None:
        d = Dict(bos_token=u.BOS, eos_token=u.EOS, pad_token=u.PAD, **kwargs)
        d.fit(s for pair in train_pairs for s in pair)
    labels = Dict(
        sequential=False, force_unk=False, use_vocab=use_vocab,
        preprocessing=preprocessing, dtype=float
    ).fit(train_labels)

    # transform [(a1, b1), ...] -> ([a1, a2], [b1, b2])
    train_pairs, train_labels = tuple(zip(*train_pairs)), list(train_labels)
    train = PairedDataset(
        train_pairs, train_labels, {'src': (d, d), 'trg': labels}, gpu=gpu,
        return_lengths=True, batch_size=batch_size)

    if include_test:
        if test is not None:
            test_pairs, test_labels = zip(*loader(os.path.expanduser(test)))
            test_pairs, test_labels = tuple(zip(*test_pairs)), list(test_labels)
            test = PairedDataset(
                test_pairs, test_labels, {'src': (d, d), 'trg': labels}, gpu=gpu,
                batch_size=batch_size, return_lengths=True)
        else:
            train, test = train.splits(test=0.1, dev=None, shuffle=True)
    else:
        test = None

    if include_valid:
        if valid is not None:
            valid_pairs, valid_labels = zip(*loader(os.path.expanduser(valid)))
            valid_pairs, valid_labels = tuple(zip(*valid_pairs)), list(valid_labels)
            valid = PairedDataset(
                valid_pairs, valid_labels, {'src': (d, d), 'trg': labels}, gpu=gpu,
                batch_size=batch_size, return_lengths=True)
        else:
            train, valid = train.splits(test=0.1, dev=None, shuffle=True)
    else:
        valid = None
    
    return train, valid, test


def load_msrp(path=MSRP_PATH, **kwargs):
    path = (path + '/msr_paraphrase_{}_tokenized.txt').format
    return load_dataset(default_pairs, path('train'), test=path('test'), **kwargs)


def load_quora(path=QUORA_PATH, **kwargs):
    path = os.path.join(path, 'quora_duplicate_questions_tokenized.tsv')
    return load_dataset(default_pairs, path, **kwargs)


def load_sick(path=SICK_PATH, **kwargs):
    path = os.path.join(path, 'SICK_{}_tokenized.txt').format
    return load_dataset(
        default_pairs, path('train'), test=path('test_annotated'), valid=path('trial'),
        preprocessing=encode_sick_label, use_vocab=False, dtype=float, **kwargs)


if __name__ == '__main__':
    # preprocess files
    import argparse
    parser = argparse.ArgumentParser(description="preprocess")
    parser.add_argument('corpus', help='e.g. "quora"')
    args = parser.parse_args()

    import ucto
    tokenizer = ucto.Tokenizer("tokconfig-eng")

    def tokenize(sent):
        tokenizer.process(sent)
        return ' '.join(list(map(str, tokenizer)))

    if args.corpus == 'quora':
        path = '/home/manjavacas/corpora/quora_duplicate_questions{}.tsv'.format
        with open(path('_tokenized'), 'w+') as f:
            for (s1, s2), label in quora_pairs(path('')):
                f.write('{}\t{}\t{}\n'.format(
                    tokenize(' '.join(s1)),
                    tokenize(' '.join(s2)),
                    label))

    elif args.corpus == 'msrp':
        path = '/home/manjavacas/corpora/MSRParaphraseCorpus/'
        path += 'msr_paraphrase_{}{}.txt'
        path = path.format
        with open(path('train', '_tokenized'), 'w+') as f:
            for (s1, s2), label in msrp_pairs(path('train', '')):
                f.write('{}\t{}\t{}\n'.format(
                    tokenize(' '.join(s1)),
                    tokenize(' '.join(s2)),
                    label))

        with open(path('test', '_tokenized'), 'w+') as f:
            for (s1, s2), label in msrp_pairs(path('test', '')):
                f.write('{}\t{}\t{}\n'.format(
                    tokenize(' '.join(s1)),
                    tokenize(' '.join(s2)),
                    label))

    elif args.corpus == 'sick':
        path = '/home/manjavacas/corpora/SICK/'
        for fname in ['SICK_test_annotated', 'SICK_train_full',
                      'SICK_train', 'SICK_trial', 'SICK']:
            getpath = (os.path.join(path, fname) + '{}.txt').format
            with open(getpath('_tokenized'), 'w+') as f:
                for (s1, s2), label in sick_pairs(getpath('')):
                    f.write('{}\t{}\t{}\n'.format(
                        tokenize(' '.join(s1)),
                        tokenize(' '.join(s2)),
                        label))
