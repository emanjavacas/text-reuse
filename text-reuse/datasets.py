
import os
import csv

from seqmod.misc import Dict, PairedDataset
import seqmod.utils as u


def msrp_pairs(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        header = next(f).strip().split('\t')
        for line in f:
            row = dict(zip(header, line.strip().split('\t')))
            pair1 = row['#1 String'].split()
            pair2 = row['#2 String'].split()
            yield (pair1, pair2), row['Quality']


def quora_pairs(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            _, _, _, q1, q2, label, *_ = line
            if label not in ('1', '0'):
                continue
            yield (q1.split(), q2.split()), label


def load_dataset(loader, path, test=None, gpu=False, batch_size=100, **kwargs):
    path = os.path.expanduser(path)

    train_pairs, train_labels = zip(*loader(path))

    d = Dict(bos_token=u.BOS, eos_token=u.EOS, pad_token=u.PAD, **kwargs)
    d.fit(s for pair in train_pairs for s in pair)
    labels = Dict(sequential=False, force_unk=False).fit(train_labels)

    # transform [(a1, b1), ...] -> ([a1, a2], [b1, b2])
    train_pairs, train_labels = tuple(zip(*train_pairs)), list(train_labels)
    train = PairedDataset(
        train_pairs, train_labels, {'src': (d, d), 'trg': labels}, gpu=gpu,
        return_lengths=True, batch_size=batch_size)

    if test is not None:
        test_pairs, test_labels = zip(*loader(os.path.expanduser(test)))
        test_pairs, test_labels = tuple(zip(*test_pairs)), list(test_labels)
        test = PairedDataset(
            test_pairs, test_labels, {'src': (d, d), 'trg': labels}, gpu=gpu,
            batch_size=batch_size, return_lengths=True)
        train, valid = train.splits(test=0.1, dev=None, shuffle=True)
    else:
        train, valid, test = train.splits(test=0.1, dev=0.1, shuffle=True)
    
    return train, valid, test


def load_msrp(path='~/corpora/MSRParaphraseCorpus/', **kwargs):
    train = os.path.join(path, 'msr_paraphrase_{}.txt'.format('train'))
    test = os.path.join(path, 'msr_paraphrase_{}.txt'.format('test'))
    return load_dataset(msrp_pairs, train, test=test, **kwargs)


def load_quora(path='~/corpora/q_quora.csv', **kwargs):
    return load_dataset(quora_pairs, path, **kwargs)
