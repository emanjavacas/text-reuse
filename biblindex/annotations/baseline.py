
import utils
import random
import numpy as np

random.seed(1001)
np.random.seed(1001)


def load_gold(path='gold.csv', lemmas=False, return_ids=False):
    src, trg = [], []
    with open(path) as f:
        for line in f:
            id1, id2, s1, s2, l1, l2 = line.strip().split('\t')

            if lemmas:
                s1, s2 = l1, l2

            s1, s2 = utils.process_sent(s1), utils.process_sent(s2)

            if return_ids:
                s1, s2 = (id1, s1), (id2, s2)

            src.append(s1)
            trg.append(s2)

    return src, trg


def load_background(path='background.bible.csv',
                    lemmas=False, return_ids=False, n_background=35000):
    bg = []
    with open(path) as f:
        for line in f:
            idx, toks, lems = line.strip().split('\t')
            s = lems if lemmas else toks
            s = utils.process_sent(s)
            assert len(line) > 0
            if return_ids:
                s = (idx, s)
            bg.append(s)

    random.shuffle(bg)

    bg = bg[:n_background]

    return bg


# take random pick from target document
def random_baseline(at=1, n_background=0, lemmas=False):
    _, true = load_gold(lemmas=lemmas, return_ids=True)
    true, _ = zip(*true)          # use ids only
    bg = load_background(lemmas=lemmas, n_background=n_background, return_ids=True)
    print(len(bg))
    if bg:
        bg, _ = zip(*bg)
    bg += true[:]

    bg = np.array(bg)
    correct = 0
    for doc in range(len(true)):
        correct += (true[doc] in bg[np.random.permutation(len(bg))][:at])

    return correct / len(true)


# random_baseline(at=50, n_background=35000) * 100
