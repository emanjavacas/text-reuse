
import collections
import math
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

import utils
from steps import steps

random.seed(1001)
np.random.seed(1001)


# take random pick from target document
def random_baseline(at=1, n_background=35000, lemmas=False):
    _, true = utils.load_gold(lemmas=lemmas, return_ids=True)
    true, _ = zip(*true)          # use ids only
    bg = utils.load_background(lemmas=lemmas, return_ids=True)
    random.shuffle(bg)
    bg = bg[:n_background]
    print(len(bg))
    if bg:
        bg, _ = zip(*bg)
    bg += true[:]

    bg = np.array(bg)
    correct = 0
    for doc in range(len(true)):
        correct += (true[doc] in bg[np.random.permutation(len(bg))][:at])

    return correct / len(true)


# random_baseline(at=100, n_background=35000) * 100

def get_vocab(sents, min_freq):
    vocab = {}
    for w, c in collections.Counter(w for s in sents for w in s).most_common():
        if c > min_freq:
            vocab[w] = len(vocab)
    return vocab


def tf_idf_baseline(src, trg, min_freq=1):
    vocab = get_vocab(src + trg, min_freq)

    src_embs = TfidfVectorizer(vocabulary=vocab).fit_transform(' '.join(s) for s in src)
    trg_embs = TfidfVectorizer(vocabulary=vocab).fit_transform(' '.join(s) for s in trg)

    D = cosine_distances(src_embs, trg_embs)

    return D


def get_d_max_dist(s, matching):
    """
    'Evaluating the literary significance of text re-use in Latin poetry'
    (James Gawley, Christopher Forstall, and Neil Coffee)

    d t : the greatest distance between two matching terms in the target
    d s : the greatest distance between two matching terms in the source
    """
    d = 0
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            if s[i] in matching and s[j] in matching:
                d = max(d, abs(i - j))

    return d


def get_d_min_freq(s, matching):
    """
    Modeling the scholars: Detecting intertextuality  through enhanced
    word-level n-gram matching

    (maximum) distance between the two most infrequent words in the phrase
    """
    by_freq = collections.defaultdict(list)
    for w, c in collections.Counter(s).most_common():
        by_freq[c].append(w)
    min_freqs = list(sorted(by_freq))
    if len(by_freq[min_freqs[0]]) >= 2:
        return get_d_max_dist(s, by_freq[min_freqs[0]])
    else:
        return get_d_max_dist(s, sum((by_freq[c] for c in min_freqs[:2]), []))


def tesserae_score(s1, s2, freqs1, freqs2, method='min_freq'):

    matching = set(s1).intersection(set(s2))
    if len(matching) < 2:
        return 0

    s1_c = collections.Counter(w for w in s1 if w in matching)
    s2_c = collections.Counter(w for w in s2 if w in matching)
    f_t = sum([1 / s1_c[w] for w in matching])
    f_s = sum([1 / s2_c[w] for w in matching])
    if method == 'min_freq':
        d_t, d_s = get_d_min_freq(s1, matching), get_d_min_freq(s2, matching)
    elif method == 'max_dist':
        d_t, d_s = get_d_max_dist(s1, matching), get_d_max_dist(s2, matching)
    else:
        raise ValueError(method + " not known")

    return math.log((f_t + f_s) / (d_t + d_s))


def tesserae_baseline(src, trg, src_freqs, trg_freqs, at=1, method='min_freq'):
    scores = []
    output = []
    for idx, s1 in enumerate(src):
        tscores = [tesserae_score(s1, s2, src_freqs, trg_freqs, method) for s2 in trg]
        idxs = np.argsort(tscores)[::-1][:at]
        # filter out those with 0 score
        idxs = [idx for idx in idxs if tscores[idx] > 0]
        output.append(([(idx, tscores[idx], len(set(s1).intersection(set(trg[idx]))))
                        for idx in idxs]))
        scores.append(1 if idx in idxs else 0)

    return sum(scores) / len(scores), output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lemmas', action='store_true')
    parser.add_argument('--n_background', default=35000, type=int)
    args = parser.parse_args()

    src, trg = utils.load_gold(lemmas=args.lemmas)
    bg = utils.load_background(lemmas=args.lemmas)
    random.shuffle(bg)
    trg += bg[:args.n_background]
    src_freqs = utils.load_bernard_freqs(lemmas=args.lemmas)
    trg_freqs = utils.get_freqs(trg)

    outputpath = 'results/lexical.{}'.format(args.n_background)

    if args.lemmas:
        outputpath += '.lemmas'
    outputpath += '.csv'

    with open(outputpath, 'w') as f:
        f.write('\t'.join(['method'] + list(map(str, steps))) + '\n')
        for method in ['max_dist', 'min_freq']:
            scores = []
            for step in steps:
                score, _ = tesserae_baseline(
                    src, trg, src_freqs, trg_freqs, at=step, method=method)
                scores.append(str(score))
            f.write('\t'.join(['tesserae-' + method] + scores) + '\n')

        D = tf_idf_baseline(src, trg)
        scores = []
        for step in steps:
            scores.append(str(utils.get_scores_at(D, at=step)))
        f.write('\t'.join(['tfidf'] + scores) + '\n')
