
import collections
import numpy as np
import pyemd


def get_histograms(s1, s2, w2i):
    w1, weights1 = zip(*[(w2i[w], c) for w, c in collections.Counter(s1).most_common()
                         if w in w2i])
    w2, weights2 = zip(*[(w2i[w], c) for w, c in collections.Counter(s2).most_common()
                         if w in w2i])
    words = sorted(list(set(w1 + w2)))
    indexer = {w: idx for idx, w in enumerate(words)}
    h1, h2 = np.zeros(len(indexer)), np.zeros(len(indexer))
    for w, weight in zip(w1, weights1):
        h1[indexer[w]] = weight / sum(weights1)
    for w, weight in zip(w2, weights2):
        h2[indexer[w]] = weight / sum(weights2)
    return h1, h2, np.array(words)


def get_wmd(s1, s2, dists, w2i):
    """
    Get WMD for two input sentences
    """
    s1, s2 = s1.split(), s2.split()
    h1, h2, words = get_histograms(s1, s2, w2i)
    D = dists[np.ix_(words, words)]
    return pyemd.emd(h1, h2, D)
