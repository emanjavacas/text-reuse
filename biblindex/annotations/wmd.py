
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


def get_wmd(s1, s2, dists, w2i, get_flow=False):
    """
    Get WMD for two input sentences
    """
    s1, s2 = s1.split(), s2.split()
    h1, h2, words = get_histograms(s1, s2, w2i)
    D = dists[np.ix_(words, words)]

    if get_flow:
        return pyemd.emd_with_flow(h1, h2, D)

    return pyemd.emd(h1, h2, D)


if __name__ == '__main__':
    import random
    import utils
    import retrieval
    import sentence_embeddings

    lemmas = True
    n_background = 1000

    src, trg = retrieval.load_gold(lemmas=lemmas)
    bg = []
    if n_background > 0:
        bg = retrieval.load_background(lemmas=lemmas)
        random.shuffle(bg)
        bg = bg[:n_background]
    vocab = set(w for s in src + trg + bg for w in s)
    W, words = utils.load_embeddings(vocab)
    # remove bg sents without words (after checking for them in the embeddings)
    bg = [s for s in bg if any(w in set(words) for w in s)]
    trg += bg
    freqs = utils.load_frequencies(words=set(words))
    freqs = [freqs.get(w, min(freqs.values())) for w in words]

    w2i = {w: idx for idx, w in enumerate(words)}
    embedder = sentence_embeddings.BOW(W, words)
    D = retrieval.get_cosine_distance(embedder.transform(src), embedder.transform(trg))
    indices, result = retrieval.get_indices(D)
