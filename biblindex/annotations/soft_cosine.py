
import random

import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import utils
from steps import steps


random.seed(1001)
np.random.seed(1001)


"""
# First approch: pretty slow as expected

def soft_cosine(src, trg, src_feat, trg_feat, vocab, S):
    num = den1 = den2 = 0
    src_emb = dict(src_feat.transform([' '.join(src)]).todok())
    trg_emb = dict(trg_feat.transform([' '.join(trg)]).todok())
    for w1 in set(src + trg):
        i = vocab[w1]
        for w2 in set(src + trg):
            j = vocab[w2]
            if i > len(S) - 1 or j > len(S) - 1:
                sim = 1 if i == j else np.mean(S)  # backoff to average similarity
            else:
                sim = S[i, j]
            num  += sim * src_emb.get((0, i), 0) * trg_emb.get((0, j), 0)
            den1 += sim * src_emb.get((0, i), 0) * src_emb.get((0, j), 0)
            den2 += sim * trg_emb.get((0, i), 0) * trg_emb.get((0, j), 0)

    return num / (math.sqrt(den1) * math.sqrt(den2))


# Testing code
src, trg = utils.load_gold()

s1s = []
for i in tqdm.tqdm(range(len(trg))):
    # s2 = soft_cosine(src[0], trg[i], src_feat, trg_feat, vocab, S)
    s1 = s[0] @ M @ t[i] / (math.sqrt(s[0] @ M @ s[0]) * math.sqrt(t[i] @ M @ t[i]))
    s1s.append(s1)
    # assert abs(s1 - s2) < 0.0001, abs(s1 - s2)
s3s = soft_cosine3(s[0], t, M)


# slow
def soft_cosine3(s, ts, M):
    num = s[None, :] @ (M @ ts.T)
    den1 = np.sqrt(s @ M @ s)
    den2 = np.sqrt(np.diag(ts @ M @ ts.T))
    return (num / ((np.ones(len(den2)) * den1) * den2))[0]


# Fully batched: still not quite there
def soft_cosine2(src, trg, vocab, S):
    src_embs = TfidfVectorizer(vocabulary=vocab).fit_transform(' '.join(s) for s in src)
    trg_embs = TfidfVectorizer(vocabulary=vocab).fit_transform(' '.join(s) for s in trg)

    M = np.zeros([len(vocab), len(vocab)]) + np.mean(S)
    M[:len(S),:len(S)] = S
    np.fill_diagonal(M, 1)

    src_embs, trg_embs = src_embs.toarray(), trg_embs.toarray()

    num = (src_embs @ M) @ trg_embs.T
    den1 = np.sqrt((src_embs @ M) @ src_embs.T)
    den2 = np.sqrt((trg_embs @ M) @ trg_embs.T)

    return num / (den1 * den2)
"""

# speed up by caching computations
def soft_cosine4(ss, ts, M):
    """
    ss : n query docs in BOW format, np.array(n, vocab)
    ts : m indexed docs in BOW format, np.array(m, vocab)
    M : similarity matrix, np.array(vocab, vocab)

    returns : sims, soft cosine similarities, np.array(n, m)
    """
    sims = np.zeros((len(ss), len(ts)))
    MtsT = M @ ts.T
    den2 = np.sqrt(np.diag(ts @ MtsT))
    for idx, s in tqdm.tqdm(enumerate(ss), total=len(ss)):
        num = s[None, :] @ MtsT
        den1 = np.sqrt(s @ M @ s)
        sims[idx] = (num / ((np.ones(len(den2)) * den1) * den2))[0]
    return np.nan_to_num(sims, copy=False)


def get_M(S, vocab, beta=1):
    """
    Transform an input similarity matrix for a possibly reduced vocabulary space
    into the similarity matrix for the whole space (i.e. include OOVs). It assumes
    that OOVs are indexed at the end of the space - e.g. for a vocab of 1001 where
    the word "aardvark" doesn't have an entry in S (1000 x 1000), the entry for
    "aardvark" is 1001. By default the similarity vector for OOVs is a one-hot vector
    implying that the word is only similar to itself.

    S : input similarity matrix (e.g. sklearn.metrics.pairwise.cosine_similarity(W)
        where W is your embedding matrix), np.array(vocab, vocab)
    vocab : list of all words in your space
    beta : raise your similarities to this power to reduce model confidence on word
        similarities
    """
    M = np.zeros([len(vocab), len(vocab)])
    M[:len(S), :len(S)] = np.power(np.clip(S, a_min=0, a_max=np.max(S)), beta)
    np.fill_diagonal(M, 1)
    return M


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lemmas', action='store_true')
    parser.add_argument('--gold_path', default='bernard-gold.csv')
    parser.add_argument('--background_path', default='bernard-background.csv')
    parser.add_argument('--embeddings_path', default=utils.LATIN)
    parser.add_argument('--stopwords_path', default='bernard.stop')
    parser.add_argument('--outputname', default='bernard')
    parser.add_argument('--n_background', default=35000, type=int)
    args = parser.parse_args()

    stopwords = utils.load_stopwords(args.stopwords_path)
    src, trg = utils.load_gold(
        path=args.gold_path, lemmas=args.lemmas, stopwords=stopwords)
    bg = utils.load_background(
        path=args.background_path, lemmas=args.lemmas, stopwords=stopwords)
    random.shuffle(bg)
    trg += bg[:args.n_background]

    original_vocab = set([w for s in src + trg for w in s])
    W, vocab = utils.load_embeddings(original_vocab, path=args.embeddings_path)
    for w in original_vocab:
        if w not in vocab:
            vocab.append(w)
    vocab = {w: idx for idx, w in enumerate(vocab)}

    print("Computing similarity matrix")
    S = cosine_similarity(W)
    print("Done")

    tfidf = TfidfVectorizer(vocabulary=vocab).fit(' '.join(s) for s in src + trg)
    src_embs = tfidf.transform(' '.join(s) for s in src).toarray()
    trg_embs = tfidf.transform(' '.join(s) for s in trg).toarray()

    outputpath = 'results/soft_cosine.{}.{}'.format(args.outputname, args.n_background)

    if args.lemmas:
        outputpath += '.lemmas'
    outputpath += '.csv'

    betas = [1, 2, 5, 7.5, 15, 100, 10000]

    with open(outputpath, 'w') as f:
        with utils.writer(f, steps, ['method', 'beta', 'a']) as write:

            for beta in betas:
                D = soft_cosine4(src_embs, trg_embs, get_M(S, vocab, beta=beta))
                write(D, input_type='sim', method='semantic', beta=beta, a=0)

            # Levenshtein
            lev = utils.get_levenshtein_S(sorted(vocab, key=lambda w: vocab[w]))
            for beta in betas:
                D = soft_cosine4(src_embs, trg_embs, lev ** beta)
                write(D, input_type='sim', method='levenshtein', beta=beta, a=0)

            # # Interpolation
            # lev = lev ** 5 # best beta
            # M = get_M(S, vocab, beta=5)
            # for a in [a / 10 for a in range(0, 11, 1)][::-1]:
            #     D = soft_cosine4(src_embs, trg_embs, a * lev + (1 - a) * M)
            #     write(D, input_type='sim', method='interpolation', beta=0, a=a)

            # Random
            M = utils.get_random_matrix(vocab)
            for beta in betas:
                D = soft_cosine4(src_embs, trg_embs, M ** beta)
                write(D, input_type='sim', method='random', beta=beta, a=0)
