
import joblib
import math
import random

import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def soft_cosine3(s, ts, M):
    num = s[None, :] @ (M @ ts.T)
    den1 = np.sqrt(s @ M @ s)
    den2 = np.sqrt(np.diag(ts @ M @ ts.T))
    return (num / ((np.ones(len(den2)) * den1) * den2))[0]


# speed up by caching computations
def soft_cosine4(ss, ts, M):
    sims = np.zeros((len(ss), len(ts)))
    MtsT = M @ ts.T
    den2 = np.sqrt(np.diag(ts @ MtsT))
    for idx, s in tqdm.tqdm(enumerate(ss), total=len(ss)):
        num = s[None, :] @ MtsT
        den1 = np.sqrt(s @ M @ s)
        sims[idx] = (num / ((np.ones(len(den2)) * den1) * den2))[0]
    return sims


def get_M(S, vocab, factor=1):
    M = np.zeros([len(vocab), len(vocab)])
    M[:len(S), :len(S)] = np.power(np.clip(S, a_min=0, a_max=np.max(S)), factor)
    np.fill_diagonal(M, 1)
    return M


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

    original_vocab = set([w for s in src + trg for w in s])
    W, vocab = utils.load_embeddings(original_vocab)
    for w in original_vocab:
        if w not in vocab:
            vocab.append(w)
    vocab = {w: idx for idx, w in enumerate(vocab)}

    print("Computing similarity matrix")
    S = cosine_similarity(W)
    print("Done")

    src_embs = TfidfVectorizer(vocabulary=vocab).fit_transform(' '.join(s) for s in src)
    trg_embs = TfidfVectorizer(vocabulary=vocab).fit_transform(' '.join(s) for s in trg)
    src_embs, trg_embs = src_embs.toarray(), trg_embs.toarray()

    outputpath = 'results/soft_cosine.{}'.format(args.n_background)

    if args.lemmas:
        outputpath += '.lemmas'
    outputpath += '.csv'

    with open(outputpath, 'w') as f:
        f.write('\t'.join(['method', 'factor'] + list(map(str, steps))) + '\n')
        for factor in [1, 1.25, 1.75, 2, 2.5, 5, 10]:
            sims = soft_cosine4(src_embs, trg_embs, get_M(S, vocab, factor=factor))
            scores = []
            for step in steps:
                scores.append(utils.get_scores_at(sims, at=step, input_type='sim'))
            scores = list(map(str, scores))
            f.write('\t'.join(['soft_cosine', str(factor)] + scores) + '\n')


# n_background = 10000
# src, trg = utils.load_gold()
# bg = utils.load_background()
# random.shuffle(bg)
# trg += bg[:n_background]
# original_vocab = set([w for s in src + trg for w in s])
# W, vocab = utils.load_embeddings(original_vocab)
# for w in original_vocab:
#     if w not in vocab:
#         vocab.append(w)
# vocab = {w: idx for idx, w in enumerate(vocab)}

# print("Computing similarity matrix")
# S = cosine_similarity(W)
# M = np.zeros([len(vocab), len(vocab)]) + np.mean(S)
# M[:len(S),:len(S)] = S
# np.fill_diagonal(M, 1)
# print("Done")

# src_embs = TfidfVectorizer(vocabulary=vocab).fit_transform(' '.join(s) for s in src)
# trg_embs = TfidfVectorizer(vocabulary=vocab).fit_transform(' '.join(s) for s in trg)

# ss,ts =src_embs.toarray(), trg_embs.toarray()

# output = np.zeros((len(ss), len(ts)))
# with timing():
#     sims = soft_cosine3(ss[0], ts, M)
#     # MtsT = M @ ts.T
#     # den2 = np.sqrt(np.diag(ts @ MtsT))
#     for idx, s in tqdm.tqdm(enumerate(ss)):
#         num = s[None, :] @ MtsT
#         den1 = np.sqrt(s @ M @ s)
#         output[idx] = (num / ((np.ones(len(den2)) * den1) * den2))[0]