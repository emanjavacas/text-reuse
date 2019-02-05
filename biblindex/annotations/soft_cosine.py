
import joblib
import math
import random

import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import utils
from steps import steps


random.seed(1001)
np.random.seed(1001)


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

    original_vocab = set([w for s in src + trg for w in s])
    W, vocab = utils.load_embeddings(original_vocab)
    for w in original_vocab:
        if w not in vocab:
            vocab.append(w)
    vocab = {w: idx for idx, w in enumerate(vocab)}
    S = utils.get_cosine_similarity(W, W)

    src_feat = TfidfVectorizer(vocabulary=vocab).fit(' '.join(s) for s in src)
    trg_feat = TfidfVectorizer(vocabulary=vocab).fit(' '.join(s) for s in trg)

    outputpath = 'results/soft_cosine.{}'.format(args.n_background)

    if args.lemmas:
        outputpath += '.lemmas'
    outputpath += '.csv'

    overall = []

    for s in tqdm.tqdm(src):
        scores = joblib.Parallel(n_jobs=joblib.cpu_count() - 1)(
            joblib.delayed(soft_cosine)(s, t, src_feat, trg_feat, vocab, S) for t in trg)
        # scores = []
        # for t in trg:
        #     scores.append(soft_cosine(s, t, src_feat, trg_feat, vocab, S))

        overall.append(np.argsort(scores)[-max(steps):])

    output = []
    for step in steps:
        output.append(np.mean([i in overall[i][-step:] for i in range(len(src))]))

    with open(outputpath, 'w') as f:
        f.write(['method'] + list(map(str, steps)) + '\n')
        f.write(['soft_cosine'] + list(map(str, output)) + '\n')
