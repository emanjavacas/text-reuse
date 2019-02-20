
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from elmoformanylangs import Embedder

import utils
from steps import steps
import sentence_embeddings
import wmd

random.seed(1001)

PATH = '/home/manjavacas/code/python/ELMoForManyLangs/models/latin'


def sif_weight(s_emb, sent, freqs, a):
    return np.mean([a / (a + freqs[w]) * emb for w, emb in zip(sent, s_emb)], 0)


def get_sif_embeddings(sents, s_embs, freqs, a=1e-3, npc=1):
    s_embs = np.array([sif_weight(emb, s, freqs, a=a) for emb, s in zip(s_embs, sents)])
    s_embs = sentence_embeddings.remove_pc(s_embs, npc=npc)
    return s_embs


def get_tfidf_embeddings(tfidf, sents, s_embs):
    sents = [' '.join(s) for s in sents]
    X = tfidf.transform(sents)
    for i, embs in enumerate(s_embs):
        sents[i] = np.mean(
            [w * emb for (_, w), emb in zip(X[i].todok().items(), embs)],
            axis=0)
    return np.array(sents)


def get_wmd(src, trg, W, w2i):
    # get word dists
    dists = 1 - cosine_similarity(W)
    D = np.zeros((len(src), len(trg)))
    for i in range(len(src)):
        for j in range(len(trg)):
            D[i, j] = wmd.get_wmd(' '.join(src[i]), ' '.join(trg[j]), dists, w2i)
    return D


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', default='bernard-gold.csv')
    parser.add_argument('--n_background', type=int, default=35000)
    parser.add_argument('--modelpath', default=PATH)
    parser.add_argument('--avoid_lexical', action='store_true')
    parser.add_argument('--threshold', type=int, default=2)
    args = parser.parse_args()

    model = Embedder(args.modelpath, batch_size=10)

    src, trg = [], []
    for s, t in zip(*utils.load_gold(lower=False, remnonalpha=False, remstop=False)):
        if args.avoid_lexical and len(set(s).intersection(set(t))) >= args.threshold:
            continue
        src.append(s.split())
        trg.append(t.split())

    print("Number of pairs", len(src))

    bg = []
    if args.n_background > 0:
        bg = utils.load_background(lower=False, remnonalpha=False, remstop=False)
        random.shuffle(bg)
        bg = [s for s in bg if len(s.split()) > 1]  # remove length-1 sentences
        bg = bg[:args.n_background]
    trg += [s.split() for s in bg]
    words = list(set(w for s in trg + src for w in s))
    freqs = utils.load_frequencies(words=set(words))
    freqs = {w: freqs.get(w, min(freqs.values())) for w in words}
    w2i = {w: idx for idx, w in enumerate(words)}

    outfile = 'results/elmo.{}.csv'.format(args.n_background)

    with open(outfile, 'w') as f:
        with utils.writer(f, steps, ['method', 'output_layer']) as write:
            for layer in [-1, 0, 1, 2]:
                print("Encoding")
                src_embs = model.sents2elmo(src, output_layer=layer)
                trg_embs = model.sents2elmo(trg, output_layer=layer)

                print("BOW", end="")
                D = cosine_similarity(
                    np.array([emb.mean(0) for emb in src_embs]),
                    np.array([emb.mean(0) for emb in trg_embs]))
                write(D, input_type='sim', 'BOW', str(layer))
                print()

                print("SIF", end="")
                D = cosine_similarity(
                    get_sif_embeddings(src, src_embs, freqs),
                    get_sif_embeddings(trg, trg_embs, freqs))
                write(D, input_type='sim', 'SIF', str(layer))

                print("TfIdf", end="")
                tfidf = TfidfVectorizer().fit([' '.join(s) for s in src + trg])
                D = cosine_similarity(
                    get_tfidf_embeddings(tfidf, src, src_embs),
                    get_tfidf_embeddings(tfidf, trg, trg_embs))
                write(D, input_type='sim', 'tfidf', str(layer))
                print()

                if layer != 0:  # don't do wmd at any other level than word
                    continue

                print("WMD", end="")
                write(get_wmd(src, trg, W, w2i), 'WMD', str(layer))
                print()
