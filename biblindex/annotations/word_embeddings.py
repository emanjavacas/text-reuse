
import tqdm
import os
import glob
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import utils
import sentence_embeddings
import wmd
from steps import steps

random.seed(1001)


def get_wmd(src, trg, W, w2i):
    # get word dists
    print("Computing similarity matrix")
    dists = 1 - cosine_similarity(W)
    print("Done")
    D = np.zeros((len(src), len(trg)))
    for i in tqdm.tqdm(range(len(src))):
        for j in range(len(trg)):
            D[i, j] = wmd.get_wmd(' '.join(src[i]), ' '.join(trg[j]), dists, w2i)

    return D


def get_indices(D, at=5):
    index = np.arange(0, len(D))
    index = np.repeat(index[:, None], at, axis=1)
    retrieved = np.argsort(D, axis=1)[:, :at]
    return retrieved, np.sum(retrieved == index, axis=1)


def plot_results(path='./results.csv', most_at=50):
    df = pd.read_csv(path, sep='\t')
    df = pd.DataFrame.from_dict(
        [{"method": group[0], '@': key, 'score': val}
         for _, group in df.iterrows()
         for key, val in list(group.items())[1:]])
    df = df.astype({'@': np.int32})
    sns.lineplot(x='@', y='score', hue='method', data=df[df['@'] < most_at],
                 markers=True, style='method')
    # ax = sns.lineplot(x='@', y='score', hue='method', data=df[df['@'] < most_at])
    # ax.set_xscale("log")
    plt.show()


def plot_background(lemma=False, at=20):
    rows = []
    for f in glob.glob('results*csv'):
        if (lemma and 'lemma' not in f) or (not lemma and 'lemma' in f):
            continue
        n_background = int(os.path.basename(f).split('.')[1])
        for _, group in pd.read_csv(f, sep='\t').iterrows():
            for key, val in list(group.items())[1:]:
                if int(key) == at:
                    rows.append({'method': group[0], 'score': val, 'n': n_background})

    sns.lineplot(x='n', y='score', hue='method', data=pd.DataFrame.from_dict(rows),
                 markers=True, style='method')
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', default='bernard-gold.csv')
    parser.add_argument('--background_path', default='bernard-background.csv')
    parser.add_argument('--stopwords_path', default='bernard.stop')
    parser.add_argument('--freqs_path', default='/home/manjavacas/corpora/latin.freqs')
    parser.add_argument('--embeddings_path', default=utils.LATIN)
    parser.add_argument('--outputname', default='bernard')
    parser.add_argument('--n_background', type=int, default=35000)
    parser.add_argument('--lemmas', action='store_true')
    parser.add_argument('--avoid_lexical', action='store_true')
    parser.add_argument('--threshold', type=int, default=2)

    args = parser.parse_args()

    stopwords = utils.load_stopwords(args.stopwords_path)
    src, trg = [], []
    for s, t in zip(*utils.load_gold(
            path=args.gold_path, lemmas=args.lemmas, stopwords=stopwords)):
        if args.avoid_lexical and len(set(s).intersection(set(t))) >= args.threshold:
            continue
        src.append(s)
        trg.append(t)


    bg = []
    if args.n_background > 0:
        bg = utils.load_background(
            path=args.background_path, lemmas=args.lemmas, stopwords=stopwords)
        random.shuffle(bg)
        bg = bg[:args.n_background]
    vocab = set(w for s in src + trg + bg for w in s)
    W, words = utils.load_embeddings(vocab, path=args.embeddings_path)
    # remove sents without words (after checking for them in the embeddings)
    src, trg = zip(*[(s, t) for s, t in zip(src, trg)
                     if any(w in set(words) for w in s) and \
                     any(w in set(words) for w in t)])
    src, trg = list(src), list(trg)
    bg = [s for s in bg if any(w in set(words) for w in s)]
    trg += bg

    freqs = utils.load_frequencies(path=args.freqs_path, words=set(words))
    freqs = [freqs.get(w, min(freqs.values())) for w in words]

    w2i = {w: idx for idx, w in enumerate(words)}

    print("Number of pairs", len(src))

    outfile = 'results/distributional.{}.{}'.format(args.outputname, args.n_background)
    if args.lemmas:
        outfile += '.lemmas'
    if args.avoid_lexical:
        outfile += '.overlap{}'.format(args.threshold)
    outfile += '.csv'

    with open(outfile, 'w') as f:
        with utils.writer(f, steps, ['method']) as write:
            # BOW
            print("BOW", end="")
            embedder = sentence_embeddings.BOW(W, words)
            D = cosine_similarity(embedder.transform(src), embedder.transform(trg))
            write(D, input_type='sim', method='BOW')
            print()

            # TfIdf
            print("TfIdf", end="")
            embedder = sentence_embeddings.TFIDF(W, words, src + trg)
            D = cosine_similarity(embedder.transform(src), embedder.transform(trg))
            write(D, input_type='sim', method='tfidf')
            print()

            # # SIF
            # print("SIF", end="")
            # embedder = sentence_embeddings.SIF(W, words, freqs)
            # D = cosine_similarity(embedder.transform(src), embedder.transform(trg))
            # write(D, input_type='sim', method='BOW')
            # print()

            # WMD
            print("WMD", end="")
            write(get_wmd(src, trg, W, w2i), method='WMD')
            print()
