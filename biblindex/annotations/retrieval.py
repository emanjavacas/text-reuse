
import os
import glob
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils
import sentence_embeddings
import wmd

random.seed(1001)


def get_cosine_distance(src, trg, batch=1000):
    src_norm = src / np.linalg.norm(src, axis=1)[:, None]
    trg_norm = trg / np.linalg.norm(trg, axis=1)[:, None]
    if len(src) <= batch:
        D = (src_norm[:, None] * trg_norm).sum(2)
    else:
        D = np.zeros((len(src), len(trg)))
        for i in range(0, len(src), batch):
            i_to = min(i+batch, len(src))
            for j in range(0, len(trg), batch):
                j_to = min(j+batch, len(trg))
                D[i:i_to, j:j_to] = (src_norm[i:i_to, None] * trg_norm[j:j_to]).sum(2)
    return 1 - D


def get_wmd(src, trg, W, w2i):
    # get word dists
    dists = get_cosine_distance(W, W)
    D = np.zeros((len(src), len(trg)))
    for i in range(len(src)):
        for j in range(len(trg)):
            D[i, j] = wmd.get_wmd(' '.join(src[i]), ' '.join(trg[j]), dists, w2i)

    return D


def get_scores_at(D, at=5):
    index = np.arange(0, len(D))
    index = np.repeat(index[:, None], at, axis=1)

    return float(np.sum(np.argsort(D, axis=1)[:, :at] == index, axis=1).mean())


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
    parser.add_argument('--n_background', type=int, default=10000)
    parser.add_argument('--lemmas', action='store_true')

    args = parser.parse_args()

    src, trg = utils.load_gold(lemmas=args.lemmas)
    bg = []
    if args.n_background > 0:
        bg = utils.load_background(lemmas=args.lemmas)
        random.shuffle(bg)
        bg = bg[:args.n_background]
    vocab = set(w for s in src + trg + bg for w in s)
    W, words = utils.load_embeddings(vocab)
    # remove bg sents without words (after checking for them in the embeddings)
    bg = [s for s in bg if any(w in set(words) for w in s)]
    trg += bg
    freqs = utils.load_frequencies(words=set(words))
    freqs = [freqs.get(w, min(freqs.values())) for w in words]

    w2i = {w: idx for idx, w in enumerate(words)}
    ats = [1, 5, 10, 20, 50]

    suffix = str(args.n_background) + ('.lemma' if args.lemmas else '')
    outfile = 'results.{}.csv'.format(suffix)

    with open(outfile, 'w') as f:
        # header
        f.write('\t'.join(['method', *list(map(str, ats))]) + '\n')

        # BOW
        print("BOW", end="")
        embedder = sentence_embeddings.BOW(W, words)
        D = get_cosine_distance(embedder.transform(src), embedder.transform(trg))
        results = []
        for at in ats:
            print(".", end='', flush=True)
            results.append(get_scores_at(D, at=at))
        f.write('\t'.join(['BOW'] + list(map(str, results))) + '\n')
        print()

        # SIF
        print("SIF", end="")
        embedder = sentence_embeddings.SIF(W, words, freqs)
        D = get_cosine_distance(embedder.transform(src), embedder.transform(trg))
        results = []
        for at in ats:
            print(".", end='', flush=True)
            results.append(get_scores_at(D, at=at))
        f.write('\t'.join(['SIF'] + list(map(str, results))) + '\n')
        print()

        # WMD
        print("WMD", end="")
        results = []
        D = get_wmd(src, trg, W, w2i)
        for at in ats:
            print(".", end='', flush=True)
            results.append(get_scores_at(D, at=at))
        f.write('\t'.join(['WMD'] + list(map(str, results))) + '\n')
        print()

        # TfIdf
        print("TfIdf", end="")
        embedder = sentence_embeddings.TFIDF(W, words)
        D = get_cosine_distance(embedder.transform(src), embedder.transform(trg))
        results = []
        for at in ats:
            print(".", end='', flush=True)
            results.append(get_scores_at(D, at=at))
        f.write('\t'.join(['TfIdf'] + list(map(str, results))) + '\n')
        print()


# src, trg = load_gold()
# bg = load_background()
# random.shuffle(bg)
# vocab = set(w for s in src + trg for w in s)
# W, words = utils.load_embeddings(vocab)
# print(len(bg))
# bg = [s for s in bg if any(w in set(words) for w in s)]
# print(len(bg))
# trg += bg
# freqs = utils.load_frequencies(words=set(words))
# freqs = [freqs.get(w, min(freqs.values())) for w in words]
# w2i = {w: idx for idx, w in enumerate(words)}
# embedder = sentence_embeddings.SIF(W, words, freqs)
# embedder = sentence_embeddings.BOW(W, words)
# D = get_cosine_distance(embedder.transform(src), embedder.transform(trg))
# get_scores_at(D, at=5)
