
import collections
import math
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.decomposition import TruncatedSVD
import tqdm

import utils
from steps import steps

random.seed(1001)
np.random.seed(1001)


# take random pick from target document
def random_baseline(at=1, path='bernard-gold.csv', n_background=35000, lemmas=False):
    _, true = utils.load_gold(path=path, lemmas=lemmas, return_ids=True)
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
    tfidf = TfidfVectorizer(vocabulary=vocab).fit(' '.join(s) for s in src + trg)

    src_embs = tfidf.transform(' '.join(s) for s in src)
    trg_embs = tfidf.transform(' '.join(s) for s in trg)

    D = cosine_distances(src_embs, trg_embs)

    return D


def bow_baseline(src, trc, min_freq=1):
    vocab = get_vocab(src + trg, min_freq)
    bow = CountVectorizer(vocabulary=vocab).fit(' '.join(s) for s in src + trg)
    src_embs = bow.transform(' '.join(s) for s in src)
    trg_embs = bow.transform(' '.join(s) for s in trg)

    return cosine_distances(src_embs, trg_embs)


def lsi_baseline(src, trg, min_freq=1, vectorizer=CountVectorizer, npc=10):
    vocab = get_vocab(src + trg, min_freq)
    vectorizer = vectorizer(vocabulary=vocab)
    embs = vectorizer.fit_transform(' '.join(s) for s in src + trg)
    embs = TruncatedSVD(n_components=npc).fit_transform(embs)

    return cosine_similarity(embs[:len(src)], embs[len(src):])
    

def get_d_min_freq(s, freqs):
    """
    Modeling the scholars: Detecting intertextuality  through enhanced
    word-level n-gram matching

    distance between the two most infrequent words in each of the two phrases
    """
    a, b, *_ = sorted(set(s), key=lambda w: freqs[w])
    return abs(s.index(a) - s.index(b))


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


def tesserae_score(s1, s2, freqs, method='max_dist'):

    matching = set(s1).intersection(set(s2))
    if len(matching) < 2:
        return 0

    f_t = sum([1 / freqs[w] for w in matching])
    f_s = sum([1 / freqs[w] for w in matching])
    if method == 'max_dist':
        d_t, d_s = get_d_max_dist(s1, matching), get_d_max_dist(s2, matching)
    elif method == 'min_freq':
        d_t, d_s = get_d_min_freq(s1, freqs), get_d_min_freq(s2, freqs)
    else:
        raise ValueError(method + " not known")

    return math.log((f_t + f_s) / (d_t + d_s))


def tesserae_baseline(src, trg, freqs, method='max_dist'):
    D = []
    for idx, s1 in tqdm.tqdm(enumerate(src)):
        D.append([tesserae_score(s1, s2, freqs, method) for s2 in trg])

    return np.array(D)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', default='bernard-gold.csv')
    parser.add_argument('--background_path', default='bernard-background.csv')
    parser.add_argument('--freqs_path', default='/home/manjavacas/corpora/latin.freqs')
    parser.add_argument('--stopwords_path', default='bernard.stop')
    parser.add_argument('--outputname', default='bernard')
    parser.add_argument('--lemmas', action='store_true')
    parser.add_argument('--n_background', default=35000, type=int)
    args = parser.parse_args()

    stopwords = utils.load_stopwords(args.stopwords_path)
    src, trg = utils.load_gold(
        path=args.gold_path, lemmas=args.lemmas, stopwords=stopwords)
    bg = utils.load_background(
        path=args.background_path, lemmas=args.lemmas, stopwords=stopwords)
    random.shuffle(bg)
    trg += bg[:args.n_background]
    vocab = set(w for s in src + trg for w in s)
    freqs = utils.load_frequencies(path=args.freqs_path, words=vocab)

    for w in vocab:
        if w not in freqs:
            freqs[w] = min(freqs.values())

    outputpath = 'results/lexical.{}.{}'.format(args.outputname, args.n_background)
    if args.lemmas:
        outputpath += '.lemmas'
    outputpath += '.csv'

    with open(outputpath, 'w') as f:
        with utils.writer(f, steps, ['method', 'npc']) as write:
            for npc in [10, 15, 20, 50]:
                write(lsi_baseline(src, trg, npc=npc), method='lsi', npc=npc)

            for method in ['max_dist', 'min_freq']:
                D = tesserae_baseline(src, trg, freqs, method=method)
                write(D, input_type='sim', method=method, npc=0)

            write(bow_baseline(src, trg), method='bow', npc=0)
            write(tf_idf_baseline(src, trg), method='tfidf', npc=0)
