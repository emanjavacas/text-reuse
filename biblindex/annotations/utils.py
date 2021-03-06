
import time
import contextlib
import collections
import os
import glob
import re
import json

import tqdm
from lxml import etree
import numpy as np
import pie


# LATIN = '/home/manjavacas/corpora/word_embeddings/latin.embeddings'
LATIN = 'bernard.fasttext.embeddings'


def get_scores_at(D, at=5, input_type='dist'):
    index = np.arange(0, len(D))
    index = np.repeat(index[:, None], at, axis=1)

    if input_type == 'dist':
        return float(np.sum(np.argsort(D, axis=1)[:, :at] == index, axis=1).mean())
    elif input_type == 'sim':
        return float(np.sum(np.argsort(D, axis=1)[:, -at:] == index, axis=1).mean())
    else:
        raise ValueError("Unknown `input_type`: {}".format(input_type))


def get_average_rank(D, input_type='dist'):
    index = np.arange(0, len(D))
    index = np.repeat(index[:, None], D.shape[1], axis=1)
    argsort = np.argsort(D, axis=1)
    _, ranking = np.where(index == argsort)
    if input_type == 'sim':
        ranking = D.shape[1] - 1 - ranking
    ranking += 1
    return float((1/ranking).sum()) / D.shape[0]


@contextlib.contextmanager
def writer(fp, steps, keys):
    # write header
    fp.write('\t'.join(['metric', '@', 'score'] + list(sorted(keys))) + '\n')
    
    def write(D, input_type='dist', **kwargs):
        for key in kwargs:
            assert key in keys, "Unknown input key {}".format(key)
        vals = list(map(str, (kwargs[k] for k in sorted(keys))))

        rank = get_average_rank(D, input_type=input_type)
        fp.write('\t'.join(['MAP', '0', str(rank)] + vals) + '\n')
        for step in steps:
            score = get_scores_at(D, at=step, input_type=input_type)
            fp.write('\t'.join(['P', str(step), str(score)] + vals) + '\n')
            fp.flush()

    yield write


def dump_scores(fp, D, steps, input_type, **kwargs):
    fp.write()
    for step in steps:
        atK = get_scores_at(D, at=step, input_type=input_type)


def load_stopwords(path='bernard.stop'):
    with open(path) as f:
        return set(w.strip() for w in f.readlines())


def process_sent(s, stopwords=set(), lower=True, remnonalpha=True, remstop=True):
    # lower
    if lower:
        s = s.lower()
    # non alpha
    if remnonalpha:
        s = re.findall(r"(?u)\b\w\w+\b", s)
    # remove stopwords
    if remstop:
        s = [w for w in s if w not in stopwords]

    return s


def load_gold(path='bernard-gold.csv', return_ids=False, lemmas=False, **kwargs):

    src, trg = [], []
    with open(path) as f:
        for line in f:
            id1, id2, s1, s2, l1, l2 = line.strip().split('\t')
            if lemmas:
                s1, s2 = l1, l2
            s1, s2 = process_sent(s1, **kwargs), process_sent(s2, **kwargs)

            if return_ids:
                s1, s2 = (id1, s1), (id2, s2)
            src.append(s1)
            trg.append(s2)

    return src, trg


def load_background(path='bernard-background.csv', return_ids=False, lemmas=False,
                    **kwargs):
    bg = []
    with open(path) as f:
        for line in f:
            idx, toks, lems = line.strip().split('\t')
            s = lems if lemmas else toks
            s = process_sent(s, **kwargs)
            if return_ids:
                s = (idx, s)
            bg.append(s)

    return bg


def load_bernard(path='../splits/SCT1-5'):

    def remove_ns(tag):
        return re.sub(r'{[^}]+}', '', tag)

    tokens, lemmas = [], []
    for f in glob.glob(os.path.join(path, '*xml')):
        try:
            with open(f) as fn:
                tree = etree.fromstring(fn.read().encode())
            doc = []
            for w in tree.xpath('.//tei:w|.//tei:pc',
                                namespaces={'tei': 'http://www.tei-c.org/ns/1.0'}):
                token = w.text
                if remove_ns(w.tag) == 'w':
                    lemma = w.attrib['lemma']
                else:
                    lemma = w.text
                doc.append((token, lemma))
            token, lemma = zip(*doc)
            tokens.append(token)
            lemmas.append(lemma)
        except Exception as e:
            print("Error", f, e)
    return tokens, lemmas


def get_freqs(docs, **kwargs):
    counts = collections.Counter(
        w for doc in docs for w in process_sent(' '.join(doc), **kwargs))

    return {w: c/sum(counts.values()) for w, c in counts.items()}


def load_bernard_freqs(lemmas=False, **kwargs):
    bern_toks, bern_lems = load_bernard()
    if lemmas:
        bern_toks = bern_lems
    return get_freqs(bern_toks, **kwargs)


def load_embeddings(words, path=LATIN):
    print("loading {} words from {}".format(len(words), path))
    embs, vocab = [], []
    with open(path) as f:
        next(f)
        for line in f:
            word, *vec = line.split()
            if word in words:
                embs.append(list(map(float, vec)))
                vocab.append(word)

    print("Found {}/{} words".format(len(vocab), len(words)))
    return np.array(embs), vocab


def load_frequencies(path='/home/manjavacas/corpora/latin.freqs', words=None):
    freqs = {}
    total = 0
    with open(path) as f:
        for line in f:
            try:
                w, freq = line.strip().split()
            except:
                continue
            freq = float(freq)
            total += freq
            if words and w not in words:
                continue
            freqs[w] = freq

    # if freqs are already normalized, values won't change
    freqs = {w: freq / total for w, freq in freqs.items()}

    return freqs


def load_bible(path='../splits/SCT1-5.json'):
    bible_by_id = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if obj['type'] == 'inexactQuotation-allusion':
                bible_by_id[obj['id']] = {'text': obj['ref'], 'url': obj['url']}
    return bible_by_id


def lemmatize(model, sent, use_beam=True, beam_width=12, device='cpu'):
    inp, _ = pie.data.pack_batch(model.label_encoder, [sent], device=device)
    return model.predict(inp, "lemma", use_beam=use_beam, beam_width=beam_width)


def get_levenshtein_S(words):
    import Levenshtein
    D = np.zeros((len(words), len(words)))
    for i in tqdm.tqdm(range(len(words)), total=len(words)):
        for j in range(i + 1, len(words)):
            D[i, j] = Levenshtein.distance(words[i], words[j])
            D[i, j] = D[i, j] / max(len(words[i]), len(words[j]))
    # make symmetric
    D += D.T
    # normalize
    D = 1 - D
    return D


def load_levenshtein(path, words):
    D = np.zeros((len(words), len(words)))
    words_ = {w: idx for idx, w in enumerate(words)}
    done = set()
    with open(path) as f:
        for line in f:
            w1, w2, d = line.strip().split('\t')
            if w1 in words_ and w2 in words_:
                D[words_[w1], words_[w2]] = float(d) / max(len(w1), len(w2))
                done.add(w1)
                done.add(w2)

    diff = done.difference(set(words_))
    if diff:
        print("missing {} words".format(len(diff)))

    D += D.T
    D = 1 - D

    return D


def get_synonyms():
    if os.path.isfile('latin.synonyms'):
        synonyms = {}
        with open('latin.synonyms') as f:
            for line in f:
                lem, *lems = line.strip().split('\t')
                synonyms[lem] = set(lems)
    else:
        from multiwordnet.wordnet import WordNet
        wn = WordNet('latin')
        synonyms = collections.defaultdict(set)
        for lemma in tqdm.tqdm(wn.lemmas, total=len(list(wn.lemmas))):
            if lemma.synonyms:
                synonyms[lemma.lemma].update(l.lemma for l in lemma.synonyms)

    return synonyms


def get_synonym_S(vocab):
    vocab_ = {w: idx for idx, w in enumerate(vocab)}
    syns = get_synonyms()
    S = np.zeros((len(vocab), len(vocab)))
    np.fill_diagonal(S, 1)
    for w in vocab:
        if w in syns:
            for w2 in syns[w]:
                if w2 in vocab_:
                    S[vocab_[w], vocab_[w2]] = 1 / len(syns[w])

    S += S.T

    return S


def get_random_matrix(vocab):
    S = np.random.normal(loc=0.5, scale=0.05, size=(len(vocab), len(vocab)))
    np.fill_diagonal(S, 1)
    return S


def pairwise_dists(embs):
    """
    Compute cosine distance matrix from every row to every row
    """
    norm = (embs * embs).sum(1, keepdims=True) ** .5
    normed = embs / norm
    return 1 - (normed @ normed.T)


def get_cosine_similarity(src, trg, batch=1000):
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
    return D


def get_cosine_distance(src, trg, **kwargs):
    return 1 - get_cosine_similarity(src, trg, **kwargs)


@contextlib.contextmanager
def timing():
    start = time.time()
    yield
    print(time.time() - start)
