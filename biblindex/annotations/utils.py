
import json
import numpy as np
import pie


LATIN = '/home/manjavacas/corpora/word_embeddings/latin.embeddings'


def load_embeddings(words, path=LATIN):
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
            w, freq = line.strip().split()
            freq = int(freq)
            total += freq
            if words and w not in words:
                continue
            freqs[w] = freq

    return {w: freq / total for w, freq in freqs.items()}


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


def pairwise_dists(embs):
    """
    Compute cosine distance matrix from every row to every row
    """
    norm = (embs * embs).sum(1, keepdims=True) ** .5
    normed = embs / norm
    return 1 - (normed @ normed.T)
