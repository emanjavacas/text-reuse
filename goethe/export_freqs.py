
import numpy as np
import pickle
import collections


if __name__ == '__main__':

    vocab = collections.Counter()
    with open('goethe-gold.csv') as f:
        for line in f:
            _, _, s1, s2, _, _ = line.strip().split('\t')
            vocab.update(s1.lower().split())
            vocab.update(s2.lower().split())

    with open('goethe-background.csv') as f:
        for line in f:
            _, s1, _ = line.strip().lower().split('\t')
            vocab.update(s1.split())

    with open('./resources/avg_freqs.pkl', 'rb') as f:
        freqs = pickle.load(f, encoding='latin1')

    # missing = set(vocab).difference(set(freqs))

    with open('goethe.freqs', 'w') as f:
        min_freq = float(min(freqs.values()))
        for w in vocab:
            f.write('\t'.join([w, str(freqs.get(w, min_freq))]) + '\n')

    embs = np.load('./resources/1800-w.npy')
    with open('./resources/1800-vocab.pkl', 'rb') as f:
        wvocab = pickle.load(f)

    with open('goethe.embeddings', 'w') as f:
        for w, emb in zip(wvocab, embs):
            if w not in vocab:
                continue
            f.write(w + '\t' + ' '.join(map(str, list(emb))) + '\n')
