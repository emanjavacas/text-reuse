
# for SIF, see: https://github.com/PrincetonML/SIF

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def remove_pc(W, npc=1):
    pc = TruncatedSVD(n_components=npc, n_iter=7).fit(W).components_
    if npc == 1:
        return W - W.dot(pc.transpose()) * pc
    else:
        return W - W.dot(pc.transpose()).dot(pc)


class SIF:
    def __init__(self, W, words, freqs, npc=1, a=1e-3):
        self.npc = npc
        self.a = a
        # attributes
        self.W = W
        self.words = {w: idx for idx, w in enumerate(words)}
        self.words_ = {idx: w for w, idx in self.words.items()}
        self.freqs = freqs

    def get_weight(self, w):
        if w not in self.words:
            raise KeyError(w)
        return self.a / (self.a + self.freqs[self.words[w]])

    def transform(self, sents):
        sents = [[self.words[w] for w in sent if w in self.words] for sent in sents]
        # get average
        sents = np.array([
            sum(self.get_weight(self.words_[w]) * self.W[w] for w in sent) / len(sent)
            for sent in sents])
        sents = remove_pc(sents, npc=self.npc)
        return sents


class TFIDF:
    def __init__(self, W, words, sents, **kwargs):
        # attributes
        self.W = W
        self.words = {w: idx for idx, w in enumerate(words)}
        self.tfidf = TfidfVectorizer().fit(' '.join(s) for s in sents)

    def transform(self, sents):
        sents = [' '.join([w for w in s if w in self.words]) for s in sents]
        X = self.tfidf.transform(sents)
        idx2w = self.tfidf.get_feature_names()
        for s_idx in range(len(sents)):
            sents[s_idx] = sum(
                weight * self.W[self.words[idx2w[i]]]
                for (_, i), weight in X[s_idx].todok().items()
            ) / len(sents[s_idx])

        return np.array(sents)


class BOW:
    def __init__(self, W, words, **kwargs):
        # attributes
        self.W = W
        self.words = {w: idx for idx, w in enumerate(words)}

    def transform(self, sents):
        sents = [[self.words[w] for w in sent if w in self.words] for sent in sents]
        sents = [sum(self.W[w] for w in s) / len(s) for s in sents]
        return np.array(sents)
