
import json
import os
import statistics

from cltk.tokenize.latin.sentence import SentenceTokenizer
try:
    tok = SentenceTokenizer()
except:
    print("Couldn't find cltk sentence tokenizer")
    tok = None

import extract
import utils

# source data
def select(*docs):
    # select the one closest to the median/mean
    lens = [(len(doc['selectedText'].split()), idx) for idx, doc in enumerate(docs)]
    median = statistics.median([l for l, _ in lens])
    idx = next(iter(sorted([(abs(l - median), idx) for l, idx in lens])))[1]
    return docs[idx]


def load_notes(user):
    fpath = "{}.notes.csv".format(user)
    if os.path.isfile(fpath):
        with open(fpath) as f:
            return set([line.split(',')[0] for line in f])
    return set()


def load_docs():
    by_user = {user: extract.load_annotations(user) for user in extract.DATASETS}
    shared = extract.get_shared(by_user)

    # remove controversial as per user notes
    for user in by_user:
        notes = load_notes(user)
        by_user[user] = {i: doc for i, doc in by_user[user].items() if i not in notes}
    docs = []

    for i in shared:
        docs.append(
            select(*[by_user[user][i] for user in by_user if i in by_user[user]]))

    for user in by_user:
        for idx, doc in by_user[user].items():
            if idx in shared:
                continue

            docs.append(doc)

    return docs


def load_source(path='../splits/SCT1-5.json'):
    with open(path) as f:
        source = {}
        for line in f:
            doc = json.loads(line)
            source[doc['id']] = doc

    return source


def enrich_with_lemmas_from_source(docs):
    source = load_source()

    for doc in docs:
        sdoc = source[doc['id']]
        text = [w['lemma'] for w in sdoc['textdata']]
        text = [w['lemma'] for w in sdoc['textcontext']['prev'][-doc['prevSpan']:]] + text
        text += [w['lemma'] for w in sdoc['textcontext']['next'][:doc['nextSpan']]]
        doc['text_lemma'] = ' '.join(text)
        assert len(doc['text_lemma'].split()) == len(doc['selectedText'].split())

    return docs


def window_sentence(sdoc, window):
    src = [w['word'] for w in sdoc['textcontext']['prev'][-window:]] + \
          [w['word'] for w in sdoc['textdata']] + \
          [w['word'] for w in sdoc['textcontext']['next'][:window]]
    src = ' '.join(src)
    src_lemma = [w['lemma'] for w in sdoc['textcontext']['prev'][-window:]] + \
                [w['lemma'] for w in sdoc['textdata']] + \
                [w['lemma'] for w in sdoc['textcontext']['next'][:window]]
    src_lemma = ' '.join(src_lemma)

    return src, src_lemma


def sentence_tokenize(sdoc, tok):
    start = len(sdoc['textcontext']['prev'])  # index of 1st in
    end = start + len(sdoc['textdata'])  # index of 1st outside
    sents = tok.tokenize(' '.join(
        [w['word'] for w in sdoc['textcontext']['prev']] +
        [w['word'] for w in sdoc['textdata']] +
        [w['word'] for w in sdoc['textcontext']['next']]))
    sums = [0]
    for s in sents:
        sums.append(sums[-1] + len(s.split()))
    s_start = s_end = None
    for idx, s in enumerate(sums):
        if s >= start and s_start is None:
            s_start = idx - 1
        if s >= end:
            s_end = idx
            break
    
    src = ' '.join(sents[s_start:s_end])
    src_lemma = [w['lemma'] for w in sdoc['textcontext']['prev']] + \
                [w['lemma'] for w in sdoc['textdata']] + \
                [w['lemma'] for w in sdoc['textcontext']['next']]

    src_lemma = ' '.join(src_lemma[sums[s_start]: sums[s_end]])

    assert len(src.split()) == len(src_lemma.split())

    return src, src_lemma


if __name__ == '__main__':
    import pie
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--lemma_model', default="./capitula.model.tar")
    args = parser.parse_args()

    model = pie.SimpleModel.load(args.lemma_model)
    docs, bible = load_docs(), utils.load_bible()
    docs = enrich_with_lemmas_from_source(docs)
    source = load_source()

    gold, window = "bernard-gold.csv", "bernard-gold-window-{}.csv".format(args.window)
    sentence = "bernard-gold-sentence.csv"
    with open(gold, "w") as f, open(window, "w") as f2, open(sentence, "w") as f3:
        for doc in docs:
            src, trg = doc['selectedText'], bible[doc['id']]
            src_id, trg, trg_id = doc['id'], trg['text'], trg['url']
            if len(src.split()) > 40:  # ignore longer than 40 words
                continue
            src_lemma = doc['text_lemma']
            trg_lemma = ' '.join(utils.lemmatize(model, trg.lower().split())['lemma'])
            f.write('\t'.join([src_id, trg_id, src, trg, src_lemma, trg_lemma]) + '\n')

            # window
            src, src_lemma = window_sentence(source[doc['id']], args.window)
            f2.write('\t'.join([src_id, trg_id, src, trg, src_lemma, trg_lemma]) + '\n')

            # sentence
            if tok is None:
                print("Not compiling sentence tokenized version")
                continue
            src, src_lemma = sentence_tokenize(source[doc['id']], tok)
            f3.write('\t'.join([src_id, trg_id, src, trg, src_lemma, trg_lemma]) + '\n')
