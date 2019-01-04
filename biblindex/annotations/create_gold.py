
import os
import statistics

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


def load_source():
    by_user = {user: extract.load_annotations(user) for user in extract.DATASETS}
    shared = extract.get_shared(by_user)

    # remove controversial as per user notes
    for user in by_user:
        notes = load_notes(user)
        by_user[user] = {i: doc for i, doc in by_user[user].items() if i not in notes}
    docs = []

    for i in shared:
        docs.append(select(*[by_user[user][i] for user in by_user if i in by_user[user]]))

    for user in by_user:
        for idx, doc in by_user[user].items():
            if idx in shared:
                continue

            docs.append(doc)

    return docs


if __name__ == '__main__':
    import pie
    model = pie.SimpleModel.load("./capitula.model.tar")
    source, bible = load_source(), utils.load_bible()

    with open("gold.csv", "w") as f:
        for doc in source:
            src, trg = doc['selectedText'], bible[doc['id']]
            src_id, trg, trg_id = doc['id'], trg['text'], trg['url']
            if len(src.split()) > 40:  # ignore longer than 40 words
                continue
            src_lemma = ' '.join(utils.lemmatize(model, src.lower().split())['lemma'])
            trg_lemma = ' '.join(utils.lemmatize(model, trg.lower().split())['lemma'])
            f.write('\t'.join([src_id, trg_id, src, trg, src_lemma, trg_lemma]) + '\n')
