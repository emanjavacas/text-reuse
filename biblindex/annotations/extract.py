
import itertools
import os
import tinydb
import difflib

ROOT = './data/source'
DATASETS = {
    'dinah':    'd{}.json',
    'jeroenDP': 'jdp{}.json',
    'jeroenDG': 'jdg{}.json'
}
PAIRS = list(itertools.combinations(list(DATASETS), 2))


def check_dups(data):
    seen = set()
    for doc in data:
        if doc['id'] in seen:
            raise ValueError("Found dup: {}".format(id))
        else:
            seen.add(doc['id'])


def check_nums(data):
    if len(data) != 279:
        print("Got {} but expected 279".format(len(data)))
        # raise


def load_annotations(user):
    data = []
    for subset in ['1', '2']:
        path = os.path.join(ROOT, DATASETS[user].format(subset))
        data += list(map(dict, tinydb.TinyDB(path).all()))
    data = [doc for doc in data if doc['type'] == 'annotation']

    check_dups(data)
    check_nums(data)

    data = {doc['id']: doc for doc in data}

    return data


def get_shared(users):
    ids = {user: set(docs) for user, docs in users.items()}
    names = list(users)
    shared = None
    for i in range(len(users)):
        user1 = names[i]
        for j in range(i + 1, len(users)):
            user2 = names[j]
            if shared is None:
                shared = ids[user1].intersection(ids[user2])
            else:
                assert len(shared) == 80 and shared == ids[user1].intersection(ids[user2])

    return list(shared)


def LCS(a, b):
    return difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))


def overlap(a, b):
    a, b = a.split(), b.split()
    lcs = LCS(a, b).size
    return lcs / ((len(a) + len(b)) - lcs)


def average_overlap(user1, user2, ids):
    scores = total = 0
    for did in ids:
        total += 1
        scores += overlap(user1[did]['selectedText'], user2[did]['selectedText'])
    return scores / total


# users = {user: load_annotations(user) for user in DATASETS}
# ids = get_shared(users)
# overlaps = {(u1, u2): average_overlap(users[u1], users[u2], ids) for (u1, u2) in PAIRS}
