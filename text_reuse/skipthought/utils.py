

def get_targets(vocab, *pairs):
    targets = []
    vocab = set(vocab)

    for ps in pairs:
        for (p1, p2), _ in ps:
            for w in p1 + p2:
                if w not in vocab:
                    targets.append(w)

    return list(set(targets))

