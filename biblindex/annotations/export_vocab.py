
import collections

import utils

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_path', default='bernard-gold.csv')
    parser.add_argument('--background_path', default='bernard-background.csv')
    parser.add_argument('--outputname', default='bernard.words')
    args = parser.parse_args()

    src, trg = utils.load_gold(path=args.gold_path)
    trg += utils.load_background(path=args.background_path)
    vocab = collections.Counter(w for s in src + trg for w in s)
    src, trg = utils.load_gold(path=args.gold_path, lemmas=True)
    trg += utils.load_background(path=args.background_path, lemmas=True)
    for s in src + trg:
        for w in s:
            vocab[w] += 1

    with open(args.outputname, 'w+') as f:
        for w, _ in vocab.most_common():
            f.write(w + '\n')
