
import os
import numpy as np
import skipthoughts

from textreuse.datasets import default_pairs, opusparcus_pairs
from textreuse.datasets import PATHS, OPUSPARCUS_PATH


if __name__ == '__main__':

    model = skipthoughts.load_model()

    for dataset in PATHS:
        for split in PATHS[dataset]:
            print(dataset, split)

            sents, scores = zip(*default_pairs(PATHS[dataset][split]))
            scores = np.array([float(s) for s in scores])
            s1, s2 = zip(*sents)
            s1, s2 = [' '.join(s) for s in s1], [' '.join(s) for s in s2]
            s1, s2 = skipthoughts.encode(model, s1), skipthoughts.encode(model, s2)

            with open('{}.{}.npz'.format(dataset.lower(), split), 'wb') as f:
                np.savez(f, s1=s1, s2=s2, scores=scores)

    for split in ('train', 'test', 'dev'):
        print("OPUS: ", split)
        sents, scores = zip(*opusparcus_pairs(OPUSPARCUS_PATH, split, maxlines=10000))
        s1, s2 = zip(*sents)
        s1, s2 = [' '.join(s) for s in s1], [' '.join(s) for s in s2]
        s1, s2 = skipthoughts.encode(model, s1), skipthoughts.encode(model, s2)

        with open('{}.{}.npz'.format('opusparcus', split), 'wb') as f:
            if split == 'train':
                np.savez(f, s1=s1, s2=s2)
            else:
                np.savez(f, s1=s2, s2=s2, scores=np.array([float(s) for s in scores]))
