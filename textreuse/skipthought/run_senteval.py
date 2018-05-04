
import logging
import os
import yaml
import numpy as np
import senteval
import seqmod.utils as u
from scipy.stats.stats import SpearmanrResult
from scipy.linalg import norm
from textreuse.skipthought.model import SkipThoughts, Loss

SENTEVAL_DATA = '/home/manjavacas/code/vendor/SentEval/data/senteval_data/'

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def types(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            for o in types(v):
                yield o
    elif isinstance(obj, (tuple, list)):
        for i in obj:
            for o in types(i):
                yield o
    else:
        yield type(obj)


def denumpify(obj):
    if isinstance(obj, SpearmanrResult):
        return [float(obj[0]), float(obj[1])]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (tuple, list)):
        return list(denumpify(o) for o in obj)
    elif isinstance(obj, np.float):
        return float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = denumpify(v)
        if 'yhat' in obj:
            del obj['yhat']     # too big
        return obj
    return obj


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--senteval_data', default=SENTEVAL_DATA)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--embeddings',
                        default='/home/corpora/word_embeddings/fasttext.wiki.en.bin')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    model = None

    def prepare(params, samples):
        global model
        model = u.load_model(args.model)
        model.eval()
        model.cpu()
        vocab = set(w for sent in samples for w in sent)
        local = set(model.encoder.embeddings.d.vocab)
        targets, words = list(vocab.difference(local)), list(vocab.union(local))
        model.encoder.embeddings.expand_space(
            args.embeddings, targets=targets, words=words)
        if args.gpu:
            model.cuda()

    def batcher(params, batch):
        return model.encode(batch, use_norm=True)

    params = {
        'task_path': args.senteval_data,
        'usepytorch': True,
        'kfold': 10,
        'batch_size': args.batch_size,
        'classifier': {
            'nhid': 0,
            'optim': 'adam,lr=0.001',
            'batch_size': 50,
            'tenacity': 1,
            'epoch_size': 1
        }
    }

    tasks = [
        # 'TREC',
        # 'MRPC',
        # 'SICKEntailment',
        'SICKRelatedness',
        'STSBenchmark',
        # 'MR',
        # 'CR',
        # 'SUBJ',
        # 'MPQA',
        # # 'SNLI',                 # super memory inefficient, leave out for now
        # 'STS14',
    ]

    dirname = os.path.dirname(args.model)
    basename = '.'.join(os.path.basename(args.model).split('.')[:-1]) + ".results.yml"
    fname = os.path.join(dirname, basename)

    if not args.test and os.path.exists(fname):
        print("Eval file already exists")
        import sys
        sys.exit(0)

    results = senteval.engine.SE(params, batcher, prepare).eval(tasks)

    if not args.test:
        with open(fname, 'w') as f:
            yaml.dump(denumpify(results), stream=f, default_flow_style=False)
