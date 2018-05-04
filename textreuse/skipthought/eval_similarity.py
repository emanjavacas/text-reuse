
import os
import tqdm
import copy

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import seqmod.utils as u
from keras.models import Sequential
from keras.layers import Activation, Dense

from textreuse.skipthought.model import SkipThoughts, Loss
from textreuse.datasets import encode_label


def encode_dataset(model, A, B, labels, use_norm=False):
    """
    Encode pairs to output features
    """
    enc1, enc2 = model.encode(A, use_norm=use_norm), model.encode(B, use_norm=use_norm)

    feats = np.concatenate([np.abs(enc1 - enc2), enc1 * enc2], axis=1)

    return feats, np.array([encode_label(l) for l in labels], dtype=np.float)


def make_model(encoding_size, nclass):
    m = Sequential([Dense(nclass, input_dim=encoding_size * 2),
                    Activation("softmax")])
    m.compile(loss="kullback_leibler_divergence", optimizer="adam")

    return m


def evaluate(model, X, y, nclass=5):
    scores = np.dot(y, np.arange(1, nclass + 1))
    preds = np.dot(model.predict_proba(X, verbose=2), np.arange(1, nclass + 1))
    p, _ = pearsonr(preds, scores)
    s, _ = spearmanr(preds, scores)
    mse = mean_squared_error(preds, scores)

    return p, s, mse
    

def train_model(train_X, train_y, dev_X, dev_y, encoding_size, nclass=5, max_epochs=1000):
    model = make_model(encoding_size, nclass)
    done, best, epochs = False, -1.0, 0

    while not done:
        epochs += 50
        model.fit(train_X, train_y, epochs=50, verbose=2, shuffle=False,
                  validation_data=(dev_X, dev_y))
        p, s, mse = evaluate(model, dev_X, dev_y, nclass=nclass)
        print("Validation pearsonr: {:g}".format(p))
        print("Validation spearmanr: {:g}".format(s))
        print("Validation MSE: {:g}".format(mse))

        if p > best:
            best_model = make_model(encoding_size, nclass)
            best_model.set_weights(model.get_weights())
            best = p
        else:
            done = True

        if epochs > max_epochs:
            break

    return best_model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--use_norm', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--embeddings',
                        default='/home/corpora/word_embeddings/fasttext.wiki.en.bin')
    args = parser.parse_args()

    import utils
    from textreuse.datasets import default_pairs, PATHS

    model = u.load_model(args.model)
    model.eval()

    targets = utils.get_targets(
        model.encoder.embeddings.d.vocab,
        default_pairs(PATHS['SICK']['train']),
        default_pairs(PATHS['SICK']['dev']),
        default_pairs(PATHS['SICK']['test']),
        default_pairs(PATHS['STSB']['train']),
        default_pairs(PATHS['STSB']['dev']),
        default_pairs(PATHS['STSB']['test']))

    model.encoder.embeddings.expand_space(
        args.embeddings,
        targets=targets, words=targets + model.encoder.embeddings.d.vocab)

    if args.gpu:
        model.cuda()

    for task in ('SICK', 'STSB'):
        
        # train
        X, y = zip(*list(default_pairs(PATHS[task]['train'])))
        (A, B) = zip(*X)
        train_X, train_y = encode_dataset(model, A, B, y, use_norm=args.use_norm)

        # dev
        X, y = zip(*list(default_pairs(PATHS[task]['dev'])))
        (A, B) = zip(*X)
        dev_X, dev_y = encode_dataset(model, A, B, y, use_norm=args.use_norm)
        best_model = train_model(
            train_X, train_y, dev_X, dev_y, model.encoder.encoding_size[1])

        # test
        X, y = zip(*list(default_pairs(PATHS[task]['test'])))
        (A, B) = zip(*X)
        test_X, test_y = encode_dataset(model, A, B, y, use_norm=args.use_norm)
        p, s, mse = evaluate(best_model, test_X, test_y)
        print("Test pearsonr: {:g}".format(p))
        print("Test spearmanr: {:g}".format(s))
        print("Test MSE: {:g}".format(mse))

        if not args.test:
            results_path = os.path.join(
                os.path.dirname(args.model),
                '{}.csv'.format(task.lower()))
            formatter = "\nModel: {}\tPearson: {:g}\tSpearman: {:g}" + \
                        "\tMSE: {:g}\tNorm: {}\tEmbs: {}"
            with open(results_path, 'a') as f:
                f.write(formatter.format(
                    os.path.basename(args.model), p, s, mse,
                    str(args.use_norm), os.path.basename(args.embeddings)))
