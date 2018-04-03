
import tqdm
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

import seqmod.utils as u


def encode_dataset(encoder, dataset, use_feats):
    """
    Encode pairs to output features
    """
    output_feats, output_labels = None, None

    for batch, labels in tqdm.tqdm(dataset):
        (inp1, len1), (inp2, len2) = batch
        (enc1, _), (enc2, _) = encoder(inp1, lengths=len1), encoder(inp2, lengths=len2)
        enc1, enc2 = enc1.data.cpu().numpy(), enc2.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        feats = np.concatenate([np.abs(enc1 - enc2), enc1 * enc2], axis=1)

        if use_feats:
            d = encoder.embeddings.d
            reserved = set([d.get_bos(), d.get_eos(), d.get_pad(), d.get_unk()])
            A = [[d.vocab[i] for i in s if i not in reserved] for s in inp1.data.t()]
            B = [[d.vocab[i] for i in s if i not in reserved] for s in inp2.data.t()]
            feats = np.concatenate([feats, count_feats(A, B)], axis=1)

        if output_labels is None:
            output_labels = labels
        else:
            output_labels = np.concatenate([output_labels, labels], axis=0)

        if output_feats is None:
            output_feats = feats
        else:
            output_feats = np.concatenate([output_feats, feats], axis=0)

    return output_feats, output_labels


def encode_dataset_debug(path, A, B, labels, use_feats):
    """
    Just for debugging, it loads preprocessed data
    """
    feats = np.load(path)
    if use_feats:
        feats = np.concatenate([feats, count_feats(A, B)], axis=1)
    return feats, np.array(labels)


def count_feats(A, B):
    """
    Compute easy set overlap features.
    (https://github.com/ryankiros/skip-thoughts/blob/master/eval_msrp.py)
    """

    def is_number(w):
        try:
            float(w)
            return True
        except ValueError:
            return False

    features = np.zeros((len(A), 6))

    for i, (tA, tB) in enumerate(zip(A, B)):

        nA = [w for w in tA if is_number(w)]
        nB = [w for w in tB if is_number(w)]

        if set(nA) == set(nB):
            features[i,0] = 1.
    
        if set(nA) == set(nB) and len(nA) > 0:
            features[i,1] = 1.
    
        if set(nA) <= set(nB) or set(nB) <= set(nA): 
            features[i,2] = 1.
    
        features[i,3] = len(set(tA) & set(tB)) / len(set(tA))
        features[i,4] = len(set(tA) & set(tB)) / len(set(tB))
        features[i,5] = 0.5 * ((len(tA) / len(tB)) + (len(tB) / len(tA)))

    return features


def eval_kfold(feats, labels, k=10, shuffle=True):
    kf = KFold(n_splits=k, shuffle=shuffle)
    Cs = [2 ** C for C in range(0, 9)]  # try values
    scores = []

    for C in Cs:
        run_scores = []

        for train, test in kf.split(feats):
            clf = LogisticRegression(C=C).fit(feats[train], labels[train])
            y_hat, y_true = clf.predict(feats[test]), labels[test]
            run_scores.append(f1_score(y_true, y_hat))

        print("C={}; f1={:.4f} (+= {:.4f})".format(
            C, np.mean(run_scores), np.std(run_scores)))

        scores.append(np.mean(run_scores))

    best_C = Cs[np.argmax(scores)]
    print("Best C: {}".format(best_C))
    return best_C


def evaluate(train_X, train_y, test_X, test_y, use_kfold=False, k=10):
    C = 4
    if use_kfold:
        C = eval_kfold(train_X, train_y, k=k)

    clf = LogisticRegression(C=C).fit(train_X, train_y)

    preds = clf.predict(test_X)
    print("F1: {:g}".format(f1_score(test_y, preds)))
    print("Acc: {:g}".format(accuracy_score(test_y, preds)))
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--use_kfold', action='store_true')
    parser.add_argument('--use_feats', action='store_true')
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if not args.debug:
        encoder = u.load_model(args.model).encoder
    
        import utils
        from text_reuse.datasets import load_msrp, MSRP_PATH, default_pairs
        path = (MSRP_PATH + '/msr_paraphrase_{}_tokenized.txt').format
        targets = utils.get_targets(
            encoder.embeddings.d.vocab,
            default_pairs(path('train')), default_pairs(path('test')))
        encoder.embeddings.expand_space(
            '/home/corpora/word_embeddings/w2v.googlenews.300d.bin', targets=targets,
            words=targets + encoder.embeddings.d.vocab)
    
        if args.gpu:
            encoder.cuda()
    
        train, _, test = load_msrp(
            include_valid=False, d=encoder.embeddings.d,
            gpu=args.gpu, batch_size=args.batch_size)
    
        train_X, train_y = encode_dataset(encoder, train, use_feats=args.use_feats)
        test_X, test_Y = encode_dataset(encoder, test, use_feats=args.use_feats)
        evaluate(train_X, train_y, test_X, test_y, use_kfold=args.use_kfold)

    else:
        from text_reuse.datasets import default_pairs, MSRP_PATH
        # train
        train_path = MSRP_PATH + 'msr_paraphrase_train_tokenized.txt'
        X, y = zip(*list(default_pairs(train_path)))
        (A, B) = zip(*X)
        y = np.array(y, dtype=np.float)
        train_X, train_y = encode_dataset_debug(
            '/home/manjavacas/train_msrp.npy', A, B, y, use_feats=args.use_feats)
    
        # test
        test_path = MSRP_PATH + 'msr_paraphrase_test_tokenized.txt'
        X, y = zip(*list(default_pairs(test_path)))
        (A, B) = zip(*X)
        y = np.array(y, dtype=np.float)
        test_X, test_y = encode_dataset_debug(
            'home/manjavacas/test_msrp.npy', A, B, y, use_feats=args.use_feats)
        evaluate(train_X, train_y, test_X, test_y, use_kfold=args.use_kfold)

