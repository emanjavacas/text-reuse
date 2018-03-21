
import tqdm
import numpy as np

from text_reuse.datasets import load_msrp


def encode_dataset(encoder, dataset):
    output_feats, output_labels = None, None

    for batch, labels in tqdm(dataset):
        (inp1, len1), (inp2, len2) = batch
        (enc1, _), (enc2, _) = encoder(inp1, lengths=len1), encoder(inp2, lengths=len2)
        feats = torch.cat([torch.abs(enc1 - enc2), enc1 * enc2], dim=1)

        if output_labels is None:
            output_labels = labels.data.cpu()
        else:
            output_labels = torch.cat([output_labels, labels.data.cpu()], dim=0)

        if output_feats is None:
            output_feats = feats.data.cpu()
        else:
            output_feats = torch.cat([output_feats, feats.data.cpu()], dim=1)

    return output_feats.numpy(), output_labels.numpy()


def feats(dataset):
    """
    Compute easy features (from https://github.com/ryankiros/skip-thoughts/blob/master/eval_msrp.py)
    """

    def is_number(w):
        try:
            float(w)
            return True
        except ValueError:
            return False

    (A, B), (d, _) = zip(*dataset.data['src']), dataset.d['src']
    bos, eos = d.get_bos(), d.get_eos()
    features = np.zeros((len(A), 6))

    for i, (tA, tB) in enumerate(zip(A, B)):

        if bos is not None:
            tA, tB = tA[1:], tB[1:]
        if eos is not None:
            tA, tB = tA[:-1], tB[:-1]

        nA = [w for w in tA if is_number(d.vocab[w])]
        nB = [w for w in tB if is_number(d.vocab[w])]

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


from text_reuse.datasets import load_msrp
train, valid, test = load_msrp()

