
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

from textreuse.datasets import encode_label, PATHS


def eval_similarity(s1, s2, scores, nclass=5, report=False):
    """
    Evaluate the similarity in terms of correlation of cosine similarity

    Parameters
    ===========
    s1 : (batch x encoding_size), normed embeddings
    s1 : (batch x encoding_size), normed embeddings
    scores : (batch, nclass)
    """
    if scores.ndim > 1:
        scores = np.dot(scores, np.arange(1, nclass + 1))

    # cosine similarity
    s1 = s1 / np.sqrt((s1 ** 2).sum(1))[:, None]
    s2 = s2 / np.sqrt((s2 ** 2).sum(1))[:, None]
    sims = np.sum(s1 * s2, axis=1)

    p, _ = pearsonr(sims, scores)
    s, _ = spearmanr(sims, scores)
    mse = mean_squared_error(sims * 5, scores)

    if report:
        print("Pearson-R: {:g}".format(p))
        print("Spearman-R: {:g}".format(s))
        print("MSE: {:g}".format(mse))
        print()

    return p, s, mse
    

def load_dataset(dataset, split):
    """
    Load encoded dataset
    """
    with open('{}.{}.npz'.format(dataset.lower(), split), 'rb') as f:
        data = dict(np.load(f).items())

    return data


def pearson_correlation(pred, true):
    """
    PyTorch implementation of Pearson
    """
    mpred = pred.unsqueeze(0) - pred.mean()
    mtrue = true.unsqueeze(0) - true.mean()
    output = (mpred * mtrue).sum(1)
    mpred = torch.sqrt(torch.sum(mpred ** 2, 1))
    mtrue = torch.sqrt(torch.sum(mtrue ** 2, 1))
    return output / (mpred * mtrue)


class Model(nn.Module):
    """
    Learning a projection matrix to fine tune pretrained sentence embeddings
    on supervised data.
    """
    def __init__(self, encoding_size):
        self.encoding_size = encoding_size
        super().__init__()

        self.W = nn.Linear(encoding_size, encoding_size, bias=None)
        for p_name, p in self.named_parameters():
            if 'bias' in p_name:
                nn.init.constant(p, 0.)
            else:
                nn.init.uniform(p, -0.1, 0.1)

    def forward(self, s1, s2):
        """
        This is basically the cosine similarity, which we aim to maximize correlation
        """
        return nn.functional.cosine_similarity(self.W(s1), self.W(s2))

    def train_model(self, train, dev, epochs, cuda=False):
        """
        Model trainer
        """
        s1, s2, scores = train['s1'], train['s2'], train['scores']
        s1_dev, s2_dev, scores_dev = dev['s1'], dev['s2'], dev['scores']
        optim = torch.optim.SGD(self.parameters(), lr=0.05, weight_decay=0.75)
        t = torch.cuda if cuda else torch
        best_corr, best_W = 0, torch.zeros(self.encoding_size, self.encoding_size)

        for epoch in range(epochs):
            optim.zero_grad()
            dist = self(Variable(t.FloatTensor(s1)), Variable(t.FloatTensor(s2)))
            corr = pearson_correlation(dist, Variable(t.FloatTensor(scores)))
            # minimize the negative correlation
            torch.autograd.backward(-corr)
            optim.step()

            if epoch % 1 == 0:
                self.eval()
                pred_s1 = self.W(Variable(t.FloatTensor(s1_dev), volatile=True))
                pred_s2 = self.W(Variable(t.FloatTensor(s2_dev), volatile=True))
                p, _, _ = eval_similarity(
                    pred_s1.data.cpu().numpy(), pred_s2.data.cpu().numpy(), scores_dev,
                    report=True)

                if p >= best_corr:
                    best_corr = p
                    best_W = self.W.weight.data.clone().cpu()
                    self.train()

                else:
                    break

        return best_corr, best_W.numpy()

if __name__ == '__main__':
    joined = True
    train, dev = load_dataset('stsb', 'train'), load_dataset('stsb', 'dev')
    if joined:                      # concatenate both datasets
       sick = load_dataset('sick', 'train')
       train['s1'] = np.concatenate([train['s1'], sick['s1']])
       train['s2'] = np.concatenate([train['s2'], sick['s2']])
       train['scores'] = np.concatenate([train['scores'], sick['scores']])
       sick = load_dataset('sick', 'dev')
       dev['s1'] = np.concatenate([dev['s1'], sick['s1']])
       dev['s2'] = np.concatenate([dev['s2'], sick['s2']])
       dev['scores'] = np.concatenate([dev['scores'], sick['scores']])
    m = Model(train['s1'].shape[1])
    m.cuda()
    
    best_corr, best_W = m.train_model(train, dev, 1000, cuda=True)
    
    for dataset in ('stsb', 'sick'):
        for split in ('test', 'dev'):
            print("::: {}-{} {} :::".format(dataset, split, 'projected'))
            data = load_dataset(dataset, split)
            eval_similarity(np.einsum('ij,bj->bi', best_W, data['s1']),
                            np.einsum('ij,bj->bi', best_W, data['s2']),
                            data['scores'],
                            report=True)
    
            print("::: {}-{} {} :::".format(dataset, split, 'raw'))
            eval_similarity(data['s1'], data['s2'], data['scores'], report=True)

"""
::: stsb-test projected :::
Pearson-R: 0.632943
Spearman-R: 0.600382
MSE: 1.55824

::: stsb-test raw :::
Pearson-R: 0.196424
Spearman-R: 0.307296
MSE: 4.37622

::: stsb-dev projected :::
Pearson-R: 0.70111
Spearman-R: 0.708331
MSE: 1.31806

::: stsb-dev raw :::
Pearson-R: 0.287251
Spearman-R: 0.396399
MSE: 4.3603

::: sick-test projected :::
Pearson-R: 0.615918
Spearman-R: 0.566827
MSE: 0.763796

::: sick-test raw :::
Pearson-R: 0.49928
Spearman-R: 0.479841
MSE: 1.02652

::: sick-dev projected :::
Pearson-R: 0.590614
Spearman-R: 0.563922
MSE: 0.745335

::: sick-dev raw :::
Pearson-R: 0.481998
Spearman-R: 0.472748
MSE: 1.05241
"""

# - English & Latin
# - CBOW & Skipthought & fine-tuned CBOW & fine-tuned Skipthought
# - normalized recall of content words
# - depending of sentence position
# - depending on vocabulary rank
