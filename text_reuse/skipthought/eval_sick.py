
import copy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

import seqmod.utils as u
from seqmod.misc import Trainer, EarlyStopping, StdLogger


def cross_entropy_with_logits(output, targets):
    "Cross entropy with logits for continuous (soft) targets"
    return torch.mean(torch.sum(-targets * F.log_softmax(output, 1), 1))


def pearsonr_hook(dataset, early_stopping):

    def hook(trainer, epoch, batch, checkpoint):
        p, s, mse = trainer.model.validate(dataset)
        trainer.log("info", "Validation pearsonr: {:g}".format(p))
        trainer.log("info", "Validation spearmanr: {:g}".format(s))
        trainer.log("info", "Validation MSE: {:g}".format(mse))
        # early stop on MSE probably instead of pearsonr?
        early_stopping.add_checkpoint(mse, copy.deepcopy(trainer.model).cpu())

    return hook


class LogisticRegression(nn.Module):
    """
    Simple Soft LogisticRegression classifier on Skipthought output encodings
    """
    def __init__(self, encoder, nclass=5):
        self.nclass = nclass
        super(LogisticRegression, self).__init__()

        self.encoder = encoder
        # don't train skipthought encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.ff = nn.Linear(encoder.encoding_size[1] * 2, nclass)

    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p

    def forward(self, inp1, inp2):
        (inp1, len1), (inp2, len2) = inp1, inp2
        (enc1, _), (enc2, _) = self.encoder(inp1, len1), self.encoder(inp2, len2)
        return self.ff(torch.cat([torch.abs(enc1 - enc2), enc1 * enc2], 1))

    def loss(self, batch_data, test=False):
        (inp1, inp2), labels = batch_data
        xent = cross_entropy_with_logits(self(inp1, inp2), labels)
        if not test:
            xent.backward()

        return (xent.data[0], ), len(labels)

    def predict(self, inp1, inp2):
        probs = F.softmax(self(inp1, inp2), dim=1)
        return probs.data.cpu() @ torch.arange(1, self.nclass + 1)

    def train_model(self, train, valid, epochs, checkpoint=10, lr=0.001):
        optimizer = optim.Adam(list(self.trainable_parameters()), lr=lr)
        trainer = Trainer(self, {'train': train}, optimizer)
        trainer.add_loggers(StdLogger())
        trainer.add_hook(pearsonr_hook(valid, EarlyStopping(3, maxsize=2)),
                         hooks_per_epoch=1)
        (best_model, _), _ = trainer.train(epochs, checkpoint, shuffle=True)

        return best_model

    def validate(self, dataset):
        preds, scores = [], []
        for (inp1, inp2), labels in dataset:
            preds.extend(self.predict(inp1, inp2).cpu().tolist())
            labels = labels.data.cpu() @ torch.arange(1, self.nclass + 1)
            scores.extend(labels.tolist())

        preds, scores = np.array(preds), np.array(scores)

        p, _ = pearsonr(preds, scores)
        s, _ = spearmanr(preds, scores)
        mse = mean_squared_error(preds, scores)

        return p, s, mse


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--epochs', default=10, type=int)

    args = parser.parse_args()

    lr = LogisticRegression(u.load_model(args.model).encoder)
    d = lr.encoder.embeddings.d

    from text_reuse.datasets import load_sick
    train, valid, test = load_sick(batch_size=args.batch_size, d=d, gpu=args.gpu)

    if args.gpu:
        lr.cuda()

    best_model = lr.train_model(train, valid, args.epochs)
    if args.gpu:
        best_model.cuda()

    p, s, mse = best_model.validate(test)
    print("Pearsonr: {:g}".format(p))
    print("Spearmanr: {:g}".format(s))
    print("MSE: {:g}".format(mse))
