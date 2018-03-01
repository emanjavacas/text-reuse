
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

from seqmod.modules.embedding import Embedding
from seqmod.modules.encoder import MaxoutWindowEncoder
from seqmod.modules.rnn_encoder import RNNEncoder
from seqmod.modules.cnn_encoder import CNNEncoder
from seqmod.modules.cnn_text_encoder import CNNTextEncoder
from seqmod.modules.ff import Highway
from seqmod.misc import Trainer, StdLogger, EarlyStopping
import seqmod.utils as u


class NormObjective(nn.Module):
    """
    Interpret the norm of the differences as logits of the input pair being similar.
    """
    def __init__(self, norm=1):
        self.norm = norm
        super(NormObjective, self).__init__()

    def forward(self, enc1, enc2, p=0.0):
        output = F.dropout(enc1 - enc2, p=p, training=self.training)
        return torch.exp(-torch.norm(output, self.norm, dim=1))

    def loss(self, pred, labels):
        return F.binary_cross_entropy(pred, labels)

    def predict(self, pred):
        return F.sigmoid(pred) > 0.5


class SigmoidObjective(nn.Module):
    """
    Project to a single unit that is interpreted as the sigmoid logit of the
    input pair being similar. Defined by Chen 2013 as Gaussian similarity model.
    """
    def __init__(self, encoding_size):
        super(SigmoidObjective, self).__init__()
        self.logits = nn.Linear(encoding_size, 1, bias=False)

    def forward(self, enc1, enc2, p=0.0):
        logits = (enc1 - enc2) ** 2
        logits = F.dropout(logits, p=p, training=self.training)
        return self.logits(logits).squeeze(1)  # (batch x 1) => (batch)

    def loss(self, pred, labels):
        return F.binary_cross_entropy_with_logits(pred, labels)

    def predict(self, pred):
        return F.sigmoid(pred) > 0.5


class CauchyObjective(nn.Module):
    def __init__(self, encoding_size):
        super(CauchyObjective, self).__init__()
        self.logits = nn.Linear(encoding_size, 1, bias=False)

    def forward(self, enc1, enc2, p=0.0):
        logits = (enc1 - enc2) ** 2
        logits = F.dropout(logits, p=p, training=self.training)
        return 1 / (1 + self.logits(logits).clamp(min=0).squeeze(1))

    def loss(self, pred, labels):
        return F.binary_cross_entropy(pred, labels)

    def predict(self, pred):
        return pred > 0.5


class ContrastiveCosineObjective(nn.Module):
    """
    Implementation from "Learning Text Similarity with Siamese Recurrent Networks"
    http://www.aclweb.org/anthology/W16-1617
    It diverges from original "contrastive loss" definition by Chopra et al 2005,
    based on euclidean distance instead of cosine (which makes sense for text).

    L_+(x_1, x_2, y) = 1/4 * (1 - cosine_similarity(x_1, x_2)) ^ 2
    L_-(x_1, x_2, y) = max(0, cosine_similarity(x_1, x_2) - margin) ^ 2

    where 1/4 is the weight which should be tune with respect to the proportion
    of positive versus negative examples.
    """
    def __init__(self, weight=0.5, margin=0.2):
        self.margin = margin
        self.weight = weight
        super(ContrastiveCosineObjective, self).__init__()

    def forward(self, enc1, enc2, p=0.0):
        return F.cosine_similarity(enc1, enc2, dim=1)

    def loss(self, sims, y):
        # cosine distance scaled down by negative sampling factor (1/4)
        pos = self.weight * (1 - sims) ** 2
        # cosine similarity must be larger than a margin for negative examples
        neg = torch.clamp(sims - self.margin, min=0) ** 2

        return torch.mean(y * pos + (1 - y) * neg)

    def predict(self, pred):
        return pred


class ContrastiveEuclideanObjective(nn.Module):
    """
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    L_+(x_1, x_2, y) = 1/2 * euclidean_dist(x_1, x_2) ^ 2
    L_-(x_1, x_2, y) = 1/2 * max(0, margin - euclidean_dist(x_1, x_2)) ^ 2
    """
    def __init__(self, weight=0.5, margin=2):
        self.weight = weight
        self.margin = margin
        super(ContrastiveEuclideanObjective, self).__init__()

    def forward(self, enc1, enc2, p=0.0):
        return F.pairwise_distance(enc1, enc2, p=p)

    def loss(self, dists, y):
        # euclidean distance must be low for positive examples
        pos = weight * dists ** 2
        # euclidean distance must at least as large as margin for negative examples
        neg = torch.clamp(self.margin - dists, min=0) ** 2

        return torch.mean(y * pos + (1- y) * neg)

    def predict(self, pred):
        return pred


class Siamese(nn.Module):
    def __init__(self, encoder, loss='sigmoid', proj_layers=0, dropout=0.2, **kwargs):
        self.dropout = dropout
        super(Siamese, self).__init__()

        self.encoder = encoder

        self.proj = None
        if proj_layers > 0:
            self.proj = Highway(encoding_size, num_layers=proj_layers, dropout=dropout)

        if loss == 'L1':
            self.objective = NormObjective(norm=1)
        elif loss == 'L2':
            self.objective = NormObjective(norm=2)
        elif loss == 'sigmoid':
            self.objective = SigmoidObjective(encoder.encoding_size[1])
        elif loss == 'cosine':
            self.objective = ContrastiveCosineObjective(**kwargs)
        elif loss == 'euclidean':
            self.objective = ContrastiveEuclideanObjective(**kwargs)
        elif loss == 'cauchy':
            self.objective = CauchyObjective(encoder.encoding_size[1])

    def forward(self, p1, p2, lengths=(None, None)):
        enc1 = self.encoder(p1, lengths=lengths[0])
        enc2 = self.encoder(p2, lengths=lengths[1])
        if isinstance(enc1, tuple):
            enc1, enc2 = enc1[0], enc2[0]  # some encoders also return hidden

        if self.proj is not None:
            enc1, enc2 = self.proj(enc1), self.proj(enc2)

        return self.objective(enc1, enc2, p=self.dropout)

    def loss(self, batch_data, test=False):
        ((pair1, pair1_len), (pair2, pair2_len)), labels = batch_data
        labels = labels.float()  # transform to float
        num_examples = sum(pair1_len + pair2_len).data[0]

        pred = self(pair1, pair2, lengths=(pair1_len, pair2_len))
        loss = self.objective.loss(pred, labels)

        if not test:
            loss.backward()

        return (loss.data[0],), num_examples

    def predict(self, pair1, pair2, lengths=(None, None)):
        return self.objective.predict(self(pair1, pair2, lengths=lengths))

    def evaluate(self, dataset):
        preds, trues = [], []

        for batch, label in dataset:
            (p1, p1_len), (p2, p2_len) = batch
            pred = self.predict(p1, p2, lengths=(p1_len, p2_len))
            trues.extend(label.data.tolist())
            preds.extend(pred.data.tolist())

        return trues, preds


def make_classification_hook(patience):

    early_stopping = None
    if patience > 0:
        early_stopping = EarlyStopping(patience)

    def hook(trainer, epoch, batch, checkpoint):
        # accuracy
        trues, preds = trainer.model.evaluate(trainer.datasets['valid'])
        if isinstance(trainer.model.objective, ContrastiveCosineObjective) or \
           isinstance(trainer.model.objective, ContrastiveEuclideanObjective):
            r, pval = pearsonr(np.array(preds), np.array(trues).astype(np.float))
            trainer.log("info", "Correlation {:.3f}".format(r))
        else:
            metric = accuracy_score(trues, preds)
            trainer.log("info", "Accuracy {:.3f}".format(metric))
        early_stopping.add_checkpoint(1 - metric)  # the lower the better

    return hook

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--corpus', default='msrp')
    parser.add_argument('--dev', default=0.1, type=float)
    # model
    parser.add_argument('--model', default='rnn')
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM', type=str)
    parser.add_argument('--emb_dim', default=100, type=int)
    parser.add_argument('--hid_dim', default=50, type=int)
    parser.add_argument('--ktop', default=1, type=int)
    parser.add_argument('--encoder_summary', default='inner-attention')
    parser.add_argument('--objective', default='manhattan')
    parser.add_argument('--proj_layers', default=0, type=int)
    # training
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--init_embeddings', action='store_true')
    parser.add_argument('--embeddings_path')
    parser.add_argument('--train_embeddings', action='store_true')
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--checkpoint', default=50, type=int)
    parser.add_argument('--hooks_per_epoch', default=2, type=int)
    args = parser.parse_args()
    
    if args.corpus.lower() == 'msrp':
        from datasets import load_msrp as loader
    elif args.corpus.lower() == 'quora':
        from datasets import load_quora as loader

    print("Loading data")
    train, valid, test = loader(gpu=args.gpu, batch_size=args.batch_size)
    print(" * {} train batches".format(len(train)))
    print(" * {} valid batches".format(len(valid)))
    print(" * {} test batches".format(len(test)))
    valid.set_batch_size(250)
    test.set_batch_size(250)
    train.sort_()

    # embeddings
    d, _ = train.d['src']
    embs = Embedding.from_dict(
        d, args.emb_dim, add_positional='cnn' in args.model.lower())

    # build network
    if args.model.lower() == 'rnn':
        encoder = RNNEncoder(
            embs, args.hid_dim, args.layers, args.cell, dropout=args.dropout,
            summary=args.encoder_summary)
    elif args.model.lower() == 'cnn':
        encoder = CNNEncoder(embs, args.hid_dim, 3, args.layers, dropout=args.dropout)
    elif args.model.lower() == 'cnntext':
        encoder = CNNTextEncoder(embs, out_channels=100, kernel_sizes=(5, 4, 3),
                                 dropout=args.dropout, ktop=args.ktop)
    elif args.model.lower() == 'mwe':
        encoder = MaxoutWindowEncoder(embs, args.layers, dropout=args.dropout)

    m = Siamese(encoder, loss=args.objective, proj_layers=args.proj_layers,
                dropout=args.dropout)

    # initialization
    u.initialize_model(
        m,
        # rnn={'type': 'orthogonal', 'args': {'gain': 1.0}},
        # cnn={'type': 'normal', 'args': {'mean': 0, 'std': 0.1}}
    )
    if args.init_embeddings:
        embs.init_embeddings_from_file(args.embeddings_path, verbose=True)

        if not args.train_embeddings:
            # freeze embeddings
            for p in embs.parameters():
                p.requires_grad = False

    print(m)
    print(" * {} trainable parameters".format(
        sum(p.nelement() for p in m.parameters() if p.requires_grad)))

    if args.gpu:
        m.cuda()

    optimizer = getattr(optim, args.optim)(
        [p for p in m.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-6)
    trainer = Trainer(
        m, {'train': train, 'valid': valid}, optimizer, max_norm=args.max_norm)

    trainer.add_loggers(StdLogger())
    trainer.add_hook(make_classification_hook(args.patience), hooks_per_epoch=1)
    trainer.train(args.epochs, args.checkpoint, shuffle=True)
