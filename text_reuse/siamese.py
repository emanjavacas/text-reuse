
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import pearsonr

from seqmod.modules.embedding import Embedding
from seqmod.modules.encoder import MaxoutWindowEncoder
from seqmod.modules.rnn_encoder import RNNEncoder
from seqmod.modules.cnn_encoder import CNNEncoder
from seqmod.modules.cnn_text_encoder import CNNTextEncoder
from seqmod.modules.ff import Highway
from seqmod.misc import Trainer, StdLogger, EarlyStopping, Checkpoint
import seqmod.utils as u

from objectives import NormObjective, SigmoidObjective, CauchyObjective
from objectives import ContrastiveCosineObjective, ContrastiveEuclideanObjective


class MeanMaxCNNEncoder(CNNEncoder):
    def forward(self, inp, **kwargs):
        # output: (seq_len x batch x hid_dim)
        output, _ = super(MeanMaxCNNEncoder, self).forward(inp, **kwargs)

        # collapse seq_len
        return torch.cat([output.mean(0), output.max(0)[0]], dim=1)

    @property
    def encoding_size(self):
        return 2, super(MeanMaxCNNEncoder, self).encoding_size[1] * 2
    


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
        else:
            raise ValueError("Unknown objective [{}]".format(loss))

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

    def score(self, pair1, pair2, lengths=(None, None)):
        return self.objective.score(self(pair1, pair2, lengths=lengths))

    def evaluate(self, dataset):
        preds, trues, scores = [], [], []

        for batch, label in dataset:
            (p1, p1_len), (p2, p2_len) = batch
            score = self.score(p1, p2, lengths=(p1_len, p2_len))
            pred = self.objective.predict(score)
            trues.extend(label.data.tolist())
            preds.extend(pred.data.tolist())
            scores.extend(score.data.tolist())

        return trues, preds, scores


def make_validation_hook(patience, checkpoint=None):

    early_stopping = None
    if patience > 0:
        early_stopping = EarlyStopping(patience)

    def hook(trainer, epoch, batch, check):
        # accuracy
        trues, preds, scores = trainer.model.evaluate(trainer.datasets['valid'])
        corr, _ = pearsonr(np.array(scores), np.array(trues).astype(np.float))
        preds = np.array(preds).clip(0, 1)  # some objectives don't return proper preds
        acc, auc = accuracy_score(trues, preds), roc_auc_score(trues, preds)
        trainer.log("info", "R {:.3f}; Acc {:.3f}; AUC {:.3f}".format(corr, acc, auc))

        model, loss = None, 1 - corr  # lower must be better
        if early_stopping is not None:
            model = copy.deepcopy(trainer.model).cpu()
            early_stopping.add_checkpoint(loss, model=model)
        if checkpoint is not None:
            if model is None:
                model = copy.deepcopy(trainer.model).cpu()
            checkpoint.save(model, loss)

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
    parser.add_argument('--embeddings_path',
                        default='/home/corpora/word_embeddings/' +
                        'glove.840B.300d.txt')
    parser.add_argument('--train_embeddings', action='store_true')
    parser.add_argument('--init_from_lm', help='path to LM model')
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--checkpoint', default=50, type=int)
    parser.add_argument('--hooks_per_epoch', default=2, type=int)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    
    if args.corpus.lower() == 'msrp':
        from datasets import load_msrp as loader
    elif args.corpus.lower() == 'quora':
        from datasets import load_quora as loader

    print("Loading data")
    train, valid, _ = loader(gpu=args.gpu, batch_size=args.batch_size)
    print(" * {} train batches".format(len(train)))
    print(" * {} valid batches".format(len(valid)))
    valid.set_batch_size(250)
    train.sort_(key=lambda pair: len(pair[0]), sort_by='src').shuffle_().stratify_()

    # embeddings
    d, _ = train.d['src']
    embs = Embedding.from_dict(d, args.emb_dim, add_positional='cnn' in args.model.lower())

    # build network
    if args.model.lower() == 'rnn':
        if args.init_from_lm is not None:
            encoder = RNNEncoder.from_lm(
                u.load_model(args.init_from_lm), embeddings=embs,
                summary=args.encoder_summary, dropout=args.dropout)
        else:
            encoder = RNNEncoder(
                embs, args.hid_dim, args.layers, args.cell, dropout=args.dropout,
                summary=args.encoder_summary)
    elif args.model.lower() == 'cnn':
        encoder = MeanMaxCNNEncoder(embs, args.hid_dim, 3, args.layers, dropout=args.dropout)
    elif args.model.lower() == 'cnntext':
        encoder = CNNTextEncoder(embs, out_channels=100, kernel_sizes=(5, 4, 3),
                                 dropout=args.dropout, ktop=args.ktop)
    elif args.model.lower() == 'mwe':
        encoder = MaxoutWindowEncoder(
            embs, args.layers, maxouts=3, downproj=128, dropout=args.dropout)

    m = Siamese(encoder, loss=args.objective, proj_layers=args.proj_layers,
                dropout=args.dropout)

    # initialization
    if not args.init_from_lm:
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
    scheduler = None            # TODO!
    trainer = Trainer(
        m, {'train': train, 'valid': valid}, optimizer, max_norm=args.max_norm)

    checkpoint, logfile = None, '/tmp/logfile'
    if args.save:
        checkpoint = Checkpoint('Siamese-%s' % args.corpus, buffer_size=3).setup(args)
        logfile = checkpoint.checkpoint_path('logfile.txt')

    trainer.add_loggers(StdLogger(outputfile=logfile))
    trainer.add_hook(make_validation_hook(args.patience, checkpoint=checkpoint),
                     hooks_per_epoch=args.hooks_per_epoch)

    trainer.train(args.epochs, args.checkpoint, shuffle=True)
