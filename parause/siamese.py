
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score

from seqmod.modules.embedding import Embedding
from seqmod.modules.encoder import RNNEncoder
import seqmod.utils as u
from seqmod.misc import Trainer, StdLogger


class SiameseRNN(nn.Module):
    def __init__(self, embeddings, hid_dim, num_layers, cell, dropout=0.0,
                 **kwargs):

        super(SiameseRNN, self).__init__()

        self.encoder = RNNEncoder(embeddings, hid_dim, num_layers, cell,
                                  dropout=dropout, **kwargs)

        if self.encoder.encoding_size[0] > 2:
            raise ValueError("Needs 2D encoding")
        
    def forward(self, pair1, pair2, pair1_len, pair2_len):
        enc1, _ = self.encoder(pair1, lengths=pair1_len)
        enc2, _ = self.encoder(pair2, lengths=pair2_len)
        # average manhattan loss -> probability
        return torch.exp(-torch.abs(enc1 - enc2).sum(1))

    def loss(self, batch_data, test=False):
        ((pair1, pair1_len), (pair2, pair2_len)), labels = batch_data

        pred = self(pair1, pair2, pair1_len, pair2_len)
        loss = F.binary_cross_entropy(pred, labels.float())
        num_examples = labels.size(0)

        if not test:
            loss.backward()

        return (loss.data[0],), num_examples

    def predict_proba(self, batch):
        (pair1, pair1_len), (pair2, pair2_len) = batch
        pred = self(pair1, pair2, pair1_len, pair2_len)
        return pred > 0.5
        

def make_accuracy_hook():

    def hook(trainer, epoch, batch, checkpoint):
        preds, trues = [], []
        for batch, label in trainer.datasets['valid']:
            trues.extend(label.data.tolist())
            pred = trainer.model.predict_proba(batch)
            preds.extend(pred.data.tolist())

        accuracy = accuracy_score(trues, preds)
        trainer.log("info", "Accuracy {:.3f}".format(accuracy))

    return hook


def make_validation_hook():

    def hook(trainer, epoch, batch, checkpoint):
        loss = trainer.validate_model()
        packed = loss.pack(labels=True)
        trainer.log("validation_end", {'epoch': epoch, 'loss': packed})

    return hook
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--corpus', default='msrp')
    parser.add_argument('--dev', default=0.1, type=float)
    # model
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--cell', default='LSTM', type=str)
    parser.add_argument('--emb_dim', default=100, type=int)
    parser.add_argument('--hid_dim', default=50, type=int)
    parser.add_argument('--encoder_summary', default='inner-attention')
    # training
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--init_embeddings', action='store_true')
    parser.add_argument('--embeddings_path')
    parser.add_argument('--max_norm', default=10., type=float)
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
    
    train, valid, test = loader(gpu=args.gpu, batch_size=args.batch_size)
    train.stratify_()

    d, _ = train.d['src']
    embs = Embedding.from_dict(d, args.emb_dim)
    m = SiameseRNN(embs, args.hid_dim, args.layers, args.cell,
                   dropout=args.dropout, summary=args.encoder_summary)
    print(m)

    u.initialize_model(m)
    
    if args.init_embeddings:
        embs.init_embeddings_from_file(args.embeddings_path, verbose=True)

        # freeze embeddings
        # for p in embs.parameters():
        #     p.requires_grad = False

    if args.gpu:
        m.cuda()

    parameters = [p for p in m.parameters() if p.requires_grad]
    optimizer = getattr(optim, args.optim)(parameters, lr=args.lr)

    trainer = Trainer(m, {'train': train, 'valid': valid}, optimizer,
                      max_norm=args.max_norm)
    trainer.add_loggers(StdLogger())
    trainer.add_hook(make_accuracy_hook(), hooks_per_epoch=1)
    trainer.add_hook(make_validation_hook(), hooks_per_epoch=10)

    trainer.train(args.epochs, args.checkpoint, shuffle=True)
