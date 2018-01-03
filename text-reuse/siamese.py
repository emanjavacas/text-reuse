
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score

from seqmod.modules.embedding import Embedding
from seqmod.modules.encoder import RNNEncoder
from seqmod.modules.dcnn_encoder import DCNNEncoder
from seqmod.modules.cnn_text_encoder import CNNTextEncoder
import seqmod.utils as u
from seqmod.misc import Trainer, StdLogger, EarlyStopping


class L1SiameseBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super(L1SiameseBase, self).__init__()

    def get_pair_encoding(self, pair1, pair2, lengths):
        raise NotImplementedError

    def forward(self, pair1, pair2, lengths=(None, None)):
        enc1, enc2 = self.get_pair_encoding(pair1, pair2, lengths=lengths)
        # average manhattan loss -> probability
        return torch.exp(-torch.abs(enc1 - enc2).sum(1))

    def loss(self, batch_data, test=False):
        ((pair1, pair1_len), (pair2, pair2_len)), labels = batch_data

        pred = self(pair1, pair2, lengths=(pair1_len, pair2_len))
        loss = F.binary_cross_entropy(pred, labels.float())
        num_examples = sum(pair1_len + pair2_len).data[0]

        if not test:
            loss.backward()

        return (loss.data[0],), num_examples

    def predict_proba(self, pair1, pair2, lengths=(None, None)):
        return self(pair1, pair2, lengths=lengths) > 0.5


class L1SiameseRNN(L1SiameseBase):
    def __init__(self, embs, hid_dim, num_layers, cell, dropout=0.0, **kwargs):

        super(L1SiameseRNN, self).__init__()

        self.encoder = RNNEncoder(
            embs, hid_dim, num_layers, cell, dropout=dropout, **kwargs)

        if self.encoder.encoding_size[0] > 2:
            raise ValueError("Needs 2D encoding")

    def get_pair_encoding(self, pair1, pair2, lengths):
        pair1_len, pair2_len = lengths
        enc1, _ = self.encoder(pair1, lengths=pair1_len)
        enc2, _ = self.encoder(pair2, lengths=pair2_len)
        return enc1, enc2


class L1SiameseCNNText(L1SiameseBase):
    def __init__(self, embs, out_channels=100, kernel_sizes=(5, 4, 3),
                 dropout=0.0, **kwargs):
        super(L1SiameseCNNText, self).__init__()

        self.encoder = CNNTextEncoder(
            embs, out_channels=out_channels, kernel_sizes=kernel_sizes,
            dropout=dropout, **kwargs)

        self.output = nn.Sequential(nn.Linear(self.encoder.encoding_size[1], 100),
                                    nn.ReLU())

    def get_pair_encoding(self, pair1, pair2, lengths=None):
        return self.output(self.encoder(pair1)), self.output(self.encoder(pair2))


class L1SiameseDCNN(L1SiameseBase):
    def __init__(self, embs, out_channels=(6, 10, 14), kernel_sizes=(7, 5, 3),
                 ktop=4, folding_factor=4, dropout=0.0, **kwargs):
        super(L1SiameseDCNN, self).__init__()

        self.encoder = DCNNEncoder(
            embs, out_channels=out_channels, kernel_sizes=kernel_sizes,
            ktop=ktop, folding_factor=folding_factor, dropout=dropout,
            **kwargs)

        self.output = nn.Sequential(nn.Linear(self.encoder.encoding_size[1], 100),
                                    nn.ReLU())

    def get_pair_encoding(self, pair1, pair2, lengths=None):
        return self.output(self.encoder(pair1)), self.output(self.encoder(pair2))


def make_nn(d, args):
    embs = Embedding.from_dict(d, args.emb_dim)

    if args.model.lower() == 'rnn':
        m = L1SiameseRNN(embs, args.hid_dim, args.layers, args.cell,
                         dropout=args.dropout, summary=args.encoder_summary)

    elif args.model.lower() == 'dcnn':
        m = L1SiameseDCNN(embs, dropout=args.dropout)

    elif args.model.lower() == 'cnntext':
        m = L1SiameseCNNText(embs, dropout=args.dropout)

    u.initialize_model(m, rnn={'type': 'orthogonal', 'args': {'gain': 1.0}},
                       cnn={'type': 'normal', 'args': {'mean': 0, 'std': 0.1}})
    
    if args.init_embeddings:
        embs.init_embeddings_from_file(args.embeddings_path, verbose=True)

        if not args.train_embeddings:
            # freeze embeddings
            for p in embs.parameters():
                p.requires_grad = False

    return m


def make_accuracy_hook():

    def hook(trainer, epoch, batch, checkpoint):
        preds, trues = [], []
        for batch, label in trainer.datasets['valid']:
            (p1, p1_len), (p2, p2_len) = batch
            trues.extend(label.data.tolist())
            pred = trainer.model.predict_proba(p1, p2, lengths=(p1_len, p2_len))
            preds.extend(pred.data.tolist())

        accuracy = accuracy_score(trues, preds)
        trainer.log("info", "Accuracy {:.3f}".format(accuracy))

    return hook


def make_validation_hook():

    def hook(trainer, epoch, batch, checkpoint):
        loss = trainer.validate_model()
        packed = loss.pack(labels=True)
        trainer.log("validation_end", {'epoch': epoch, 'loss': packed})

        print([c.weight.norm().data[0] for c in trainer.model.encoder.conv])

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
    parser.add_argument('--encoder_summary', default='inner-attention')
    # training
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--optim', default='Adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--init_embeddings', action='store_true')
    parser.add_argument('--embeddings_path')
    parser.add_argument('--train_embeddings', action='store_true')
    parser.add_argument('--max_norm', default=5., type=float)
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--word_dropout', default=0.0, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--checkpoint', default=200, type=int)
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
    train.stratify_()

    d, _ = train.d['src']
    m = make_nn(d, args)
    print(m)

    print(" * {} trainable parameters".format(
        sum(p.nelement() for p in m.parameters() if p.requires_grad)))

    if args.gpu:
        m.cuda()

    parameters = [p for p in m.parameters() if p.requires_grad]
    optimizer = getattr(optim, args.optim)(parameters, lr=args.lr)

    early_stopping = None
    if args.patience > 0:
        early_stopping = EarlyStopping(args.patience)
    trainer = Trainer(m, {'train': train, 'valid': valid}, optimizer,
                      max_norm=args.max_norm, early_stopping=early_stopping)

    trainer.add_loggers(StdLogger())
    trainer.add_hook(make_accuracy_hook(), hooks_per_epoch=1)
    trainer.add_hook(make_validation_hook(), hooks_per_epoch=10)
    trainer.train(args.epochs, args.checkpoint, shuffle=True)
