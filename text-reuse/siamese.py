
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score

from seqmod.modules.embedding import Embedding
from seqmod.modules.rnn_encoder import RNNEncoder
from seqmod.modules.cnn_encoder import CNNEncoder
from seqmod.modules.cnn_text_encoder import CNNTextEncoder
from seqmod.modules.ff import Highway
import seqmod.utils as u
from seqmod.misc import Trainer, StdLogger, EarlyStopping


class Siamese(nn.Module):
    def __init__(self, encoder, objective='manhattan', margin=0.2,
                 proj_layers=0, dropout=0.2):
        """
        objective: one of ('sigmoid', 'manhattan', 'contrastive')
        """
        self.objective = objective
        self.margin = margin
        self.dropout = dropout
        super(Siamese, self).__init__()

        self.encoder = encoder
        encoding_size = encoder.encoding_size[1]

        self.proj = None
        if proj_layers > 0:
            self.proj = Highway(encoding_size, proj_layers, dropout=dropout)

        if self.objective == 'sigmoid':
            # project to single unit output
            self.logits = nn.Linear(self.encoder.encoding_size[1], 1)

    def contrastive_loss(self, sims, y):
        """
        Implementation from "Learning Text Similarity with Siamese Recurrent Networks"
        http://www.aclweb.org/anthology/W16-1617
        It diverges from original "contrastive loss" definition by Chopra et al 2005,
        based on euclidean distance instead of cosine (which makes sense for text).
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        - This paper's definition:

        L_+(x_1, x_2, y) = 1/4 * (1 - cosine_similarity(x_1, x_2)) ^ 2
        L_-(x_1, x_2, y) = max(0, cosine_similarity(x_1, x_2) - margin) ^ 2

        - Original definition:
        L_+(x_1, x_2, y) = 1/2 * euclidean_dist(x_1, x_2) ^ 2
        L_-(x_1, x_2, y) = 1/2 * max(0, margin - euclidean_dist(x_1, x_2)) ^ 2
        """
        # cosine distance scaled down by negative sampling factor (1/4)
        pos = 0.25 * (1 - sims) # ** 2
        # cosine similarity must be larger than a margin for negative examples
        neg = torch.clamp(sims - self.margin, min=0)  # ** 2

        return torch.mean(y * pos + (1 - y) * neg)

    def forward(self, p1, p2, lengths=(None, None)):
        enc1, enc2 = self.encoder(p1, lengths[0]), self.encoder(p2, lengths[1])
        if isinstance(enc1, tuple):
            enc1, enc2 = enc1[0], enc2[0]  # some encoders also return hidden

        if self.proj is not None:
            enc1, enc2 = self.proj(enc1), self.proj(enc2)

        if self.objective == 'sigmoid':
            output = F.dropout(enc1 - enc2, p=self.dropout, training=self.training)
            output = self.logits(output).squeeze(1)  # (batch x 1) => (batch)
            return output

        elif self.objective == 'manhattan':
            output = F.dropout(enc1 - enc2, p=self.dropout, training=self.training)
            return torch.exp(-torch.norm(output, 1, dim=1))

        elif self.objective == 'contrastive':
            return F.cosine_similarity(enc1, enc2, dim=1)

    def loss(self, batch_data, test=False):
        ((pair1, pair1_len), (pair2, pair2_len)), labels = batch_data
        labels = labels.float()  # transform to float
        num_examples = sum(pair1_len + pair2_len).data[0]

        pred = self(pair1, pair2, lengths=(pair1_len, pair2_len))
        if self.objective == 'sigmoid':
            loss = F.binary_cross_entropy_with_logits(pred, labels)

        elif self.objective == 'manhattan':
            loss = F.binary_cross_entropy(pred, labels)

        elif self.objective == 'contrastive':
            loss = self.contrastive_loss(pred, labels)

        if not test:
            loss.backward()

        return (loss.data[0],), num_examples

    def predict(self, pair1, pair2, lengths=(None, None), threshold=0.5):
        pred = self(pair1, pair2, lengths=lengths)

        if self.objective == 'sigmoid':
            # pred is logits
            return F.sigmoid(pred) > threshold

        elif self.objective == 'manhattan':
            return pred > threshold

        elif self.objective == 'contrastive':
            # pred is cosine: should be transformed to (0, 1)?
            return pred > threshold

    def evaluate(self, dataset, threshold=0.5):
        preds, trues = [], []

        for batch, label in dataset:
            (p1, p1_len), (p2, p2_len) = batch
            pred = self.predict(p1, p2, lengths=(p1_len, p2_len), threshold=threshold)
            trues.extend(label.data.tolist())
            preds.extend(pred.data.tolist())

        return trues, preds


def make_rnn_siamese(embs, hid_dim, num_layers, cell,
                     dropout=0.0, summary='inner-attention', **kwargs):
    encoder = RNNEncoder(
        embs, hid_dim, num_layers, cell, dropout=dropout, summary=summary)

    return Siamese(encoder, **kwargs)


def make_cnn_siamese(embs, hid_dim, kernel_size, num_layers, dropout=0.0, **kwargs):
    encoder = CNNEncoder(embs, hid_dim, kernel_size, num_layers, dropout=dropout)

    return Siamese(encoder, **kwargs)


def make_cnn_text_siamese(embs, out_channels=100, kernel_sizes=(5, 4, 3),
                          dropout=0.0, ktop=1, **kwargs):
    encoder = CNNTextEncoder(
        embs, out_channels=out_channels, kernel_sizes=kernel_sizes,
        dropout=dropout, ktop=ktop)

    return Siamese(encoder, **kwargs)


def make_accuracy_hook(patience):

    early_stopping = None
    if patience > 0:
        early_stopping = EarlyStopping(patience)

    def hook(trainer, epoch, batch, checkpoint):
        # accuracy
        trues, preds = trainer.model.evaluate(trainer.datasets['valid'])
        accuracy = accuracy_score(trues, preds)
        trainer.log("info", "Accuracy {:.3f}".format(accuracy))
        early_stopping.add_checkpoint(accuracy, copy.deepcopy(trainer.model))

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
    train.stratify_()

    # embeddings
    d, _ = train.d['src']
    embs = Embedding.from_dict(
        d, args.emb_dim, add_positional='cnn' in args.model.lower())

    # build network
    if args.model.lower() == 'rnn':
        m = make_rnn_siamese(embs, args.hid_dim, args.layers, args.cell,
                             dropout=args.dropout, summary=args.encoder_summary,
                             objective=args.objective, proj_layers=args.proj_layers)

    elif args.model.lower() == 'cnntext':
        m = make_cnn_text_siamese(embs, dropout=args.dropout, ktop=args.ktop,
                                  objective=args.objective, proj_layers=args.proj_layers)

    elif args.model.lower() == 'cnn':
        m = make_cnn_siamese(embs, args.hid_dim, 3, args.layers, dropout=args.dropout,
                             objective=args.objective, proj_layers=args.proj_layers)

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
        [p for p in m.parameters() if p.requires_grad], lr=args.lr)
    trainer = Trainer(m, {'train': train, 'valid': valid}, optimizer,
                      max_norm=args.max_norm)

    trainer.add_loggers(StdLogger())
    trainer.add_hook(make_accuracy_hook(args.patience), hooks_per_epoch=1)
    trainer.train(args.epochs, args.checkpoint, shuffle=True)
