
import copy

import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from scipy.stats import pearsonr

from seqmod.modules.embedding import Embedding
from seqmod.modules.rnn_encoder import RNNEncoder
from seqmod.misc import Trainer, StdLogger, EarlyStopping, Checkpoint
import seqmod.utils as u

from textreuse.siamese import Siamese, MeanMaxCNNEncoder


def make_binary_validation_hook(patience, checkpoint=None):

    early_stopping = None
    if patience > 0:
        early_stopping = EarlyStopping(patience)

    def hook(trainer, epoch, batch, check):
        scores, trues, preds = trainer.model.evaluate(trainer.datasets['valid'])
        # correlation
        corr, _ = pearsonr(np.array(scores), np.array(trues).astype(np.float))
        trainer.log("info", "Pearsonr {:.3f}".format(corr))
        # acc & auc
        if len(preds) > 0:
            preds = np.array(preds)
            acc, auc = accuracy_score(trues, preds), roc_auc_score(trues, preds)
            trainer.log("info", "Acc {:.3f}; AUC {:.3f}".format(acc, auc))
        # checkpoints
        model, loss = None, 1 - corr  # lower must be better
        if early_stopping is not None:
            model = copy.deepcopy(trainer.model).cpu()
            early_stopping.add_checkpoint(loss, model=model)
        if checkpoint is not None:
            if model is None:
                model = copy.deepcopy(trainer.model).cpu()
            checkpoint.save(model, loss)

    return hook


def make_multinomial_validation_hook(patience, checkpoint=None):

    early_stopping = None
    if patience > 0:
        early_stopping = EarlyStopping(patience)

    def hook(trainer, epoch, batch, check):
        scores, trues, _ = trainer.model.evaluate(trainer.datasets['valid'])
        # transform trues to scalar form
        trues = np.dot(np.array(trues), np.arange(1, len(trues[0]) + 1))
        scores = np.array(scores)
        # correlation
        (corr, _), mse = pearsonr(scores, trues), mean_squared_error(trues, scores)
        trainer.log("info", "Pearsonr {:.3f}; MSE {:.3f}".format(corr, mse))
        # checkpoints
        model, loss = None, 1 - corr  # lower must be better
        if early_stopping is not None:
            model = copy.deepcopy(trainer.model).cpu()
            early_stopping.add_checkpoint(mse, model=model)
        if checkpoint is not None:
            if model is None:
                model = copy.deepcopy(trainer.model).cpu()
            checkpoint.save(model, mse)

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
    parser.add_argument('--init_from_skipthoughts', help='path to ST model')
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

    binary = True
    if args.corpus.lower() == 'msrp':
        from datasets import load_msrp as loader
    elif args.corpus.lower() == 'quora':
        from datasets import load_quora as loader
    elif args.corpus.lower() == 'sick':
        from datasets import load_sick as loader
        binary = False

    d, encoder = None, None
    if args.init_from_skipthoughts:
        encoder = u.load_model(args.init_from_skipthoughts).encoder
        d = encoder.embeddings.d
        for p in encoder.parameters():
            p.requires_grad = False

    print("Loading data")
    train, valid, _ = loader(gpu=args.gpu, batch_size=args.batch_size, d=d)
    print(" * {} train batches".format(len(train)))
    print(" * {} valid batches".format(len(valid)))
    valid.set_batch_size(250)
    if binary:
        train.sort_(key=lambda pair: len(pair[0]), sort_by='src').shuffle_().stratify_()
    else:
        train.sort_(key=lambda pair: len(pair[0]), sort_by='src')

    if encoder is None:
        # embeddings
        d, _ = train.d['src']
        embs = Embedding.from_dict(
            d, args.emb_dim, add_positional='cnn' in args.model.lower())

        # build network
        if args.model.lower() == 'rnn' and args.init_from_lm:
            encoder = RNNEncoder.from_lm(
                u.load_model(args.init_from_lm), embeddings=embs,
                summary=args.encoder_summary, dropout=args.dropout)
            for p in encoder.parameters():
                p.requires_grad = False
        elif args.model.lower() == 'rnn':
            encoder = RNNEncoder(
                embs, args.hid_dim, args.layers, args.cell, dropout=args.dropout,
                summary=args.encoder_summary)
        elif args.model.lower() == 'cnn':
            encoder = MeanMaxCNNEncoder(
                embs, args.hid_dim, 3, args.layers, dropout=args.dropout)
        elif args.model.lower() == 'cnntext':
            encoder = CNNTextEncoder(embs, out_channels=100, kernel_sizes=(5, 4, 3),
                                     dropout=args.dropout, ktop=args.ktop)
        elif args.model.lower() == 'mwe':
            encoder = MaxoutWindowEncoder(
                embs, args.layers, maxouts=3, downproj=128, dropout=args.dropout)

    m = Siamese(encoder, args.objective,
                proj_layers=args.proj_layers, dropout=args.dropout,
                nclass=5)

    # initialization
    if not (args.init_from_lm or args.init_from_skipthoughts):
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
        checkpoint = Checkpoint('Siamese-{}'.format(args.corpus), keep=3).setup(args)
        logfile = checkpoint.checkpoint_path('logfile.txt')

    trainer.add_loggers(StdLogger(outputfile=logfile))

    if binary:
        hook = make_binary_validation_hook(args.patience, checkpoint=checkpoint)
    else:
        hook = make_multinomial_validation_hook(args.patience, checkpoint=checkpoint)
    trainer.add_hook(hook, hooks_per_epoch=args.hooks_per_epoch)

    trainer.train(args.epochs, args.checkpoint, shuffle=True)
