
import math
import time
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch import optim

from seqmod.modules.embedding import Embedding
from seqmod.modules.rnn_encoder import RNNEncoder
from seqmod.modules.torch_utils import flip, pack_sort, shards
from seqmod.misc import Checkpoint, text_processor
import seqmod.utils as u

from text_reuse.skipthought.dataiter import DataIter


def run_decoder(decoder, thought, hidden, inp, lengths):
    """
    Single decoder run
    """
    # project thought across length dimension
    inp = torch.cat([inp, thought.unsqueeze(0).repeat(len(inp), 1, 1)], dim=2)
    # pack for faster processing
    inp, unsort = pack_sort(inp, lengths)
    output, _ = decoder(inp, hidden)
    # unpack and turn into original batch order
    output, _ = unpack(output)
    output = output[:, unsort]

    return output


MODES = {
    'next': (False, True),      # predict next sentence
    'prev': (True, False),      # predict prev sentence
    'double': (True, True),     # predict both previous and next (diff params)
    'clone': (True, True),      # predict both previous and next (shared params)
}


class Loss(nn.Module):

    def __init__(self, embeddings, cell, thought_dim, hid_dim,
                 dropout=0.0, mode='clone'):

        if mode not in MODES:
            raise ValueError("Unknown mode {}".format(mode))

        self.mode = mode
        self.hid_dim = hid_dim
        super(Loss, self).__init__()

        # Embedding
        self.embeddings = embeddings
        inp_size = thought_dim + embeddings.embedding_dim

        # RNN
        self.rnn_n, self.rnn_p = None, None
        if self.mode == 'next':
            self.rnn_n = getattr(nn, cell)(inp_size, hid_dim)
        elif self.mode == 'prev':
            self.rnn_p = getattr(nn, cell)(inp_size, hid_dim)
        elif self.mode == 'double':
            self.rnn_n = getattr(nn, cell)(inp_size, hid_dim)
            self.rnn_p = getattr(nn, cell)(inp_size, hid_dim)
        elif self.mode == 'clone':
            self.rnn_n = getattr(nn, cell)(inp_size, hid_dim)
            self.rnn_p = self.rnn_n
        self._all_rnns = [self.rnn_p, self.rnn_n]

        nll_weight = torch.ones(len(embeddings.d))
        if embeddings.d.get_pad() is not None:
            nll_weight[embeddings.d.get_pad()] = 0
        self.register_buffer('nll_weight', nll_weight)

        # logits
        self.logits = nn.Linear(hid_dim, embeddings.num_embeddings)

    def forward(self, thought, hidden, sents):
        for idx, (sent, rnn) in enumerate(zip(sents, self._all_rnns)):
            if sent is not None:
                if rnn is None:
                    raise ValueError("Unexpected input at pos {}".format(idx + 1))
    
                (sent, lengths) = sent
                # rearrange targets for loss
                inp, target, lengths = sent[:-1], sent[1:], lengths - 1
                num_examples = lengths.sum()
                # run decoder
                inp = self.embeddings(inp)
                output = run_decoder(rnn, thought, hidden, inp, lengths)
    
                yield output, target, num_examples

    def loss(self, thought, hidden, sents, test=False):
        loss, num_examples = 0, 0

        for out, trg, examples in self.forward(thought, hidden, sents):
            for shard in shards({{'output': out, 'target': trg}}, test=test, size=200):
                shard_loss = F.cross_entropy(
                    self.logits(shard['output'].view(-1, self.hid_dim)),
                    shard['target'].view(-1), size_average=False,
                    weight=self.nll_weight, ignore_index=self.embeddings.d.get_pad())
                shard_loss /= examples

                if not test:
                    shard_loss.backward(retain_graph=True)

                loss += shard_loss.data[0]
            num_examples += examples

        return math.exp(loss), num_examples


class SkipThoughts(nn.Module):

    def __init__(self,
                 embeddings,
                 # model opts
                 mode,
                 cell='GRU',
                 hid_dim=2400,
                 num_layers=1,
                 bidi=True,
                 summary='last',
                 dropout=0.0,
                 # training opts
                 max_norm=5.,
                 optimizer='Adam',
                 lr=0.001,
                 save_freq=1000,
                 update_freq=25,
                 checkpoint=None):

        # training
        self.max_norm = max_norm
        self.save_freq = save_freq
        self.update_freq = update_freq
        self.checkpoint = checkpoint
        super(SkipThoughts, self).__init__()

        # model params
        self.encoder = RNNEncoder(embeddings, hid_dim, num_layers, cell,
                                  bidi=bidi, dropout=dropout, summary=summary,
                                  train_init=False, add_init_jitter=False)
        self.decoder = Loss(
            embeddings, cell, self.encoder.encoding_size[1], hid_dim,
            mode=mode, dropout=dropout)

        self.optimizer = getattr(optim, optimizer)(list(self.parameters()), lr)

    def report_params(self):
        enc_params = sum([p.nelement() for p in self.encoder.parameters()])
        dec_params = sum([p.nelement() for p in self.decoder.parameters()])
        emb_params = sum([p.nelement() for p in self.encoder.embeddings.parameters()])
        print((" * {} emb parameters\n"
               " * {} enc parameters\n"
               " * {} dec parameters\n"
               "   => total {}").format(
                   emb_params,
                   enc_params - emb_params,
                   dec_params - emb_params,
                   emb_params + enc_params + dec_params - 2 * emb_params))

    def train(self, batches):
        start, total_loss, total_examples, num_batches = time.time(), 0, 0, 0
        log_batches = tqdm.tqdm(
            batches, postfix={'loss': 'unknown', 'words/sec': 'unknown'})

        for idx, ((inp, lengths), sents) in enumerate(log_batches):

            # loss & update
            self.optimizer.zero_grad()
            thought, hidden = self.encoder(inp, lengths)
            loss, num_examples = self.decoder.loss(thought, hidden, sents)
            torch.nn.utils.clip_grad_norm(self.parameters(), self.max_norm)
            self.optimizer.step()

            # report & save
            total_loss += loss
            total_examples += num_examples
            num_batches += 1

            if (idx + 1) % self.save_freq == 0 and checkpoint is not None:
                # must go before the update logic to avoid division by zero
                self.checkpoint.save_nbest(self, loss=total_loss / num_batches)
                self.checkpoint.save_nlast(self)

            if (idx + 1) % self.update_freq == 0:
                restart = time.time()
                speed = total_examples / (restart - start)
                log_batches.set_postfix(
                    {'loss': '{:.3f}'.format(total_loss / num_batches),
                     'words/sec': '{:.2f}'.format(speed)})
                start, total_loss, total_examples, num_batches = restart, 0, 0, 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--paths', nargs='+', required=True)
    parser.add_argument('--dict_path', required=True)
    parser.add_argument('--min_len', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=35)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='word')
    # model
    parser.add_argument('--mode', default='clone')
    parser.add_argument('--emb_dim', type=int, default=620)
    parser.add_argument('--hid_dim', type=int, default=2400)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--cell', default='GRU')
    parser.add_argument('--summary', default='last')
    parser.add_argument('--train_init', action='store_true')
    parser.add_argument('--init_embeddings', action='store_true')
    parser.add_argument('--embeddings_path',
                        default='/home/corpora/word_embeddings/' +
                        'glove.840B.300d.txt')
    # training
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_schedule_checkpoints', type=int, default=1)
    parser.add_argument('--lr_schedule_factor', type=float, default=1)
    parser.add_argument('--max_norm', type=float, default=5.)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    embeddings = Embedding.from_dict(u.load_model(args.dict_path), args.emb_dim)

    checkpoint = None
    if not args.test:
        modeldir = 'SkipThoughts-{}'.format(args.mode)
        checkpoint = Checkpoint(modeldir, keep=3).setup(args)

    m = SkipThoughts(embeddings, args.mode, cell=args.cell, hid_dim=args.hid_dim,
                     num_layers=args.num_layers, summary=args.summary,
                     dropout=args.dropout, max_norm=args.max_norm,
                     optimizer=args.optimizer, lr=args.lr,
                     save_freq=args.save_freq, checkpoint=checkpoint)

    print(m)
    print()
    print("Parameter stats ...\n")
    m.report_params()

    print()
    print("Initializing parameters ...\n")
    u.initialize_model(
        m,
        rnn={'type': 'rnn_orthogonal', 'args': {'forget_bias': True}},
        emb={'type': 'uniform', 'args': {'a': -0.1, 'b': 0.1}})

    if args.init_embeddings:
        embeddings.init_embeddings_from_file(args.embeddings_path, verbose=True)

    if args.gpu:
        m.cuda()

    dataiter = DataIter(m.encoder.embeddings.d, *args.paths, includes=MODES[args.mode],
                        min_len=args.min_len, max_len=args.max_len, gpu=args.gpu)

    print()
    print("Starting training ...\n")
    try:
        for epoch in range(1, args.epochs + 1):
            print("***{} Epoch #{} {}***".format("---" * 4, epoch, "---" * 4))
            m.train(dataiter.batch_generator(args.batch_size, buffer_size=100000))
            print()
    except KeyboardInterrupt:
        print("Bye!")
    finally:
        print("Finished!")
