
import os
import glob
import math
import time
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch import optim

from seqmod.modules.embedding import Embedding
from seqmod.modules.rnn_encoder import RNNEncoder
from seqmod.modules.softmax import FullSoftmax, SampledSoftmax
from seqmod.modules.torch_utils import flip, shards
from seqmod.misc import Checkpoint, text_processor
import seqmod.utils as u

from text_reuse.skipthought.dataiter import DataIter


def run_decoder(decoder, thought, hidden, inp, lengths):
    """
    Single decoder run
    """
    # project thought across length dimension
    inp = torch.cat([inp, thought.unsqueeze(0).repeat(len(inp), 1, 1)], dim=2)

    # sort faster processing
    lengths, sort = torch.sort(lengths, descending=True)
    _, unsort = sort.sort()
    if isinstance(hidden, tuple):
        hidden = hidden[0][:,sort,:], hidden[1][:,sort,:]
    else:
        hidden = hidden[:,sort,:]

    # pack
    inp = pack(inp[:, sort], lengths.tolist())

    # run decoder
    output, _ = decoder(inp, hidden)

    # unpack & unsort
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
                 dropout=0.0, mode='clone', softmax='tied'):

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

        if softmax == 'single':
            self.logits = FullSoftmax(
                hid_dim, embeddings.embedding_dim, embeddings.num_embeddings)
        elif softmax == 'tied':
            self.logits = FullSoftmax(
                hid_dim, embeddings.embedding_dim, embeddings.num_embeddings,
                tie_weights=True)
            self.logits.tie_embedding_weights(embeddings)
        elif softmax == 'sampled':
            self.logits = SampledSoftmax(
                hid_dim, embeddings.embedding_dim, embeddings.num_embeddings)
        else:
            raise ValueError("Unknown softmax {}".format(softmax))

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
            dec_loss = 0

            for shard in shards({'out': out, 'trg': trg}, test=test, size=20):
                out, trg = shard['out'].view(-1, self.hid_dim), shard['trg'].view(-1)
                if isinstance(self.logits, SampledSoftmax) and self.training:
                    out, new_trg = self.logits(out, targets=trg, normalize=False)
                    shard_loss = F.cross_entropy(out, new_trg, size_average=False)
                else:
                    shard_loss = F.cross_entropy(
                        self.logits(out, normalize=False), trg, size_average=False,
                        weight=self.nll_weight)
                shard_loss /= examples

                if not test:
                    shard_loss.backward(retain_graph=True)

                dec_loss += shard_loss.data[0]

            loss += math.exp(dec_loss)
            num_examples += examples

        return loss, num_examples


class SkipThoughts(nn.Module):

    def __init__(self,
                 embeddings,
                 # model opts
                 mode,
                 softmax='single',
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
                 save_freq=50000,
                 update_freq=500,
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
            mode=mode, dropout=dropout, softmax=softmax)

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

    def train_model(self, batches):
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
                # self.checkpoint.save_nbest(self, loss=total_loss / num_batches)
                self.eval()
                self.checkpoint.save_nlast(self)
                self.train()
                print(" ")

            if (idx + 1) % self.update_freq == 0:
                restart = time.time()
                speed = total_examples / (restart - start)
                log_batches.set_postfix(
                    {'loss': '{:.3f}'.format(total_loss / num_batches),
                     'words/sec': '{:.2f}'.format(speed)})
                start, total_loss, total_examples, num_batches = restart, 0, 0, 0

    def encode(self, sents, batch_size=100):
        """
        Encode a number of input sentences, where each sentence is a list of strings
        """
        d = self.encoder.embeddings.d
        sents = list(d.transform(sents))
        feats = np.zeros((len(sents), self.encoder.encoding_size[1]), dtype=np.float)
        processed = 0

        while len(sents) > 0:
            num_examples = min(batch_size, len(sents))
            batch = sents[:num_examples]
            inp, lens = d.pack(batch, return_lengths=True)
            inp, lens = Variable(inp, volatile=True), torch.LongTensor(lens)

            if next(self.parameters()).is_cuda:
                inp, lens = inp.cuda(), lens.cuda()

            output, _ = self.encoder(inp, lens)
            output = output.data.cpu().numpy()

            for idx, f in enumerate(output):
                feats[processed + idx,:] = f

            sents = sents[num_examples:]
            processed += num_examples

        return feats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--paths', required=True)
    parser.add_argument('--dict_path', required=True)
    parser.add_argument('--min_len', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=35)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='word')
    # model
    parser.add_argument('--mode', default='clone')
    parser.add_argument('--softmax', default='single')
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
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--save_freq', type=int, default=50000)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    embeddings = Embedding.from_dict(u.load_model(args.dict_path), args.emb_dim)

    checkpoint = None
    if not args.test:
        modeldir = 'SkipThoughts-{}'.format(args.mode)
        checkpoint = Checkpoint(modeldir, keep=5).setup(args)

    m = SkipThoughts(embeddings, args.mode, cell=args.cell, hid_dim=args.hid_dim,
                     num_layers=args.num_layers, summary=args.summary,
                     softmax=args.softmax, dropout=args.dropout,
                     max_norm=args.max_norm, optimizer=args.optimizer, lr=args.lr,
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

    m.train()

    paths = glob.glob(os.path.expanduser(args.paths))

    dataiter = DataIter(m.encoder.embeddings.d, *paths, includes=MODES[args.mode],
                        min_len=args.min_len, max_len=args.max_len, gpu=args.gpu,
                        always_reverse=args.mode=='clone')

    print()
    print("Starting training ...\n")
    try:
        for epoch in range(1, args.epochs + 1):
            print("***{} Epoch #{} {}***".format("---" * 4, epoch, "---" * 4))
            m.train_model(dataiter.batch_generator(args.batch_size, buffer_size=100000))
            print()
    except KeyboardInterrupt:
        print("Bye!")
    finally:
        print("Finished!")
