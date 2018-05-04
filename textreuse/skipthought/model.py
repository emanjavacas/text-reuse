
import math
import time
import tqdm

from scipy.linalg import norm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch import optim

from seqmod.modules.rnn_encoder import RNNEncoder
from seqmod.modules.softmax import FullSoftmax, SampledSoftmax


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


class Loss(nn.Module):

    MODES = {
        'next': (False, True),      # predict next sentence
        'prev': (True, False),      # predict prev sentence
        'double': (True, True),     # predict both previous and next (diff params)
        'clone': (True, True),      # predict both previous and next (shared params)
        'self': True
    }

    def __init__(self, embeddings, cell, thought_dim, hid_dim,
                 dropout=0.0, mode='double', softmax='tied'):

        if mode not in Loss.MODES:
            raise ValueError("Unknown mode {}".format(mode))

        self.mode = mode
        self.hid_dim = hid_dim
        super(Loss, self).__init__()

        # Embedding
        self.embeddings = embeddings
        inp_size = thought_dim + embeddings.embedding_dim

        # RNN
        if self.mode == 'self':
            self.rnn = getattr(nn, cell)(inp_size, hid_dim)
            self._all_rnns = [self.rnn]

        else:
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
        loss, report_loss, num_examples = 0, 0, 0

        for out, trg, examples in self.forward(thought, hidden, sents):
            out, trg = out.view(-1, self.hid_dim), trg.view(-1)
            dec_loss = 0

            if isinstance(self.logits, SampledSoftmax) and self.training:
                out, new_trg = self.logits(out, targets=trg, normalize=False)
                dec_loss = F.cross_entropy(out, new_trg, size_average=False)
            else:
                dec_loss = F.cross_entropy(
                    self.logits(out, normalize=False), trg, size_average=False,
                    weight=self.nll_weight)

            dec_loss /= examples
            loss += dec_loss

            # report
            report_loss = math.exp(dec_loss.data[0])
            num_examples += examples

        if not test:
            loss.backward()

        return report_loss, num_examples


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
            self.train()

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

        # save end of epoch
        self.eval()
        self.checkpoint.save_nlast(self)

    def encode(self, sents, use_norm=True, batch_size=100):
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

        if use_norm:
            feats = feats / np.sqrt((feats**2).sum(1))[:, None]

        return feats
