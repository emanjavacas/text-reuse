
import torch
import torch.nn as nn

from seqmod.modules.encoder import MaxoutWindowEncoder
from seqmod.modules.cnn_encoder import CNNEncoder
from seqmod.modules.cnn_text_encoder import CNNTextEncoder
from seqmod.modules.ff import Highway

from textreuse.objectives import NormObjective, SigmoidObjective, CauchyObjective
from textreuse.objectives import MultinomialObjective
from textreuse.objectives import ContrastiveCosineObjective, ContrastiveEuclideanObjective
from textreuse.objectives import ContrastiveMahalanobisObjective


class MeanMaxCNNEncoder(CNNEncoder):
    def forward(self, inp, **kwargs):
        # output: (seq_len x batch x hid_dim)
        output, _ = super(MeanMaxCNNEncoder, self).forward(inp, **kwargs)

        # collapse seq_len
        return torch.cat([output.mean(0), output.max(0)[0]], dim=1)

    @property
    def encoding_size(self):
        return 2, super(MeanMaxCNNEncoder, self).encoding_size[1] * 2


def make_objective(objtype, encoding_size, nclass=None, weight=None, margin=None):
    kwargs = {}
    if weight is not None:
        kwargs['weight'] = weight
    if margin is not None:
        kwargs['margin'] = margin

    if objtype == 'L1':
        objective = NormObjective(norm=1)
    elif objtype == 'L2':
        objective = NormObjective(norm=2)
    elif objtype == 'sigmoid':
        objective = SigmoidObjective(encoding_size)
    elif objtype == 'cosine':
        objective = ContrastiveCosineObjective(**kwargs)
    elif objtype == 'euclidean':
        objective = ContrastiveEuclideanObjective(**kwargs)
    elif objtype == 'cauchy':
        objective = CauchyObjective(encoding_size)
    elif objtype == 'mahalanobis':
        objective = ContrastiveMahalanobisObjective(encoding_size, **kwargs)
    elif objtype == 'multinomial':
        objective = MultinomialObjective(encoding_size, nclass=nclass)
    else:
        raise ValueError("Unknown objective [{}]".format(objtype))

    return objective


class Siamese(nn.Module):
    def __init__(self, encoder, objective, proj_layers=0, dropout=0.0, **kwargs):
        self.dropout = dropout
        super(Siamese, self).__init__()

        self.encoder = encoder

        self.proj = None
        if proj_layers > 0:
            self.proj = Highway(encoding_size, num_layers=proj_layers, dropout=dropout)

        self.objective = make_objective(objective, encoder.encoding_size[1], **kwargs)

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
            scores.extend(score.data.tolist())
            trues.extend(label.data.tolist())

            # prediction doesn't make sense for distance-based objectives
            if hasattr(self.objective, 'predict'):
                pred = self.objective.predict(score)
                preds.extend(pred.data.tolist())

        return scores, trues, preds
