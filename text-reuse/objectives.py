
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BaseContrastiveObjective(nn.Module):
    def __init__(self, weight=0.5, margin=2):
        self.weight, self.margin = weight, margin
        super(BaseContrastiveObjective, self).__init__()

    def loss(self, dists, y):
        # distance scaled down by negative sampling factor
        pos = self.weight * (dists ** 2)
        # distance must be at least as large as margin for negative examples
        neg = torch.clamp(self.margin - dists, min=0) ** 2

        return torch.mean(y * pos + (1 - y) * neg)

    def predict(self, pred):
        return pred


class ContrastiveCosineObjective(BaseContrastiveObjective):
    """
    Implementation from "Learning Text Similarity with Siamese Recurrent Networks"
    http://www.aclweb.org/anthology/W16-1617
    It diverges from original "contrastive loss" definition by Chopra et al 2005,
    based on euclidean distance instead of cosine (which makes sense for text).

    L_+(x_1, x_2, y) = 1/4 * (1 - cosine_similarity(x_1, x_2)) ^ 2
    L_-(x_1, x_2, y) = max(0, cosine_similarity(x_1, x_2) - margin) ^ 2

    where 1/4 is the weight which should be tune with respect to the proportion
    of positive versus negative examples.

    Note that 1 - cosine_similarity represents the cosine distance in range [0, 2].
    A different approach would be to 
    """
    def __init__(self, weight=0.5, margin=2):
        super(ContrastiveCosineObjective, self).__init__(weight=weight, margin=margin)

    def forward(self, enc1, enc2, **kwargs):
        # Angular distance in range [0, 1]
        # import math
        # return torch.acos(F.cosine_similarity(enc1, enc2, dim=1)) / math.pi
        return 1 - F.cosine_similarity(enc1, enc2, dim=1)


class ContrastiveEuclideanObjective(BaseContrastiveObjective):
    """
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    L_+(x_1, x_2, y) = 1/2 * euclidean_dist(x_1, x_2) ^ 2
    L_-(x_1, x_2, y) = 1/2 * max(0, margin - euclidean_dist(x_1, x_2)) ^ 2
    """
    def __init__(self, weight=0.5, margin=2):
        super(ContrastiveEuclideanObjective, self).__init__(weight=weight, margin=margin)

    def forward(self, enc1, enc2, **kwargs):
        return F.pairwise_distance(enc1, enc2)


class ContrastiveMahalanobisObjective(nn.Module):
    def __init__(self, encoding_size, weight=0.5, margin=1):
        super(ContrastiveMahalanobisObjective, self).__init__(weight=weight, margin=margin)
        # PSD weights
        self.W = nn.Parameter(torch.Tensor(encoding_size, encoding_size))

    def custom_init(self):
        torch.nn.init.xavier_normal(self.W, gain=1)

    def forward(self, enc1, enc2, **kwargs):
        """
        Mahalanobis distance with learned covariance matrix:

        \sqrt((x_1 - x_2)^T * (W^T*W) * (x_1 - x_2))
        """
        dists = enc1 - enc2
        batch, encoding_size = dists.size()
        # get PSD projection
        M = (self.W @ self.W).unsqueeze(0).extend(batch, encoding_size, encoding_size)
        # project feature-wise distances
        output = M.bmm(dists.unsqueeze(2))  # (batch x encoding_size x 1)
        # dot product with feature-wise distances
        return (output.squeeze(2) * dists).sum(1)  # (batch)
