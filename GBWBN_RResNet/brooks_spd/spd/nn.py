import torch as th
import torch.nn as nn
from . import functional
import geoopt
from geoopt.manifolds.symmetric_positive_definite import SymmetricPositiveDefinite
from geoopt.manifolds import Stiefel

dtype = th.double
device = th.device('cpu')


class BiMap(nn.Module):
    """
    Input X: (batch_size,hi) SPD matrices of size (ni,ni)
    Output P: (batch_size,ho) of bilinearly mapped matrices of size (no,no)
    Stiefel parameter of size (ho,hi,ni,no)
    """

    def __init__(self, ho, hi, ni, no):
        super(BiMap, self).__init__()
        self._W = geoopt.ManifoldParameter(th.empty(ho, hi, ni, no, device=device, dtype=dtype), manifold=Stiefel())
        self._ho = ho
        self._hi = hi
        self._ni = ni
        self._no = no
        functional.init_bimap_parameter(self._W)

    def forward(self, X):
        return functional.bimap_channels(X, self._W)


class LogEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of log eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return functional.LogEig.apply(P)


class SqmEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of sqrt eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return functional.SqmEig.apply(P)


class ReEig(nn.Module):
    """
    Input P: (batch_size,h) SPD matrices of size (n,n)
    Output X: (batch_size,h) of rectified eigenvalues matrices of size (n,n)
    """

    def forward(self, P):
        return functional.ReEig.apply(P)


class BaryGeom(nn.Module):
    '''
    Function which computes the Riemannian barycenter for a batch of data using the Karcher flow
    Input x is a batch of SPD matrices (batch_size,1,n,n) to average
    Output is (n,n) Riemannian mean
    '''

    def forward(self, x):
        return functional.BaryGeom(x)


class BatchNormSPD(nn.Module):
    """
    Input X: (N,h) SPD matrices of size (n,n) with h channels and batch size N
    Output P: (N,h) batch-normalized matrices
    SPD parameter of size (n,n)
    """

    def __init__(self, n, po=0.5):
        super(__class__, self).__init__()
        self.momentum = 0.1
        self.running_mean = th.eye(n, dtype=dtype, device=device)
        eyes = th.eye(n, n, dtype=dtype, device=device)
        diag_indices = th.arange(n)
        eyes[diag_indices, diag_indices] += th.arange(1, n + 1) * 1e-7
        self.weight = geoopt.ManifoldParameter(eyes, manifold=SymmetricPositiveDefinite())
        self.M = geoopt.ManifoldParameter(th.eye(n, n, dtype=dtype, device=device),
                                          manifold=SymmetricPositiveDefinite())
        self.running_var = nn.Parameter(th.ones(n, dtype=dtype, device=device))
        self.shift = nn.Parameter(th.ones(1, dtype=dtype, device=device))
        self.eps = 1e-05
        self.pow = th.tensor(po, dtype=dtype)

    def forward(self, X):
        N, h, n, n = X.shape
        X_batched = X.permute(2, 3, 0, 1).contiguous().view(n, n, N * h, 1).permute(2, 3, 0, 1).contiguous()
        X_batched = functional.PowerEig.apply(X_batched, self.pow)
        X_batched = functional.CongrG(X_batched, self.M, 'neg')
        weight = self.weight
        weight = functional.PowerEig.apply(weight[None, None, :, :], self.pow)[0, 0]
        weight = functional.CongrG1(weight, self.M, 'neg')
        if self.training:
            mean = functional.BaryGeom(X_batched)
            var = functional.cal_var(X_batched, self.pow)
            with th.no_grad():
                self.running_mean.data = functional.geodesic(self.running_mean, mean, self.momentum)
                self.running_var.data = (1 - self.momentum) * self.running_var + self.momentum * var
            X_centered = functional.pal1(mean, functional.LogG(X_batched, mean))
            X_centered = functional.scale1(X_centered, var, self.eps, self.shift)
        else:
            X_centered = functional.pal1(self.running_mean, functional.LogG(X_batched, self.running_mean))
            X_centered = functional.scale1(X_centered, self.running_var, self.eps, self.shift)
        X_normalized = functional.ExpG(functional.pal2(self.weight, functional.LogG1(X_centered)), weight)
        X_normalized = functional.CongrG(X_normalized, self.M, 'pos')
        X_normalized = functional.PowerEig.apply(X_normalized, 1 / self.pow)
        return X_normalized.permute(2, 3, 0, 1).contiguous().view(n, n, N, h).permute(2, 3, 0, 1).contiguous()
