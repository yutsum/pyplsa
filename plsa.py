import torch
import numpy as np
import functools
import scipy
import scipy.optimize

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def normalize(mat, dim=()):
    """ Normalize matrix by summing-up with specified dim
    """
    s = mat.sum(dim=dim, keepdim=True)
    s[s == 0] = 1
    return mat / s


def unsqueeze_ot(mat, i_s, n):
    """ Unsqueeze every dimention from 0 to (n - 1) in turn other than i_s(list) dimensions

        >>> unsqeeze_ot(torch.tensor([1.0, 2.0]), [0], 2)
        tensor([[1.0], [2.0]]) # [2, 1]
        >>> unsqeeze_ot(torch.tensor([1.0, 2.0]), [], 2)
        tensor([[[1.0, 2.0]]) # [1, 1, 2]
        >>> unsqeeze_ot(torch.tensor([1.0, 2.0]), [1], 2)
        tensor([1.0, 2.0]) #  [2], because 0-th dimension is not squeezed
        >>> unsqeeze_ot(torch.tensor([1.0, 2.0]), [1], 3)
        tensor([[[1.0], [2.0]]]) #  [1, 2, 1]
    """
    r = mat.clone()
    for j in range(n):
        if not(j in i_s):
            r.unsqueeze_(j)
    return r


def complement(x1, x2):
    r = x1.copy()
    for e in x2:
        if e in x1:
            r.remove(e)
    return r


def run_plsa_numpy(
        comat, nclass, niter,
        seed=0, nintvl_lik=None, with_pz_given_xi=False):
    p = PLSA(comat, nclass, seed)
    p.em_algorithm(niter, nintvl_lik=nintvl_lik)
    n = p.data.dim()
    r = {
        'pz': p.pz.numpy(),
        'pxi_given_zs': [v.numpy() for v in p.pxi_given_zs],
        'loglik': p.loglik
    }

    if with_pz_given_xi:
        pzxs = p.calc_pzxs()
        r.update({'pz_given_xi': [
            normalize(
                pzxs.sum(dim=complement(list(range(1, n+1)), [i+1])),
                [0]).numpy()
            for i in range(p.data.dim())]})
    return r


def kl_divergence(p, q, dim=0):
    # return (p * (p / q).log()).sum(dim=dim)
    m = (p / q).log()
    m[q == 0] = 9999
    m[p == 0] = 0
    return (p * m).sum(dim=dim)


def cluster_matching(ps1, ps2):
    # p(xi|z) is given as list {p(xi|z) \in M(nz x nxi) ; i}
    return


def calc_pxiall_given_z(pxi_given_zs):
    n = len(pxi_given_zs)
    ps = [unsqueeze_ot(pxi_given_zs[j], [0, j + 1], n + 1)
          for j in range(len(pxi_given_zs))]
    return normalize(functools.reduce(
            torch.mul, ps[1:], ps[0]), tuple(range(1, len(pxi_given_zs) + 1)))


class PLSA:
    def __init__(self, data, nclass, seed=0):
        """ Multidimensional PLSA class (*)
            intialized with co-occurance array and number of clusters
            (*)  P(x_1, ..., x_m, c) = \Prod_i P(x_i|c) P(c),
                 m is the dimension of the co-occurance array, typically two.
            >>> p1 = PLSA(data, nclass)
            >>> p1.em_algorith(100)  # EM-Algothim iteration 100 times
            >>> print(p1.pz)         # P(Z) : probability of each clusters
            >>> print(p1.pxi_given_zs)  # [P(x_i|z) \\in M(size of x_i, nclusters)]
            >>> print(p1.loglik)     # list of log likelihood during EM-Alg.
        """
        self.data = torch.tensor(data, dtype=torch.float).to(device)
        self.nclass = nclass
        self.seed = seed
        self.reset()

    def reset(self):
        torch.random.manual_seed(self.seed)
        nclass = self.nclass
        self.init_pz = normalize(torch.ones(nclass)).to(device)
        self.init_pxi_given_zs = [normalize(torch.rand(nclass, n), 1).to(device)
                                  for n in self.data.size()]
        self.pz = self.init_pz
        self.pxi_given_zs = self.init_pxi_given_zs
        self.loglik = {}

    def calc_pzxs(self):
        n = self.data.dim()
        # construct probability tensor list to be mltiplied with broadcast
        # note: v1.ger(v2) == v1.unsqueeze(1) * v2.unsqueeze(0)
        ps = [unsqueeze_ot(self.pz, [0], n + 1)] +\
             [unsqueeze_ot(self.pxi_given_zs[j], [0, j + 1], n + 1)
              for j in range(len(self.pxi_given_zs))]
        return functools.reduce(torch.mul, ps[1:], ps[0])

    def em_algorithm(self, niter, nintvl_lik=None):
        if nintvl_lik is None:
            nintvl_lik = max(np.floor(niter / 20.0), 1)
        for i in range(niter):
            # E-Step
            n = self.data.dim()
            pzxs = self.calc_pzxs()
            pz_given_xs = normalize(pzxs, 0)
            # Record log likelihood
            if i % nintvl_lik == 0:
                pxs = pzxs.sum(dim=0)
                pxs[pxs == 0] = 1
                ll = (pxs.log() * self.data).sum().item()
                self.loglik[i] = ll
            # M-Step
            tmp = pz_given_xs * self.data
            self.pxi_given_zs = [
                normalize(torch.sum(tmp, [j + 1 for j in range(n) if j != k]), 1)
                for k in range(n)]
            self.pz = normalize(torch.sum(tmp, list(range(1, n + 1))))

    # [FIXME][BROKEN] Draft
    def ga_optimize(self, **args):
        def func(x):
            return x.sum()
        nparam = 10
        scipy.optimize.differential_evolution(
            func,
            [(0, 1) for i in range(nparam)], **args)
