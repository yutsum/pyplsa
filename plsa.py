import torch
import numpy as np
import functools

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def normalize(mat, dim=()):
    """ Normalize matrix by summing-up with specified dim
    """
    s = mat.sum(dim=dim, keepdim=True)
    s[s == 0] = 1
    return mat / s


def unsqueeze_ot(mat, i_s, n):
    """ Unsqueeze other than i_s(list) dimensions up to n

        >>> unsqeeze_ot(torch.tensor([1.0, 2.0]), 0, 2)
        tensor([[1.0], [2.0]])
    """
    r = mat.clone()
    for j in range(n):
        if not(j in i_s):
            r.unsqueeze_(j)
    return r


def run_plsa(comat, nclass, niter, seed=0):
    p = PLSA(comat, nclass, seed)
    p.em_algorithm(niter)
    return {'pz': p.pz, 'pxi_given_zs': p.pxi_given_zs}


class PLSA:
    def __init__(self, data, nclass, seed=0):
        """
            >>> p1 = PLSA(data, nclass)
            >>> p1.em_algorith(100)
            >>> print(p1.pz)
            >>> print(p1.pxi_given_zs)
        """
        self.data = torch.tensor(data, dtype=torch.float).to(device)
        self.nclass = nclass
        self.seed = seed
        self.reset()

    def reset(self):
        torch.random.manual_seed(self.seed)
        nclass = self.nclass
        self.init_pz = normalize(torch.ones(nclass)).to(device)
        self.init_pxi_given_zs = [torch.rand(nclass, n).to(device) / nclass
                                  for n in self.data.size()]
        self.pz = self.init_pz
        self.pxi_given_zs = self.init_pxi_given_zs
        self.loglik = []

    def calc_pzxs(self):
        n = self.data.dim()
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
            if i % nintvl_lik == 0:
                pxs = pzxs.sum(dim=0)
                pxs[pxs == 0] = 1
                ll = (pxs.log() * self.data).sum().item()
                self.loglik = self.loglik + [ll]
            pz_given_xs = normalize(pzxs, 0)
            # M-Step
            tmp = pz_given_xs * self.data
            self.pxi_given_zs = [
                normalize(torch.sum(tmp, [j + 1 for j in range(n) if j != k]), 1)
                for k in range(n)]
            self.pz = normalize(torch.sum(tmp, list(range(1, n + 1))))
