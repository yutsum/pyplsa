import torch
import numpy as np
import functools

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def normalize(mat, dim=()):
    s = mat.sum(dim=dim, keepdim=True)
    s[s == 0] = 1
    return mat / s


def unsqueeze_n(mat, n):

    for i in range(n):
        r = mat.unsqueeze(mat.dim())
    return r


def unsqueeze_ot(mat, i_s, n):
    r = mat.clone()
    for j in range(n):
        if not(j in i_s):
            r.unsqueeze_(j)
    return r


class PLSA:
    def __init__(self, data, nclass, seed=0):
        # self.init_pz = normalize(torch.tensor(range(1, nclass+1)).float()).to(device)
        self.init_pz = normalize(torch.ones(nclass)).to(device)
        self.data = torch.tensor(data, dtype=torch.float).to(device)
        torch.random.manual_seed(seed)
        self.init_pxi_given_zs = [torch.rand(nclass, n).to(device) / nclass
                                    for n in self.data.size()]
        self.pz = self.init_pz
        self.pxi_given_zs = self.init_pxi_given_zs

    def em_algorithm(self, niter):
        for i in range(niter):
            # E-Step
            n = self.data.dim()
            ps = [unsqueeze_ot(self.pz, [0], n + 1)] +\
                    [unsqueeze_ot(self.pxi_given_zs[j], [0, j + 1], n + 1)
                        for j in range(len(self.pxi_given_zs))]
            self.ps = ps
            self.pzxs = functools.reduce(torch.mul, ps[1:], ps[0])
            self.pz_given_xs = normalize(self.pzxs, 0)
            # M-Step
            tmp = self.pz_given_xs * self.data
            self.tmp = tmp
            self.pxi_given_zs = [
                normalize(torch.sum(tmp, [j + 1 for j in range(n) if j != k]), 1)
                    for k in range(n)]
            self.pz = normalize(torch.sum(tmp, list(range(1, n + 1))))
