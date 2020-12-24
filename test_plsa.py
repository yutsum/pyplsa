import unittest
import plsa
import torch
import time


def tdiff(m1, m2):
    return torch.sum(abs(m1.cpu() - m2.cpu()))


class TestUtility(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.m1 = torch.tensor([[1.0, 1], [3, 4]])
        self.m2 = torch.stack((torch.tensor([[1.0, 1], [3, 4]]), 4*self.m1))
        self.m3 = torch.tensor([1.0, 2.0])

    def test_normalize_all(self):
        self.assertTrue((self.m1 / 9).eq(plsa.normalize(self.m1)).all())

    def test_normalize_0(self):
        self.assertTrue(
            torch.tensor([[1.0/4, 1/5], [3/4, 4/5]]).eq(
                plsa.normalize(self.m1, 0)).all())

    def test_normalize_0_ng(self):
        self.assertFalse(
            (1.1*torch.tensor([[1.0/4, 1/5], [3/4, 4/5]])).eq(
                plsa.normalize(self.m1, 0)).all())

    def test_normalize_1(self):
        self.assertAlmostEqual(
            torch.sum(abs(torch.tensor(
                [[1.0/2, 1/2], [3/7, 4/7]]) - plsa.normalize(self.m1, 1))), 0)

    def test_normalize_3_0(self):
        self.assertTrue(
            (self.m2 / (5*self.m2[0, :, :])).eq(plsa.normalize(self.m2, 0)).all())

    def test_unsqueeze_ot_0(self):
        self.assertEqual(0, tdiff(
            torch.tensor([[1.0], [2]]),
            plsa.unsqueeze_ot(self.m3, [0], 2)))

    def test_unsqueeze_ot_1(self):
        self.assertEqual(0, tdiff(
            torch.tensor([[1.0, 2]]), plsa.unsqueeze_ot(self.m3, [1], 2)))

    def test_kl_divergence_1(self):
        self.assertEqual(0, plsa.kl_divergence(
            torch.tensor([0.1, 0.2, 0.3]),
            torch.tensor([0.1, 0.2, 0.3])))

    def test_kl_divergence_2(self):
        self.assertAlmostEqual(0.1386294365, plsa.kl_divergence(
            torch.tensor([0.2, 0.2, 0.3]),
            torch.tensor([0.1, 0.2, 0.3])))

    def test_kl_divergence_3(self):
        self.assertAlmostEqual(0, tdiff(
            torch.tensor([0.04440307, 0, 0]),
            plsa.kl_divergence(
                torch.tensor([[0.2, 0.2, 0.3], [0.8, 0.8, 0.7]]),
                torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]))))


class TestPLSA(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        data1 = torch.tensor([[3, 0, 0], [1, 0, 0], [0, 1, 1]])
        self.data1 = data1
        nclass = 2
        self.plsa1 = plsa.PLSA(data1, nclass)
        data2 = torch.tensor(
            [
                [[3, 0, 0], [1, 0, 0], [0, 0, 0]],
                [[3, 0, 0], [1, 0, 0], [0, 1, 1]],
                [[0, 0, 0], [0, 0, 0], [0, 1, 1]]
            ])
        nclass = 2
        self.plsa2 = plsa.PLSA(data2, nclass)

    def test_em_algorithm_yet(self):
        self.plsa1 = plsa.PLSA(self.plsa1.data, len(self.plsa1.pz))
        self.plsa1.em_algorithm(2)
        ans = [torch.tensor([[0.7500, 0.2500, 0.0000], [0.0000, 0.0000, 1.0000]]),
               torch.tensor([[1.0000, 0.0000, 0.0000], [0.0000, 0.5000, 0.5000]])]
        for i in range(len(ans)):
            self.assertNotEqual(0, tdiff(self.plsa1.pxi_given_zs[i], ans[i]))

    def test_em_algorithm(self):
        self.plsa1.em_algorithm(10)
        ans = [torch.tensor([[0.75, 0.25, 0.00], [0.00, 0.00, 1.00]]),
               torch.tensor([[1.00, 0.00, 0.00], [0.00, 0.50, 0.50]])]
        # print(self.plsa1.loglik)
        for i in range(len(ans)):
            self.assertEqual(0, tdiff(self.plsa1.pxi_given_zs[i], ans[i]))

    def test_em_algorithm_dim0(self):
        # Example of document likelihood estimation with given p(z) and p(x1|z)
        self.plsa1.data = torch.tensor([[4, 0, 0], [1, 5, 0], [1, 2, 1]],
                                       dtype=torch.float)
        self.plsa1.reset(inds_fixed=[-1, 0])
        self.plsa1.em_algorithm(10, inds_fixed=[-1, 0])
        ans = [torch.tensor([[0.75, 0.25, 0.00], [0.00, 0.00, 1.00]]),
               torch.tensor([[0.50, 0.50, 0.00], [0.25, 0.50, 0.25]])]
        # print(self.plsa1.loglik)
        for i in range(len(ans)):
            self.assertEqual(0, tdiff(self.plsa1.pxi_given_zs[i], ans[i]))

    def test_em_algorithm_3d(self):
        self.plsa2.em_algorithm(20)
        ans = [
            torch.tensor([[0.00, 0.50, 0.50], [0.50, 0.50, 0.00]]),
            torch.tensor([[0.00, 0.00, 1.00], [0.75, 0.25, 0.00]]),
            torch.tensor([[0.00, 0.50, 0.50], [1.00, 0.00, 0.00]])]
        # print(self.plsa2.loglik)
        for i in range(len(ans)):
            self.assertEqual(0, tdiff(self.plsa2.pxi_given_zs[i], ans[i]))

    def test_em_algorithm_time(self):
        data1 = 10*torch.rand((10000, 100))
        nclass =8 
        p1 = plsa.PLSA(data1, nclass)
        start = time.time()
        p1.em_algorithm(8)
        end = time.time()
        # print(self.plsa1.loglik)
        self.assertLessEqual(end - start, 1)  # less than 1 sec

    def test_run_plsa_numpy(self):
        ans = plsa.run_plsa_numpy(self.data1, 2, 10)
        # print(ans)
        pass

if __name__ == '__main__':
    unittest.main()
