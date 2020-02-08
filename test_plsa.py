import unittest
import plsa
import torch


def tdiff(m1, m2):
    return torch.sum(abs(m1 - m2))


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


class TestPLSA(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        data1 = torch.tensor([[3, 0, 0], [1, 0, 0], [0, 1, 1]])
        nclass = 2
        self.plsa1 = plsa.PLSA(data1, nclass)

    def test_em_algorithm_yet(self):
        self.plsa1 = plsa.PLSA(self.plsa1.data, len(self.plsa1.pz))
        self.plsa1.em_algorithm(2)
        ans = [torch.tensor([[0.7500, 0.2500, 0.0000], [0.0000, 0.0000, 1.0000]]),
               torch.tensor([[1.0000, 0.0000, 0.0000], [0.0000, 0.5000, 0.5000]])]
        for i in range(len(ans)):
            self.assertNotEqual(0, tdiff(self.plsa1.pxi_given_zs[i], ans[i]))

    def test_em_algorithm(self):
        self.plsa1.em_algorithm(10)
        ans = [torch.tensor([[0.7500, 0.2500, 0.0000], [0.0000, 0.0000, 1.0000]]),
               torch.tensor([[1.0000, 0.0000, 0.0000], [0.0000, 0.5000, 0.5000]])]
        print(self.plsa1.loglik)
        for i in range(len(ans)):
            self.assertEqual(0, tdiff(self.plsa1.pxi_given_zs[i], ans[i]))


if __name__ == '__main__':
    unittest.main()
