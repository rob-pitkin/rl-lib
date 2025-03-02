import unittest
from network import QNetwork
import torch


class TestQNetwork(unittest.TestCase):
    def test_input_params(self):
        net = QNetwork(1, 1, "relu", [1])
        self.assertTrue(hasattr(net, "layers"))
        self.assertTrue(len(list(net.layers)) > 0)

    def test_forward(self):
        with torch.no_grad():
            torch.manual_seed(1)
            net = QNetwork(3, 3, "relu", [1])
            t = torch.randn(1, 3)
            res = net.forward(t)
            res1 = net.forward(t)
            self.assertEqual(res.shape, torch.Size((1, 3)))
            self.assertTrue(torch.equal(res, res1))

    def test_backward_pass(self):
        net = QNetwork(3, 4, "relu")
        x = torch.randn(1, 3, dtype=torch.float32, requires_grad=True)
        target = torch.randn(1, 4, dtype=torch.float32)
        obj = torch.nn.MSELoss()

        out = net.forward(x)
        loss = obj(out, target)
        loss.backward()

        self.assertTrue(x.grad is not None)


if __name__ == "__main__":
    unittest.main()
