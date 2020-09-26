import unittest
import torch

from rudder.rudder_main import Rudder


class MyTestCase(unittest.TestCase):

    def test_repeated_reward(self):
        rudder = Rudder(1, "cuda", 1, 1, 1, 1, 1, None)
        tensor = torch.zeros(20)
        done = torch.zeros(20)
        done[5] = 1
        done[15] = 1
        done[-1] = 1
        tensor[5] = 0.9
        tensor[15] = 0.8
        tensor[-1] = 0.4
        tensor2 = tensor.clone()
        tensor[0:5] = tensor[5]
        tensor[6:15] = tensor[15]
        tensor[16:20] = tensor[-1]

        repeated_reward = rudder.create_repeated_reward([tensor2])

        self.assertTrue(torch.eq(repeated_reward, tensor).all())




if __name__ == '__main__':
    unittest.main()
