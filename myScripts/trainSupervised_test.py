import unittest

import numpy as np

from myScripts.ReplayBuffer import ProcessData
from .trainSupervised import Training

class MyTestCase(unittest.TestCase):

    def test_create_partial_episodes_from_failed_episode(self):
        # should return 4 episodes with lengths of 32,64,96,128
        # last one is original episode
        training = Training()
        new_episode = ProcessData()
        new_episode.rewards = np.zeros(128)
        new_episode.actions = np.zeros(128)
        new_episode.images = np.zeros(128)
        new_episode.instructions = np.zeros(128)
        new_episode.dones = np.zeros(128)
        new_episode.mission = "test mission"
        lst = training.create_partial_episodes_from_failed_episode(new_episode)
        self.assertEqual(4,len(lst))
        self.assertEqual(len(lst[0].dones),32)
        self.assertEqual(len(lst[1].dones),32*2)
        self.assertEqual(len(lst[2].dones),32*3)
        self.assertEqual(len(lst[3].dones) , 32*4)






if __name__ == '__main__':
    unittest.main()
