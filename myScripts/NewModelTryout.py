import pickle
import re

import babyai
import gym
import numpy as np
import torch
from babyai.utils.demos import transform_demos
from myScripts.MyACModel import ACModelRudder
from myScripts.ReplayBuffer import ProcessData


class Revolution:
    def __init__(self):
        env = gym.make("BabyAI-PutNextLocal-v0")
        self.preprocess_obss = babyai.utils.ObssPreprocessor('newDataColl0.01', env.observation_space, 'newDataColl0.01')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.acmodel = ACModelRudder(self.preprocess_obss.obs_space, env.action_space, use_memory=True)
        self.batch_size = 64
        self.max_sequence_length = 128
        shape = (self.batch_size, self.max_sequence_length)

        self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)


    def train_one_batch(self,obs):
        """

        :param obs: dict with padded images and instructions ( steps x batch size)
        :return:
        """
        for i in range(self.max_sequence_length):
            # Do one agent-environment interaction
            # print("checkpoint 4")
            done = obs[i]
            preprocessed_obs = self.preprocess_obss(obs, device=self.device)
            model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            dist = model_results['dist']
            value = model_results['value']
            memory = model_results['memory']
            extra_predictions = model_results['extra_predictions']
            embedding = model_results['embedding']
            rudder_value = model_results['rudder_value']
            logits = model_results['logits']

            action = dist.sample()
            # print("checkpoint 5")

    def load_generated_demos(self, path, max_steps=128):
        with open(path, "rb") as f:
            vocab = {"put": 1, "the": 2, "grey": 3, "key": 4, "next": 5, "to": 6, "red": 7,
                     "box": 8, "yellow": 9, "blue": 10, "green": 11, "purple": 12, "ball": 13}
            episodes = pickle.load(f)
        demos = transform_demos(episodes)
        device = "cuda"
        transformed = []
        for demo in demos:
            p = ProcessData()
            p.mission = demo[0][0]["mission"].lower()
            step_count = len(demo)
            reward = 1 - 0.9 * (step_count / max_steps)
            assert 0 <= reward < 1
            if step_count == max_steps:
                reward = 0
            p.rewards = torch.zeros(len(demo))
            p.rewards[-1] = reward
            p.dones = [el[2] for el in demo]

            images = np.array([action[0]["image"] for action in demo])
            images = torch.tensor(images, device=device, dtype=torch.float)
            p.images = images

            p.actions = torch.tensor([action[1] for action in demo], device="cuda", dtype=torch.long)
            tokens = re.findall("([a-z]+)", demo[0][0]["mission"].lower())
            assert p.mission.split() == tokens
            p.instructions = torch.from_numpy(np.array([vocab[token] for token in tokens])).repeat(len(demo), 1)
            # dummy elements
            p.embeddings = torch.zeros(1)
            p.values = torch.zeros(1)
            assert len(p.dones) == len(p.rewards) == len(p.actions) == len(p.instructions) == len(p.images) == len(
                demo)
            transformed.append(p)
        return transformed

    def start(self):
        episodes = self.load_generated_demos("../scripts/demos/train/1000")
        images=episodes[0].images
        instructions = episodes