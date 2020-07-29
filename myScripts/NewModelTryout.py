import os
import pickle
import re
from pathlib import Path
import matplotlib.pyplot as plt
import babyai
import gym
import numpy as np
import torch
from babyai.utils.demos import transform_demos
from myScripts.MyACModel import ACModelRudder
from myScripts.ReplayBuffer import ProcessData
from torch.nn.utils.rnn import pad_sequence


class Obs_Object:
    def __init__(self):
        self.image = None
        self.instr = None
        self.done = None


class Revolution:
    def __init__(self):
        self.entropy_coef = 0.01
        env = gym.make("BabyAI-PutNextLocal-v0")
        self.preprocess_obss = babyai.utils.ObssPreprocessor('newDataColl0.01', env.observation_space,
                                                             'newDataColl0.01')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_dim = 128
        self.train_same_batch = True
        self.batch_size = 1
        self.use_actions = False
        self.train_epochs=100
        # self.batches_per_epoch = int(10000/self.batch_size)
        self.batches_per_epoch = 30

        self.acmodel = ACModelRudder(self.preprocess_obss.obs_space, env.action_space, self.use_actions,
                                     memory_dim=self.memory_dim,
                                     use_memory=True).to(self.device)
        self.lr = 1e-2
        self.max_grad_norm = 0.5
        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), self.lr)
        self.max_sequence_length = 128
        shape = (self.batch_size, self.max_sequence_length)

        self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.quality_threshold = 0.8
        self.aux_loss_multiplier = 0.5
        self.mu = 1
        self.episodes = None

    def train_one_batch(self, obs):
        """

        :param obs: dict with padded images and instructions ( steps x batch size)
        :return:
        """
        memory = torch.zeros(1, self.acmodel.memory_dim)
        memory = torch.cat([torch.zeros(self.batch_size, self.acmodel.memory_dim),
                            torch.zeros(self.batch_size, self.acmodel.memory_dim)],
                           dim=1).to(self.device)
        full_predictions = []
        dists = []
        actions = []
        for i in range(len(obs)):
            # print(i)
            model_results = self.acmodel(obs[i], memory)
            value = model_results['value']
            memory = model_results['memory']
            dists.append(model_results["dist"])
            full_predictions.append(value)
            actions.append(obs[i].action)
        return full_predictions, dists, actions
        # print("checkpoint 5")

    def calc_quality(self, diff):
        # diff is g - gT_hat -->  see rudder paper A267
        quality = 1 - (torch.abs(diff) / self.mu) * 1 / (1 - self.quality_threshold)
        quality = torch.min(quality)
        return quality

    def paper_loss3(self, predictions, returns, pred_plus_ten_ts):

        diff = predictions[-1] - returns
        # Main task: predicting return at last timestep
        quality = self.calc_quality(diff)
        main_loss = diff ** 2
        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        continuous_loss = torch.mean((predictions[:, :] - returns.repeat(predictions.shape[0], 1)) ** 2)
        # continuous_loss = self.mse_loss(predictions[:, :], returns[..., None])

        # loss Le of the prediction of the output at t+10 at each time step t
        le10_loss = 0.0
        # if episode is smaller than 10 the follwoing would produce a NAN
        # if predictions.shape[1] > 10:
        #     pred_chunk = predictions[:, 10:]
        #     le10_loss = torch.mean((pred_chunk - pred_plus_ten_ts[:, :-10]) ** 2)

        # le10_loss = self.mse_loss(pred_chunk, pred_plus_ten_ts[:, :-10])

        # Combine losses
        aux_loss = continuous_loss  # + le10_loss
        loss = main_loss + self.aux_loss_multiplier * aux_loss
        return loss, quality, (main_loss.detach().clone().mean(), aux_loss.detach().clone().mean())

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
            p.instructions = torch.from_numpy(np.array([vocab[token] for token in tokens])).repeat(len(demo), 1).to(
                self.device)
            # dummy elements
            p.embeddings = torch.zeros(1)
            p.values = torch.zeros(1)
            assert len(p.dones) == len(p.rewards) == len(p.actions) == len(p.instructions) == len(p.images) == len(
                demo)
            transformed.append(p)
        return transformed

    def my_reverse_pad_sequence(self, sequences, batch_first=False, padding_value=0):
        r""" like original pad_sequence but zero padding at the beginning not at the end
        """

        # assuming trailing dimensions and type of all the Tensors
        # in sequences are same and fetching those from sequences[0]
        max_size = sequences[0].size()
        trailing_dims = max_size[1:]
        max_len = max([s.size(0) for s in sequences])
        if batch_first:
            out_dims = (len(sequences), max_len) + trailing_dims
        else:
            out_dims = (max_len, len(sequences)) + trailing_dims

        out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            # use index notation to prevent duplicate references to the tensor
            if batch_first:
                out_tensor[i, -length:, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor

        return out_tensor

    def pad_generic(self, prefilled, lst, zeros_upfront):
        if zeros_upfront:
            for i, done_list in enumerate(lst):
                prefilled[i][-len(done_list):] = done_list
        else:
            for i, done_list in enumerate(lst):
                prefilled[i][:len(done_list)] = done_list
        return prefilled

    def pad_dones(self, dones, zeros_upfront=False):
        padded_dones = np.ones((self.batch_size, self.max_sequence_length), dtype=bool)
        return self.pad_generic(padded_dones, dones, zeros_upfront)

    def pad_rewards(self, rewards, zeros_upfront=False):
        padded_rewards = torch.zeros(self.batch_size, self.max_sequence_length)
        return self.pad_generic(padded_rewards, rewards, zeros_upfront)

    def update_parameters(self, batch_loss):
        self.optimizer.zero_grad()
        # APEX
        # with amp.scale_loss(batch_loss, self.optimizer) as scaled_loss:
        #     scaled_loss.backward()
        batch_loss = batch_loss.mean()
        batch_loss.backward()
        grad_norm = sum(
            p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()
        # print("loss", batch_loss.item())
        return batch_loss.detach()

    def prepare_batch(self, step,episodes):
        # caching
        # if not self.episodes:
        #     self.episodes = self.load_generated_demos("../scripts/demos/BabyAI-PutNextLocal-v0_agent.pkl")
        #
        # episodes = self.episodes
        batch = episodes[self.batch_size * step:self.batch_size * (step + 1)]
        images = [e.images for e in batch]
        instructions = [e.instructions for e in batch]
        dones = [e.dones for e in batch]
        padded_dones = self.pad_dones(dones, zeros_upfront=True)
        rewards = [e.rewards for e in batch]
        rewards = self.pad_rewards(rewards, zeros_upfront=True)
        actions = [e.actions + 1 for e in batch]

        for a in actions:
            assert 0 not in a

        actions = self.my_reverse_pad_sequence(actions, batch_first=True)
        instructions = self.my_reverse_pad_sequence(instructions, batch_first=True)
        images = self.my_reverse_pad_sequence(images, batch_first=True)
        dones = padded_dones
        max_padded_len = images.shape[1]
        image_shape = episodes[0].images[0].shape
        instr_shape = episodes[0].instructions[0].shape
        # padding_obj = Obs_Object()
        # padding_obj.image = torch.zeros(image_shape).unsqueeze(0)
        # padding_obj.instr = torch.zeros(instr_shape).unsqueeze(0)
        # padding_obj.done = True
        #
        # obss = [padding_obj for i in range(self.max_sequence_length)]
        # episode = episodes[0]
        obss = []
        for i in range(max_padded_len):
            mask = 1 - torch.tensor(dones[:, i], device=self.device, dtype=torch.float)
            obs = Obs_Object()
            obs.image = images[:, i]  # *mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            obs.instr = instructions[:, i]  # *mask.unsqueeze(1)
            obs.done = dones[:, i]
            obs.action = actions[:, i]
            obss.append(obs)
        return obss, rewards

    def plot_losses(self,main_losses,aux_losses,run):

        plt.title("bs " + str(self.batch_size) +
                  " lr " + str(self.lr) + " incl actions "
                  + str(self.use_actions) + " same batch "
                  + str(self.train_same_batch))
        plt.yscale('log')
        plt.plot(main_losses, label="main")
        plt.plot(aux_losses, label="aux")
        plt.legend(loc="upper left")
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(19.2, 10.8)
        path = "/home/nick/Documents/jku/Ma/noActionsMultipleBatch/"
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + "action" + str(run), dpi=100)
        plt.close()

    def calculate_losses(self,batch_nr,episodes):
        # print("collect data")
        if self.train_same_batch:
            obss, rewards = self.prepare_batch(0,episodes)
        else:
            obss, rewards = self.prepare_batch(batch_nr,episodes)
        # print("train")
        predictions, dists, actions = self.train_one_batch(obss)
        # predictions=torch.stack(predictions,dim=0).to(self.device)
        # returns=rewards.sum(dim=1).to(self.device)
        # batch_loss,_,(main_l,aux_l) =self.paper_loss3(predictions,returns,None )
        # print("pred",predictions[-1].mean())
        # print("calc loss")
        batch_loss, accuracy = self.imitation_loss(dists, actions)
        # print("acc",accuracy)
        # print("update params")
        loss = self.update_parameters(batch_loss)
        # main_losses.append(loss)
        # aux_losses.append(accuracy)
        # main_losses.append(main_l)
        # aux_losses.append(aux_l)
        return loss, accuracy

    def start(self, run,files):
        main_losses = []
        aux_losses = []
        for i in range(self.train_epochs):
            print(i,"/",self.train_epochs)
            for file in files:
                episodes = self.load_generated_demos(file)
                for batch_nr in range(self.batches_per_epoch):
                    main_l,aux_l = self.calculate_losses(batch_nr,episodes)
                    main_losses.append(main_l)
                    aux_losses.append(aux_l)


        self.plot_losses(main_losses,aux_losses,run)

    def imitation_loss(self, dists, actions):
        final_loss = 0
        total_frames = self.batch_size * len(actions)
        final_entropy, final_policy_loss, final_value_loss = 0, 0, 0
        accuracy = 0
        for index, dist in enumerate(dists):
            action_step = actions[index]
            entropy = dist.entropy().mean()
            policy_loss = -dist.log_prob(action_step).mean()
            loss = policy_loss - self.entropy_coef * entropy
            action_pred = dist.probs.max(1, keepdim=True)[1]
            accuracy += float((action_pred == action_step.unsqueeze(1)).sum()) / total_frames
            final_loss += loss
            final_entropy += entropy
            final_policy_loss += policy_loss

        final_loss /= len(actions)
        return final_loss,accuracy


for i in range(3):
    r = Revolution()
    path="/home/nick/PycharmProjects/babyRudder/babyai/scripts/demos/botDemos"
    files = os.listdir(path)
    files_without_dir=[]
    for file in files:
        if os.path.isfile(path + file):
            files_without_dir.append(path+file)
    r.start(i,files_without_dir)
