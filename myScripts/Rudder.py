import torch
# from apex import amp
from myScripts.MyNet import Net
from myScripts.ReplayBuffer import ReplayBuffer, ProcessData
import numpy as np
# test test
import multiprocessing as mp
import logging

from torch.nn.utils import clip_grad_value_


class Rudder:
    def __init__(self, mem_dim, nr_procs, obs_space, instr_dim, ac_embed_dim, image_dim, action_space, device):
        self.clip_value = 0.5
        self.replay_buffer = ReplayBuffer(nr_procs, ac_embed_dim, device)
        self.train_timesteps = False
        self.net = Net(image_dim, obs_space, instr_dim, ac_embed_dim, action_space).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=1e-5)
        # self.optimizer = torch.optim.Adam(self.net.parameters())

        self.device = device
        self.first_training_done = False
        self.mu = 20
        self.quality_threshold = 0.8
        self.last_hidden = [None] * nr_procs
        # For the first timestep we will take (0-predictions[:, :1]) as redistributed reward
        self.last_predicted_reward = [None] * nr_procs
        self.parallel_train_done = False
        self.current_quality=0
        # mpl = mp.log_to_stderr()
        # mpl.setLevel(logging.INFO)
        self.updates = 0

        ### APEX
        # self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level="O1")

    def calc_quality(self, diff):
        # diff is g - gT_hat -->  see rudder paper A267
        quality = 1 - (np.abs(diff.item()) / self.mu) * 1 / (1 - self.quality_threshold)
        return quality

    def lossfunction(self, predictions, returns):
        diff = predictions[:, -1] - returns
        # Main task: predicting return at last timestep
        quality = self.calc_quality(diff)
        main_loss = torch.mean(diff) ** 2
        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        aux_loss = torch.mean(predictions[:, :] - returns[..., None]) ** 2
        # Combine losses
        loss = main_loss + aux_loss * 0.5
        return loss, quality

    def get_lstm_prediction(self, proc_id, data, done, batch):
        if not batch:
            data = data.get_timestep_data(len(data.dones) - 1)
        hidden = self.last_hidden[proc_id]
        pred, hidden = self.net(data, hidden, not self.train_timesteps)
        # 1 timestep samples are 2d
        # if not pred.ndim == 3:
        #     pred = pred.unsqueeze(0)
        # if not self.train_timesteps:
        try:
            pred = pred[-1][-1][-1]
        except:
            pred = pred[-1][-1]
        # print(pred.shape)

        if not self.last_predicted_reward[proc_id]:
            # first timestep
            pred_reward = 0 - pred
        else:
            pred_reward = pred - self.last_predicted_reward[proc_id]
        if done:
            self.last_hidden[proc_id] = None
            self.last_predicted_reward[proc_id] = None
        else:
            self.last_predicted_reward[proc_id] = pred
            self.last_hidden[proc_id] = hidden
        return pred_reward

    def get_transformer_prediction(self, proc_id, data, done):
        episode = self.replay_buffer.proc_data_buffer[proc_id]
        pred, _ = self.net(episode, None, True)
        # print(pred)
        try:
            pred = pred[-1][-1][-1]
        except:
            print(pred.shape)
            pred = pred[-1][-1]
        # print(pred)
        if self.last_predicted_reward[proc_id] == None:
            # first timestep
            pred_reward = 0 - pred
        else:
            pred_reward = pred - self.last_predicted_reward[proc_id]
        if done:
            # self.last_hidden[proc_id] = None
            self.last_predicted_reward[proc_id] = None
        else:
            self.last_predicted_reward[proc_id] = pred
            # self.last_hidden[proc_id] = hidden
        return pred_reward

    def predict_reward(self, embeddings, actions, rewards, dones, instructions, images):
        predictions = []
        use_transformer = False
        batch = not self.train_timesteps
        for proc_id, done in enumerate(dones):
            data = self.replay_buffer.proc_data_buffer[proc_id]
            with torch.no_grad():
                if use_transformer:
                    pred_reward = self.get_transformer_prediction(proc_id, data, done)
                else:
                    pred_reward = self.get_lstm_prediction(proc_id, data, done, batch)
                predictions.append(pred_reward)
        return torch.stack(predictions).squeeze()

    def predict_full_episode(self, episode: ProcessData):
        predictions, hidden = self.net(episode, None, True)
        return predictions

    def predict_every_timestep(self, episode: ProcessData):
        hidden = None
        predictions = []
        for i, done in enumerate(episode.dones):
            ts_data = episode.get_timestep_data(i)
            pred, hidden = self.net(ts_data, hidden)
            predictions.append(pred)
            if done:
                break
        # assert done == True
        predictions = torch.cat(predictions, dim=1)
        return predictions

    def feed_network_MOCK(self, episode: ProcessData):
        loss = np.random.uniform(0, 1)
        r = np.random.uniform(-20, 20)
        if r < 0:
            r = 0
        returns = r
        quality = 0.1

        return loss, returns, quality

    def feed_network(self, episode: ProcessData):
        if self.train_timesteps:
            predictions = self.predict_every_timestep(episode)
        else:
            predictions = self.predict_full_episode(episode)
        returns = torch.sum(episode.rewards, dim=-1)
        # returns = np.sum(episode.rewards)
        loss, quality = self.lossfunction(predictions, returns)

        return loss, returns, quality, predictions.detach().clone()

    def inference_and_set_metrics(self, episode: ProcessData):
        with torch.no_grad():
            # in train and set metrics rewards are a tensor

            # episode.rewards=torch.from_numpy(np.array(episode.rewards)).to(self.device)
            try:
                # print("episode.rewards",episode.rewards)
                episode.rewards = torch.stack(episode.rewards)
            except:
                pass
            assert isinstance(episode.rewards, torch.Tensor)

            loss, returns, quality, _ = self.feed_network(episode)
            loss = loss.detach().item()
            returnn = returns.detach().item()
            episode.loss = loss
            episode.returnn = returnn

            return quality

    def train_and_set_metrics_MOCK(self, episode):
        episode.loss = np.random.uniform(0, 1)
        r = np.random.uniform(-20, 20)
        if r < 0:
            r = 0
        episode.returnn = r
        quality = 0.1
        return quality

    def train_and_set_metrics(self, episode, episode_id):
        # loss, returnn, quality = self.train_one_episode(episode)
        loss, returns, quality, predictions = self.feed_network(episode)

        ### APEX
        # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #     scaled_loss.backward()
        ###
        loss.backward()
        clip_grad_value_(self.net.parameters(), self.clip_value)
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss = loss.detach().item()
        returnn = returns.detach().item()
        # print("Loss",loss)
        episode.loss = loss
        episode.returnn = returnn
        self.replay_buffer.fast_losses[episode_id] = loss
        # print("loss", loss)
        return quality, predictions

    def train_full_buffer(self):
        # print("train_full_buffer")
        losses = []
        full_predictions = []
        last_timestep_prediction = []
        last_rewards = []
        bad_quality = True
        while bad_quality:
            qualities_bools = set()
            qualities=[]
            for epoch in range(5):
                episodes, episodes_ids = self.replay_buffer.sample_episodes()
                # episodes = self.replay_buffer.sample_episodes()
                for i, episode in enumerate(episodes):
                    quality, predictions = self.train_and_set_metrics(episode, episodes_ids[i])
                    # quality=self.train_and_set_metrics(episode)
                    full_predictions.append(predictions.unsqueeze(0))
                    last_timestep_prediction.append(predictions[0][-1].item())
                    losses.append(episode.loss)
                    last_rewards.append(episode.rewards[-1].item())
                    # assert episode.returnn==self.replay_buffer.fast_returns[episodes_ids[i]]
                    qualities_bools.add(quality>0)
                    qualities.append(quality)
            self.current_quality=np.mean(qualities)
            print("sample {} return {:.2f} loss {:.6f}".format(episodes_ids[-1], episode.returnn, episode.loss))
            if False not in qualities_bools:
                bad_quality = False
        # full_predictions = torch.cat(full_predictions, dim=-1)
        return np.mean(losses), np.mean(last_timestep_prediction), np.mean(
            last_rewards)  # , torch.mean(full_predictions)

    def remove_uninteresting_return_episodes(self, complete_episodes):
        return [e for e in complete_episodes if e.rewards[-1] not in set(self.replay_buffer.get_returns())]

    def new_add_to_replay_buffer(self, complete_episodes, debug=False):
        replaced = False
        replaced_ids = set()
        # print("in new_add_to_replay_buffer")
        i = 0
        for ce in complete_episodes:
            if debug:
                print("inference_and_set_metrics", i)
                if i == 248:
                    print("so close")
                i += 1

            self.inference_and_set_metrics(ce)
            if self.replay_buffer.buffer_full():
                if debug:
                    print("new_get_replacement_index")
                idx = self.replay_buffer.new_get_replacement_index(ce)
            else:
                idx = self.replay_buffer.added_episodes
                self.replay_buffer.added_episodes += 1
                # assert idx!=self.replay_buffer
            if idx != -1:
                if debug:
                    print("new_replace_episode_data")
                self.replay_buffer.new_replace_episode_data(idx, ce)
                replaced = True
                replaced_ids.add(idx)
        if debug:
            print("after loop")
        # if replaced and self.replay_buffer.buffer_full():
        # print("replaced", replaced_ids)
        # self.train_full_buffer()
        # self.first_training_done=True
        # print('non zero returns', np.count_nonzero(self.replay_buffer.fast_returns))

        # if self.parallel_train_done:
        #     print("recalc")
        #     self.recalculate_all_losses()
        #     print("recalc done")
        # if self.updates%1==0:
        #     print("recalc")
        #     self.recalculate_all_losses()
        #     print("recalc done")
        # self.updates+=1

        # print("leaving new_add_to_replay_buffer")

    def recalculate_all_losses(self):
        for i in range(self.replay_buffer.max_size):
            episode = self.replay_buffer.get_episode_from_tensors(i)
            self.inference_and_set_metrics(episode)
            self.replay_buffer.fast_losses[i] = episode.loss

    def add_timestep_data(self, debug, queue_in_rudder, *args):
        # print("in ts data 1")
        complete_episodes = self.replay_buffer.add_timestep_data(*args)
        # print("in ts data 2")
        # if self.replay_buffer.buffer_full():

        self.new_add_to_replay_buffer(complete_episodes, debug)
        if debug:
            print("in ts data 4")
            if queue_in_rudder.empty():
                print("feed queue inside")
                queue_in_rudder.put(self.replay_buffer)
        # print(queue.empty())
        return

        # self.do_the_rest(complete_episodes)

    def add_timestep_data_MOCK(self, *args):
        complete_episodes = self.replay_buffer.add_timestep_data(*args)
        self.consider_adding_complete_episodes_to_buffer(complete_episodes)
