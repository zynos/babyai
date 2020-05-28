import torch
from torch.nn import MSELoss
# from apex import amp
from myScripts.MyNet import Net
from myScripts.ReplayBuffer import ReplayBuffer, ProcessData
import numpy as np
# test test
import multiprocessing as mp
import logging

from torch.nn.utils import clip_grad_value_


class Rudder:

    def __init__(self):
        # for testing supervised
        self.aux_loss_multiplier = 0.1
        self.train_timesteps = False

    # def __init__(self, mem_dim, nr_procs, obs_space, instr_dim, ac_embed_dim, image_dim, action_space, device):
    #     self.nr_procs = nr_procs
    #     self.use_transformer = False
    #     self.aux_loss_multiplier = 0.1
    #     self.clip_value = 0.5
    #     self.frames_per_proc = 40
    #     self.replay_buffer = ReplayBuffer(nr_procs, ac_embed_dim, device, self.frames_per_proc)
    #     self.train_timesteps = False
    #     self.net = Net(image_dim, obs_space, instr_dim, ac_embed_dim, action_space, device).to(device)
    #     self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-6, weight_decay=1e-6)
    #     # self.optimizer = torch.optim.Adam(self.net.parameters())
    #     self.device = device
    #     self.first_training_done = False
    #     self.mu = 1
    #     self.quality_threshold = 0.8
    #     self.last_hidden = [None] * nr_procs
    #     # For the first timestep we will take (0-predictions[:, :1]) as redistributed reward
    #     self.last_predicted_reward = [None] * nr_procs
    #     self.parallel_train_done = False
    #     self.grad_norms=[]
    #     self.grad_norm = 0
    #     self.current_quality = 0
    #     # mpl = mp.log_to_stderr()
    #     # mpl.setLevel(logging.INFO)
    #     self.updates = 0
    #     self.mse_loss = MSELoss()
    #     ### APEX
    #     # self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level="O1")

    def calc_quality(self, diff):
        # diff is g - gT_hat -->  see rudder paper A267
        quality = 1 - (np.abs(diff.item()) / self.mu) * 1 / (1 - self.quality_threshold)
        return quality

    def paper_loss3(self, predictions, returns, pred_plus_ten_ts):

        diff = predictions[:, -1] - returns
        # Main task: predicting return at last timestep
        quality = self.calc_quality(diff)
        main_loss = diff ** 2
        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        continuous_loss = torch.mean((predictions[:, :] - returns[..., None]) ** 2)
        # continuous_loss = self.mse_loss(predictions[:, :], returns[..., None])

        # loss Le of the prediction of the output at t+10 at each time step t
        le10_loss = 0.0
        # if episode is smaller than 10 the follwoing would produce a NAN
        if predictions.shape[1] > 10:
            pred_chunk = predictions[:, 10:]
            le10_loss = torch.mean((pred_chunk - pred_plus_ten_ts[:, :-10]) ** 2)

        # le10_loss = self.mse_loss(pred_chunk, pred_plus_ten_ts[:, :-10])

        # Combine losses
        aux_loss = continuous_loss + le10_loss
        loss = main_loss + self.aux_loss_multiplier * aux_loss
        return loss, quality, (main_loss.detach().clone().item(), aux_loss.detach().clone().item())

        # def lossfunction(self, predictions, returns):

    #     diff = predictions[:, -1] - returns
    #     # Main task: predicting return at last timestep
    #     quality = self.calc_quality(diff)
    #     main_loss = torch.mean(diff) ** 2
    #     # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
    #     aux_loss = torch.mean(predictions[:, :] - returns[..., None]) ** 2
    #     # Combine losses
    #     loss = main_loss + aux_loss * 0.5
    #     return loss, quality

    # def get_lstm_prediction(self, proc_id, data, done, batch):
    #     if not batch:
    #         data = data.get_timestep_data(len(data.dones) - 1)
    #     hidden = self.last_hidden[proc_id]
    #     pred, hidden = self.net(data, hidden, not self.train_timesteps)
    #     # 1 timestep samples are 2d
    #     # if not pred.ndim == 3:
    #     #     pred = pred.unsqueeze(0)
    #     # if not self.train_timesteps:
    #     try:
    #         pred = pred[-1][-1][-1]
    #     except:
    #         pred = pred[-1][-1]
    #     # print(pred.shape)
    #
    #     if not self.last_predicted_reward[proc_id]:
    #         # first timestep
    #         pred_reward = 0 - pred
    #     else:
    #         pred_reward = pred - self.last_predicted_reward[proc_id]
    #     if done:
    #         self.last_hidden[proc_id] = None
    #         self.last_predicted_reward[proc_id] = None
    #     else:
    #         self.last_predicted_reward[proc_id] = pred
    #         self.last_hidden[proc_id] = hidden
    #     return pred_reward

    def get_transformer_prediction(self, proc_id, data, done):
        episode = self.replay_buffer.proc_data_buffer[proc_id]
        pred, _ = self.net(episode, None, True, True)
        # print(pred)
        pred = pred[-1][-1][-1]
        # try:
        #     pred = pred[-1][-1][-1]
        # except:
        #     print(pred.shape)
        #     pred = pred[-1][-1]
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

    # def predict_episodes_per_pID(self, episodes, proc_id):
    #     current_ts = 0
    #     end_pred = []
    #     for e in episodes:
    #         current_ts = e.end_timestep
    #         pred, _ = self.net(e, None, not self.train_timesteps)
    #         pred = pred[-1].squeeze(1)[-current_ts:]
    #         end_pred.append(pred)
    #     if current_ts - self.frames_per_proc - 1 != 0:
    #         # one unfinished episode in proc buffer
    #         episode = self.replay_buffer.proc_data_buffer[proc_id]
    #         pred, _ = self.net(episode, None, not self.train_timesteps)
    #         pred = pred[-1].squeeze(1)
    #         end_pred.append(pred)
    #     return torch.cat(end_pred, dim=-1)

    # def new_predict_reward(self):
    #     all_predictions=[]
    #     for proc_id in range(self.nr_procs):
    #         episodes = self.replay_buffer.process_queue[proc_id]
    #         if episodes:
    #             full_pred = self.predict_episodes_per_pID(episodes, proc_id)
    #             self.replay_buffer.process_queue[proc_id] = []
    #             all_predictions.append(full_pred)
    #         else:
    #             episode = self.replay_buffer.proc_data_buffer[proc_id]
    #             pred=self.predict_full_episode(episode)
    #             pred=pred[-1].squeeze(1)[-self.frames_per_proc:]
    #             all_predictions.append(pred)
    #     return all_predictions
    #
    def do_part_prediction(self, proc_id, timestep):
        with torch.no_grad():
            episode = self.replay_buffer.proc_data_buffer[proc_id]
            pred, _ = self.predict_full_episode(episode)
            episode_len = len(episode.dones)
            to_add = min(episode_len, timestep + 1)
            previous_pred = None
            # might be out of range because frames per proc is smaller than episode len
            try:
                previous_pred = pred.squeeze()[-(to_add + 1)]
            except:
                pass
            # single dimension
            try:
                pred = pred.squeeze()[-to_add:]
            except IndexError:
                pred = pred.squeeze().unsqueeze(0)
            # we need to get the differences to previous timestep predictions
            previous_timesteps = torch.zeros(pred.shape, device=self.device)
            # leave the first value at zero to not subtract anything from first prediction value
            previous_timesteps[1:] = pred[:-1]

            if previous_pred is not None:
                pred[0] = pred[0] - previous_pred
            else:
                # pred[0] = 0 - pred[0]
                pred[0] = pred[0]
            pred -= previous_timesteps
            # enforce same return
            returns = torch.sum(torch.tensor(episode.rewards[-to_add:], device=self.device))
            predicted_returns = pred.sum()
            prediction_error = returns - predicted_returns
            pred += prediction_error / pred.shape[0]

            self.replay_buffer.current_predictions[proc_id].append(pred.detach().clone())

    def new_predict_reward(self, dones, timestep):
        self.net.eval()
        for i, done in enumerate(dones):
            if done and timestep != self.frames_per_proc - 1:
                self.do_part_prediction(i, timestep)
            if timestep == self.frames_per_proc - 1:
                self.do_part_prediction(i, timestep)

        if timestep == self.frames_per_proc - 1:
            # output shape = (self.num_frames_per_proc, self.num_procs)
            all_frames = []
            for proc_id, _ in enumerate(dones):
                all_frames.append(torch.cat(self.replay_buffer.current_predictions[proc_id], dim=-1))
                self.replay_buffer.current_predictions[proc_id] = []
            # the first time we have incomplete episodes as an input
            # which started before training was done so stack will fail if not all records are complete
            try:
                ret = torch.stack(all_frames)
            except:
                ret = None
            return ret

    def predict_reward(self, embeddings, actions, rewards, dones, instructions, images):
        predictions = []
        batch = not self.train_timesteps
        for proc_id, done in enumerate(dones):
            data = self.replay_buffer.proc_data_buffer[proc_id]
            with torch.no_grad():
                if self.use_transformer:
                    pred_reward = self.get_transformer_prediction(proc_id, data, done)
                else:
                    pred_reward = self.get_lstm_prediction(proc_id, data, done, batch)
                predictions.append(pred_reward)
        return torch.stack(predictions).squeeze()

    def predict_full_episode(self, episode: ProcessData):
        predictions, hidden, pred_plus_ten_ts = self.net(episode, None, True, self.use_transformer)
        return predictions, pred_plus_ten_ts

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
            predictions, pred_plus_ten_ts = self.predict_full_episode(episode)
        returns = torch.sum(episode.rewards, dim=-1).to(self.device)
        # returns = np.sum(episode.rewards)
        loss, quality, raw_loss = self.paper_loss3(predictions, returns, pred_plus_ten_ts)
        # loss, quality = self.lossfunction(predictions, returns)

        return loss, returns, quality, predictions.detach().clone(), raw_loss

    def inference_and_set_metrics(self, episode: ProcessData):
        self.net.eval()
        with torch.no_grad():
            # in train and set metrics rewards are a tensor

            # episode.rewards=torch.from_numpy(np.array(episode.rewards)).to(self.device)
            try:
                # print("episode.rewards",episode.rewards)
                episode.rewards = torch.stack(episode.rewards)
                # episode.values = torch.stack(episode.values)
            except:
                pass
            assert isinstance(episode.rewards, torch.Tensor)

            loss, returns, quality, _, raw_loss = self.feed_network(episode)
            loss = loss.detach().item()
            returnn = returns.detach().item()
            episode.loss = loss
            episode.returnn = returnn

            return quality,raw_loss

    def train_and_set_metrics_MOCK(self, episode):
        episode.loss = np.random.uniform(0, 1)
        r = np.random.uniform(-20, 20)
        if r < 0:
            r = 0
        episode.returnn = r
        quality = 0.1
        return quality

    def train_and_set_metrics(self, episode):
        self.net.train()
        # loss, returnn, quality = self.train_one_episode(episode)
        loss, returns, quality, predictions,raw_loss = self.feed_network(episode)

        ### APEX
        # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #     scaled_loss.backward()
        ###
        loss.backward()
        grad_norm = sum(
            p.grad.data.norm(2) ** 2 for p in self.net.parameters() if p.grad is not None) ** 0.5
        clip_grad_value_(self.net.parameters(), self.clip_value)

        self.optimizer.step()
        self.optimizer.zero_grad()

        loss = loss.detach().item()
        returnn = returns.detach().item()
        # print("Loss",loss)
        episode.loss = loss
        episode.returnn = returnn

        # print("loss", loss)
        return quality, predictions, episode, grad_norm,raw_loss

    def train_full_buffer(self):
        # print("train_full_buffer")
        self.net.train()
        losses = []
        full_predictions = []
        last_timestep_prediction = []
        last_rewards = []
        bad_quality = True
        while bad_quality:
            qualities_bools = set()
            qualities = []
            for epoch in range(5):
                episodes, episodes_ids = self.replay_buffer.sample_episodes()
                # episodes = self.replay_buffer.sample_episodes()
                for i, episode in enumerate(episodes):
                    quality, predictions, episode, grad_norm, raw_loss = self.train_and_set_metrics(episode)
                    self.grad_norms.append(grad_norm.item())
                    self.replay_buffer.fast_losses[episodes_ids[i]] = episode.loss
                    full_predictions.append(predictions.unsqueeze(0))
                    last_timestep_prediction.append(predictions[0][-1].item())
                    losses.append(episode.loss)
                    last_rewards.append(episode.rewards[-1].item())
                    # assert episode.returnn==self.replay_buffer.fast_returns[episodes_ids[i]]
                    qualities_bools.add(quality > 0)
                    qualities.append(np.clip(quality, 0.0, 0.01))
            self.current_quality = np.mean(qualities)
            # self.current_quality = 0.25
            print("sample {} return {:.2f} loss {:.4f} predX {:.2f}  rewX {:.2f} pred[-1] {:.2f} rew[-1]  {:.2f} ".
                  format(episodes_ids[-1], episode.returnn, episode.loss,
                         predictions.mean().item(), episode.rewards.mean().item(),
                         predictions[-1][-1][-1].item(), episode.rewards[-1].item()))
            if False not in qualities_bools:
                bad_quality = False
        self.grad_norm = np.mean(self.grad_norms)
        self.grad_norms = []
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

    # def recalculate_all_losses(self):
    #     for i in range(self.replay_buffer.max_size):
    #         episode = self.replay_buffer.get_episode_from_tensors(i)
    #         self.inference_and_set_metrics(episode)
    #         self.replay_buffer.fast_losses[i] = episode.loss

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
