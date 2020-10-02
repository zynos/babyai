import numpy as np
import torch
from scipy.stats import rankdata
from torch.distributions import Categorical


class RudderReplayBuffer:
    def __init__(self, nr_procs, frames_per_proc, device):
        self.sample_amount = 8
        self.added_episodes = 0
        self.max_size = 128
        self.nr_procs = nr_procs
        self.frames_per_proc = frames_per_proc
        self.masks = torch.zeros(self.max_size, frames_per_proc, device=device)
        self.rewards = torch.zeros(self.max_size, frames_per_proc, device=device)
        self.values = torch.zeros(self.max_size, frames_per_proc, device=device)
        self.actions = torch.zeros(self.max_size, frames_per_proc, device=device)
        self.obs = [None] * self.max_size
        self.losses = np.zeros(self.max_size)
        self.returns = np.zeros(self.max_size)
        self.dones = torch.zeros(self.max_size, frames_per_proc, device=device)

    def buffer_full(self):
        return self.added_episodes == self.max_size

    def get_ranks(self, new_loss, new_return):
        losses, returns = self.get_losses_and_returns()
        losses.append(new_loss)
        returns.append(new_return)
        combined_ranks = self.get_combined_ranks(losses, returns)

        return combined_ranks

    def get_lowest_ranking_and_idx(self, combined_ranks):
        return np.min(combined_ranks), np.argmin(combined_ranks)

    def new_get_replacement_index(self, new_loss, new_return):
        combined_ranks = self.get_ranks(new_loss, new_return)
        new_episode_rank = combined_ranks[-1]
        # we don't want to get the new sample as potential minimum so remove it
        combined_ranks = combined_ranks[:-1]
        lowest_rank, low_index = self.get_lowest_ranking_and_idx(combined_ranks)
        if lowest_rank < new_episode_rank:
            return low_index
        return -1

    def fill_single_class_member_array(self, class_member, new_data_array):
        class_member[:, :] = new_data_array.transpose(0, 1)[:class_member.shape[0], :class_member.shape[1]]

    # def store_sequence_chunks(self, masks, rewards, values, actions, obs):
    #     self.fill_single_class_member_array(self.masks, masks)
    #     self.fill_single_class_member_array(self.rewards, rewards)
    #     self.fill_single_class_member_array(self.values, values)
    #     self.fill_single_class_member_array(self.actions, actions)
    #     self.returns = torch.sum(rewards, dim=0)
    #     self.obs = list(map(list, zip(*obs)))

    def add_single_sequence(self, masks, rewards, values, actions, obs, returnn, loss, dones, index):
        self.masks[index] = masks
        self.rewards[index] = rewards
        self.values[index] = values
        self.actions[index] = actions
        self.obs[index] = obs
        self.dones[index] = dones
        self.returns[index] = returnn.item()
        self.losses[index] = loss.item()
        if not self.buffer_full():
            self.added_episodes += 1

    def get_single_sequence(self, nr):
        return self.obs[nr], self.masks[nr], self.rewards[nr].clone(), self.actions[nr], self.values[nr], self.dones[nr]

    def get_combined_ranks(self, losses, returns):
        # we need the deviation of the mean return per episode
        mean_return = np.mean(returns)
        ret_deviations = [np.abs(mean_return - ret) for ret in returns]

        ranked_losses = rankdata(losses)
        ranked_ret_deviations = rankdata(ret_deviations)
        combined_ranks = ranked_ret_deviations + ranked_losses
        return combined_ranks

    def encountered_different_returns(self):
        ret_set = set(self.get_returns())
        return len(ret_set) > 1

    def get_returns(self):
        return list(self.returns)

    def get_losses(self):
        return list(self.losses)

    def get_losses_and_returns(self):
        losses = self.get_losses()
        returns = self.get_returns()
        assert len(losses) == len(returns) == self.max_size
        return losses, returns

    def sample_episodes(self):
        losses, retruns = self.get_losses_and_returns()
        combined_ranks = self.get_combined_ranks(losses, retruns)
        # assert len(combined_ranks)<=self.max_size
        probs = torch.nn.functional.softmax(torch.tensor(combined_ranks), dim=0)
        if len(probs) > self.max_size:
            print("fail3")
        m = Categorical(probs)
        episodes = []
        ids = []
        for _ in range(self.sample_amount):
            episode_id = m.sample()
            if episode_id >= self.max_size:
                print('fail')
            episode = self.get_single_sequence(episode_id)
            # assert episode.returnn==self.fast_returns[episode_id]
            episodes.append(episode)
            ids.append(episode_id)
        return episodes, ids
