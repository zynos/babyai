import torch
from scipy.stats import rankdata
import numpy as np
from torch.distributions import Categorical
from copy import deepcopy


class ProcessData():
    def __init__(self):
        # input proc_id * feature
        self.actions = []
        self.rewards = []
        self.dones = []
        self.embeddings = []
        self.images = []
        self.instructions = []

    def add_single_timestep(self, embedding, action, reward, done, instruction, image):
        self.embeddings.append(embedding)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.images.append(image)
        self.instructions.append(instruction)

    def get_timestep_data(self, timestep):
        dummy = ProcessData()
        dummy.rewards.append(self.rewards[timestep])
        dummy.actions.append(self.actions[timestep])
        dummy.dones.append(self.dones[timestep])
        dummy.embeddings.append(self.embeddings[timestep])
        dummy.images.append(self.images[timestep])
        dummy.instructions.append(self.instructions[timestep])
        return dummy

        # return {key: value[timestep] for key, value in self.__dict__.items()
        #         if not key.startswith('__') and not callable(key)}
        # return {key: value[timestep] for key, value in self.__dict__.items()
        #         if not key.startswith('__') and not callable(key) and isinstance(value, list)}

    def self_destroy(self):
        del self.actions
        torch.cuda.empty_cache()
        del self.rewards
        torch.cuda.empty_cache()
        del self.dones
        torch.cuda.empty_cache()
        del self.embeddings
        torch.cuda.empty_cache()
        del self.images
        torch.cuda.empty_cache()
        del self.instructions
        torch.cuda.empty_cache()
        try:
            del self.loss
            torch.cuda.empty_cache()
            del self.returnn
            torch.cuda.empty_cache()
        except:
            pass


class ReplayBuffer:
    def __init__(self, nr_procs, embed_dim, device):
        self.max_steps = 128
        self.embed_dim = embed_dim
        self.device = device
        self.sample_amount = 8
        self.nr_procs = nr_procs
        self.added_episodes = 0
        self.proc_data_buffer = [ProcessData() for _ in range(self.nr_procs)]
        # self.proc_data_buffer_for_prediction = [ProcessData() for _ in range(self.nr_procs)]
        self.max_size = 128
        # self.replay_buffer = [None] * self.max_size
        self.big_counter = 0

        self.embeddings = torch.zeros((self.max_size, self.max_steps, embed_dim)).to(device)
        self.images = torch.zeros((self.max_size, self.max_steps, 7, 7, 3)).to(device)
        self.instructions = torch.zeros((self.max_size, self.max_steps, 9), dtype=torch.int64).to(device)
        self.rewards = torch.zeros((self.max_size, self.max_steps)).to(device)
        self.actions = torch.zeros((self.max_size, self.max_steps), dtype=torch.int64).to(device)
        self.dones = np.zeros((self.max_size, self.max_steps), dtype=bool)

        self.fast_returns = np.zeros(self.max_size)
        self.fast_losses = np.zeros(self.max_size)
        self.process_queue = [[] for _ in range(nr_procs)]
        self.current_predictions = [[] for _ in range(nr_procs)]

    def get_cloned_copy(self):
        instance = ReplayBuffer(self.nr_procs, self.embed_dim, self.device)
        instance.embeddings = self.embeddings.detach().clone()
        instance.instructions = self.instructions.detach().clone()
        instance.rewards = self.rewards.detach().clone()
        instance.actions = self.actions.detach().clone()
        instance.dones = self.dones

        instance.fast_returns = self.fast_returns
        instance.fast_losses = self.fast_losses
        return instance

    def buffer_full(self):
        return self.added_episodes == self.max_size

    def detach_and_clone(self, *args):
        res = []
        for a in args:
            try:
                tensor = a.detach().clone()
                res.append(tensor)
            except:
                res.append(a)
        return res

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
        return list(self.fast_returns)
        # return [r.returnn for r in self.replay_buffer]

    def get_losses(self):
        return list(self.fast_losses)
        # return [[l.loss for l in self.replay_buffer]]

    def get_losses_and_returns(self):
        losses = self.get_losses()
        returns = self.get_returns()
        assert len(losses) == len(returns) == self.max_size
        return losses, returns

    def get_ranks(self, new_episode: ProcessData):
        losses, returns = self.get_losses_and_returns()
        losses.append(new_episode.loss)
        returns.append(new_episode.returnn)
        combined_ranks = self.get_combined_ranks(losses, returns)

        return combined_ranks

    def get_lowest_ranking_and_idx(self, combined_ranks):
        return np.min(combined_ranks), np.argmin(combined_ranks)

    def add_data_to_process_buffer(self, data_list):
        # data is embeddings,actions,rewards,dones,instructions,images
        complete_episodes = []
        procs_to_init = []
        for proc_id in range(self.nr_procs):
            el = [data[proc_id] for data in data_list]
            self.proc_data_buffer[proc_id].add_single_timestep(*el)
            if self.proc_data_buffer[proc_id].dones[-1] == True:
                procs_to_init.append(proc_id)
                complete_episodes.append(self.proc_data_buffer[proc_id])
                # self.proc_data_buffer[proc_id]=None
                # self.proc_data_buffer[proc_id] = ProcessData()
        # print("complete episodes",len(complete_episodes))
        # self.proc_data_buffer_for_prediction=deepcopy(self.proc_data_buffer)
        self.procs_to_init = procs_to_init
        # self.init_process_data(procs_to_init)
        # del data_list
        return complete_episodes

    def init_process_data(self, procs_to_init, timestep):
        for p_id in procs_to_init:
            self.proc_data_buffer[p_id].end_timestep = timestep
            assert self.proc_data_buffer[p_id].dones[-1] == True
            self.process_queue[p_id].append(self.proc_data_buffer[p_id])
            # self.proc_data_buffer[p_id].self_destroy()
            self.proc_data_buffer[p_id] = None
            self.proc_data_buffer[p_id] = ProcessData()
            # print("init",p_id)

    def add_timestep_data(self, embeddings, actions, rewards, dones, instructions, images):
        result = self.detach_and_clone(embeddings, actions, rewards, dones, instructions, images)
        complete_episodes = self.add_data_to_process_buffer(result)
        # del result
        return complete_episodes

    def new_get_replacement_index(self, ce):
        combined_ranks = self.get_ranks(ce)
        new_episode_rank = combined_ranks[-1]
        # we don't want to get the new sample as potential minimum so remove it
        combined_ranks = combined_ranks[:-1]
        lowest_rank, low_index = self.get_lowest_ranking_and_idx(combined_ranks)
        if lowest_rank < new_episode_rank:
            return low_index
        return -1

    def new_replace_episode_data(self, idx, episode: ProcessData):
        # offset=10
        # the first 10 will never be overwritten for debug causes only!!!
        # idx = np.random.randint(offset,self.replay_buffer.max_size)
        stacked = torch.stack(episode.embeddings)
        self.embeddings[idx][:len(stacked)] = stacked
        stacked = torch.stack(episode.images)
        self.images[idx][:len(stacked)] = stacked
        stacked = torch.stack(episode.instructions)
        self.instructions[idx][:len(stacked)] = stacked
        # stacked = torch.stack(episode.rewards)
        stacked = episode.rewards
        # self.replay_buffer.fast_returns[idx] = torch.sum(stacked)
        self.rewards[idx][:len(stacked)] = stacked
        stacked = np.array(episode.dones)
        self.dones[idx][:len(stacked)] = stacked
        stacked = torch.stack(episode.actions)
        self.actions[idx][:len(stacked)] = stacked

        self.fast_losses[idx] = episode.loss
        self.fast_returns[idx] = episode.returnn

    def get_episode_from_tensors(self, id):
        episode = ProcessData()
        max_idx = np.where(self.dones[id] == True)[0][0]
        # we want to include last element
        max_idx = max_idx + 1
        episode.dones = self.dones[id][:max_idx]
        # assert len(np.where(episode.dones==True))==1
        episode.rewards = self.rewards[id][:max_idx]
        episode.actions = self.actions[id][:max_idx]
        episode.embeddings = self.embeddings[id][:max_idx]
        episode.images = self.images[id][:max_idx]
        episode.instructions = self.instructions[id][:max_idx]
        # episode.returnn=self.fast_returns[id]
        return episode

    def sample_episodes(self):
        losses, retruns = self.get_losses_and_returns()
        combined_ranks = self.get_combined_ranks(losses, retruns)
        # assert len(combined_ranks)<=self.max_size
        probs = torch.nn.functional.softmax(torch.tensor(combined_ranks))
        if len(probs) > self.max_size:
            print("fail3")
        m = Categorical(probs)
        episodes = []
        ids = []
        for _ in range(self.sample_amount):
            episode_id = m.sample()
            if episode_id >= self.max_size:
                print('fail')
            episode = self.get_episode_from_tensors(episode_id)
            # assert episode.returnn==self.fast_returns[episode_id]
            episodes.append(episode)
            ids.append(episode_id)
        return episodes, ids
