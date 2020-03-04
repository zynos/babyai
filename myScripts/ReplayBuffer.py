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
        return {key: value[timestep] for key, value in self.__dict__.items()
                if not key.startswith('__') and not callable(key) and isinstance(value, list)}

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
    def __init__(self, nr_procs):
        self.sample_amount = 8
        self.nr_procs = nr_procs
        self.added_episodes = 0
        self.proc_data_buffer = [ProcessData() for _ in range(self.nr_procs)]
        self.max_size = 128
        self.replay_buffer = [None] * self.max_size
        self.big_counter=0

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

    def get_combined_ranks(self,losses,returns):
        # we need the deviation of the mean return per episode
        mean_return = np.mean(returns)
        ret_deviations = [np.abs(mean_return - ret) for ret in returns]

        ranked_losses = rankdata(losses)
        ranked_ret_deviations = rankdata(ret_deviations)
        combined_ranks = ranked_ret_deviations + ranked_losses
        return combined_ranks

    def get_losses_and_returns(self):
        losses = [l.loss for l in self.replay_buffer]
        returns = [r.returnn for r in self.replay_buffer]
        return losses,returns

    def get_ranks(self,new_episode:ProcessData):
        losses, returns = self.get_losses_and_returns()
        losses.append(new_episode.loss)
        returns.append(new_episode.returnn)
        combined_ranks = self.get_combined_ranks(losses,returns)

        return combined_ranks

    def get_lowest_ranking_and_idx(self, combined_ranks):
        return np.min(combined_ranks), np.argmin(combined_ranks)

    def try_to_replace_old_episode(self, episode):
        combined_ranks = self.get_ranks(episode)
        new_episode_rank = combined_ranks[-1]
        # we don't want to get the new sample as potential minimum so remove it
        combined_ranks=combined_ranks[:-1]
        lowest_rank, low_index = self.get_lowest_ranking_and_idx(combined_ranks)
        # if lowest ranked episode is lower than the new episode add it to buffer
        if lowest_rank < new_episode_rank:
            # dummy_episode=ProcessData()
            # dummy_episode.loss=episode.loss
            # dummy_episode.returnn = episode.returnn
            # dummy_episode.embeddings=episode.embeddings
            # episode.self_destroy()
            # del episode
            # print("replace with",lowest_rank,new_episode_rank)
            # self.replay_buffer[low_index].self_destroy()
            self.replay_buffer[low_index] = episode
            self.added_new_episode=True
            # self.big_counter+=1
            # print("big counter",self.big_counter)
            # #this might counter leak
            # if self.big_counter>=200:
            #     self.added_episodes=0
            #     self.replay_buffer = None
            #     self.replay_buffer = [None] * self.max_size
        else:
            self.added_new_episode = True
        #     del episode

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
        self.init_process_data(procs_to_init)
        # del data_list
        return complete_episodes

    def init_process_data(self, procs_to_init):
        for p_id in procs_to_init:
            # self.proc_data_buffer[p_id].self_destroy()
            self.proc_data_buffer[p_id]=None
            self.proc_data_buffer[p_id] = ProcessData()
            # print("init",p_id)

    def add_timestep_data(self, embeddings, actions, rewards, dones, instructions, images):
        result = self.detach_and_clone(embeddings, actions, rewards, dones, instructions, images)
        complete_episodes = self.add_data_to_process_buffer(result)
        # del result
        return complete_episodes

    def sample_episodes(self):
        losses,retruns=self.get_losses_and_returns()
        combined_ranks= self.get_combined_ranks(losses,retruns)
        probs=torch.nn.functional.softmax(torch.tensor(combined_ranks))
        m = Categorical(probs)
        episodes=[]
        for _ in range(self.sample_amount):
            episode_id = m.sample()
            episodes.append(self.replay_buffer[episode_id])
        return episodes

