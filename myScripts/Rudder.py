import torch
from myScripts.MyNet import Net
from myScripts.ReplayBuffer import ReplayBuffer, ProcessData
import numpy as np
#test test


class Rudder:
    def __init__(self, mem_dim, nr_procs, obs_space, instr_dim, ac_embed_dim, image_dim, action_space, device):
        self.replay_buffer = ReplayBuffer(nr_procs,ac_embed_dim,device)
        self.net = Net(image_dim, obs_space, instr_dim, ac_embed_dim, action_space).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=1e-5)
        self.device = device
        self.first_training_done = False
        self.mu=20
        self.quality_threshold=0.8
        self.last_hidden=[None] * nr_procs
        # For the first timestep we will take (0-predictions[:, :1]) as redistributed reward
        self.last_predicted_reward = [None] * nr_procs


    def calc_quality(self,diff):
        # diff is g - gT_hat -->  see rudder paper A267
        quality=1-(np.abs(diff.item())/self.mu)*1/(1-self.quality_threshold)
        return quality


    def lossfunction(self, predictions, returns):
        diff=predictions[:, -1] - returns
        # Main task: predicting return at last timestep
        quality=self.calc_quality(diff)
        main_loss = torch.mean(diff) ** 2
        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        aux_loss = torch.mean(predictions[:, :] - returns[..., None]) ** 2
        # Combine losses
        loss = main_loss + aux_loss * 0.5
        return loss,quality

    def predict_reward(self,embeddings, actions, rewards, dones, instructions, images):
        predictions=[]
        for proc_id,done in enumerate(dones):
            data= {"embeddings":embeddings[proc_id],"actions":actions[proc_id],
               "instructions":instructions[proc_id],"images":images[proc_id]}
            with torch.no_grad():
                hidden=self.last_hidden[proc_id]
                pred, hidden = self.net(data, hidden)
                if self.last_predicted_reward[proc_id]==None:
                    #first timestep
                    pred_reward=0-pred
                else:
                    pred_reward=pred-self.last_predicted_reward[proc_id]
                if done:
                    self.last_hidden[proc_id]=None
                    self.last_predicted_reward[proc_id] = None
                else:
                    self.last_predicted_reward[proc_id]=pred
                    self.last_hidden[proc_id]=hidden
                predictions.append(pred_reward)
        return torch.stack(predictions).squeeze()

    def predict_every_timestep(self, episode: ProcessData):
        hidden = None
        predictions = []
        for i, done in enumerate(episode.dones):
            ts_data = episode.get_timestep_data(i)
            pred, hidden = self.net(ts_data, hidden)
            predictions.append(pred)
            if done:
                break
        assert done == True
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
        predictions = self.predict_every_timestep(episode)
        returns = torch.sum(episode.rewards,dim=-1)
        # returns = np.sum(episode.rewards)
        loss,quality = self.lossfunction(predictions, returns)

        return loss, returns,quality

    def inference_and_set_metrics(self, episode: ProcessData):
        with torch.no_grad():
            # in train and set metrics rewards are a tensor
            episode.rewards=torch.tensor(episode.rewards).to(self.device)
            loss, returns,quality = self.feed_network(episode)
            loss = loss.detach().item()
            returnn = returns.detach().item()
            episode.loss=loss
            episode.returnn=returnn

            return quality


    def train_and_set_metrics_MOCK(self,episode):
        episode.loss=np.random.uniform(0,1)
        r=np.random.uniform(-20, 20)
        if r<0:
            r=0
        episode.returnn = r
        quality=0.1
        return quality

    def train_and_set_metrics(self, episode):
        # loss, returnn, quality = self.train_one_episode(episode)
        loss, returns, quality = self.feed_network(episode)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss = loss.detach().item()
        returnn = returns.detach().item()
        # print("Loss",loss)
        episode.loss = loss
        episode.returnn = returnn
        # print("loss", loss)
        return quality > 0

    def train_full_buffer(self):
        print("train_full_buffer")
        qualities=set()
        for epoch in range(5):
            episodes = self.replay_buffer.sample_episodes()
            for episode in episodes:
                quality=self.train_and_set_metrics(episode)
                qualities.add(quality)
        print("last sample loss",episode.loss)
        if False in qualities:
            self.train_full_buffer()

    def remove_uninteresting_return_episodes(self,complete_episodes):
        return [e for e in complete_episodes if e.rewards[-1] not in set(self.replay_buffer.get_returns())]


    def new_replace_episode_data(self,idx,episode:ProcessData):
        # offset=10
        #the first 10 will never be overwritten for debug causes only!!!
        # idx = np.random.randint(offset,self.replay_buffer.max_size)
        stacked=torch.stack(episode.embeddings)
        self.replay_buffer.embeddings[idx][:len(stacked)]=stacked
        stacked = torch.stack(episode.images)
        self.replay_buffer.images[idx][:len(stacked)] = stacked
        stacked = torch.stack(episode.instructions)
        self.replay_buffer.instructions[idx][:len(stacked)] = stacked
        # stacked = torch.stack(episode.rewards)
        stacked = episode.rewards
        # self.replay_buffer.fast_returns[idx] = torch.sum(stacked)
        self.replay_buffer.rewards[idx][:len(stacked)] = stacked
        stacked = np.array(episode.dones)
        self.replay_buffer.dones[idx][:len(stacked)] = stacked
        stacked = torch.stack(episode.actions)
        self.replay_buffer.actions[idx][:len(stacked)] = stacked

        self.replay_buffer.fast_losses[idx]=episode.loss
        self.replay_buffer.fast_returns[idx]=episode.returnn

    def new_get_replacement_index(self,ce):
        combined_ranks = self.replay_buffer.get_ranks(ce)
        new_episode_rank = combined_ranks[-1]
        # we don't want to get the new sample as potential minimum so remove it
        combined_ranks = combined_ranks[:-1]
        lowest_rank, low_index = self.replay_buffer.get_lowest_ranking_and_idx(combined_ranks)
        if lowest_rank < new_episode_rank:
            return low_index
        return -1

    def new_add_to_replay_buffer(self,complete_episodes):
        replaced=False
        for ce in complete_episodes:
            self.inference_and_set_metrics(ce)
            if self.replay_buffer.buffer_full():
                idx=self.new_get_replacement_index(ce)
                if idx!=-1:
                    self.new_replace_episode_data(idx,ce)
                    replaced=True

            else:
                idx=self.replay_buffer.added_episodes
                self.new_replace_episode_data(idx,ce)
                self.replay_buffer.added_episodes+=1
        if replaced and self.replay_buffer.buffer_full():
            self.train_full_buffer()



    def add_timestep_data(self, *args):
        complete_episodes = self.replay_buffer.add_timestep_data(*args)

        if self.replay_buffer.buffer_full():
            complete_episodes=self.remove_uninteresting_return_episodes(complete_episodes)
        self.new_add_to_replay_buffer(complete_episodes)
        # self.do_the_rest(complete_episodes)


    def add_timestep_data_MOCK(self, *args):
        complete_episodes = self.replay_buffer.add_timestep_data(*args)
        self.consider_adding_complete_episodes_to_buffer(complete_episodes)


