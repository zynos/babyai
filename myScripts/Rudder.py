import torch
from myScripts.MyNet import Net
from myScripts.ReplayBuffer import ReplayBuffer, ProcessData
import numpy as np
#test test
class Rudder:
    def __init__(self, mem_dim, nr_procs, obs_space, instr_dim, ac_embed_dim, image_dim, action_space, device):
        self.replay_buffer = ReplayBuffer(nr_procs)
        self.net = Net(image_dim, obs_space, instr_dim, ac_embed_dim, action_space).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=1e-4)
        self.device = device
        self.first_training_done = False
        self.mu=20
        self.quality_threshold=0.8


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

    def predict_every_timestep(self, episode: ProcessData):
        hidden = None
        predictions = []
        for i, done in enumerate(episode.dones):
            ts_data = episode.get_timestep_data(i)
            pred, hidden = self.net(ts_data, hidden)
            predictions.append(pred)
        assert done == True
        predictions = torch.cat(predictions, dim=1)
        return predictions

    def feed_network(self, episode: ProcessData):
        predictions = self.predict_every_timestep(episode)
        returns = torch.sum(torch.tensor(episode.rewards, device=self.device), dim=-1)
        loss,quality = self.lossfunction(predictions, returns)
        return loss, returns,quality

    def train_one_episode(self, episode: ProcessData):
        loss, returns,quality = self.feed_network(episode)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss = loss.detach().item()
        returnn = returns.detach().item()

        return loss, returnn,quality



    def add_to_buffer_or_discard(self, new_episode):
        # get loss and return of this episode
        with torch.no_grad():
            loss, returnn,quality = self.feed_network(new_episode)
            new_episode.loss = loss.detach().item()
            new_episode.returnn = returnn.detach().item()
        self.replay_buffer.try_to_replace_old_episode(new_episode)






    def fill_empy_buffer(self, complete_episode):
        self.replay_buffer.replay_buffer[self.replay_buffer.added_episodes] = complete_episode
        self.train_and_set_metrics(complete_episode)
        self.replay_buffer.added_episodes += 1
        print("added ", self.replay_buffer.added_episodes)


    def consider_adding_complete_episodes_to_buffer(self, complete_episodes):
        for ce in complete_episodes:
            if not self.replay_buffer.buffer_full():
                self.fill_empy_buffer(ce)
            else:
                if self.replay_buffer.buffer_full():
                    if not self.first_training_done:
                        self.train_full_buffer()
                        self.first_training_done = True
                self.add_to_buffer_or_discard(ce)

        # del complete_episodes

    def train_and_set_metrics(self, episode):
        loss, returnn, quality = self.train_one_episode(episode)
        episode.loss = loss
        episode.returnn = returnn
        print("loss", loss)
        return quality > 0

    def train_full_buffer(self):
        qualities=set()
        for epoch in range(5):
            episodes = self.replay_buffer.sample_episodes()
            for episode in episodes:
                quality=self.train_and_set_metrics(episode)
                qualities.add(quality)

        if False in qualities:
            self.train_full_buffer()

    def add_timestep_data(self, *args):
        complete_episodes = self.replay_buffer.add_timestep_data(*args)
        self.consider_adding_complete_episodes_to_buffer(complete_episodes)

