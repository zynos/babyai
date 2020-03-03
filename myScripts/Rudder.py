import torch
from myScripts.MyNet import Net
from myScripts.ReplayBuffer import ReplayBuffer, ProcessData
import numpy as np

class Rudder:
    def __init__(self, mem_dim, nr_procs, obs_space, instr_dim, ac_embed_dim, image_dim, action_space, device):
        self.replay_buffer = ReplayBuffer(nr_procs)
        self.net = Net(image_dim, obs_space, instr_dim, ac_embed_dim, action_space).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=1e-4)
        self.device = device
        self.first_training_done = False

    def lossfunction(self, predictions, returns):
        # Main task: predicting return at last timestep
        main_loss = torch.mean(predictions[:, -1] - returns) ** 2
        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        if returns.item() > 0:
            print('d')
        aux_loss = torch.mean(predictions[:, :] - returns[..., None]) ** 2
        # Combine losses
        loss = main_loss + aux_loss * 0.5
        return loss

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
        loss = self.lossfunction(predictions, returns)
        return loss, returns

    def train_one_episode(self, episode: ProcessData):
        loss, returns = self.feed_network(episode)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss = loss.detach().item()
        returnn = returns.item()
        return loss, returnn



    def add_to_buffer_or_discard(self, new_episode):
        # get loss and return of this episode
        with torch.no_grad():
            loss, returnn = self.feed_network(new_episode)
            new_episode.loss = loss
            new_episode.returnn = returnn
        self.replay_buffer.try_to_replace_old_episode(new_episode)









    def consider_adding_complete_episodes_to_buffer(self, complete_episodes):
        for ce in complete_episodes:
            if not self.replay_buffer.buffer_full():
                self.replay_buffer.replay_buffer[self.replay_buffer.added_episodes] = ce
                self.replay_buffer.added_episodes += 1
                # print("added ",self.added_episodes)
            else:
                self.add_to_buffer_or_discard(ce)

        del complete_episodes

    def train_full_buffer(self):
        episodes = self.replay_buffer.replay_buffer
        for episode in episodes:
            print("train one ep")
            loss, returnn = self.train_one_episode(episode)
            episode.loss = loss
            episode.returnn = returnn
            print('info')

    def add_timestep_data(self, *args):
        complete_episodes = self.replay_buffer.add_timestep_data(*args)

        if self.replay_buffer.buffer_full() and not self.first_training_done:
            self.train_full_buffer()
