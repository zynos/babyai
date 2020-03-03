import torch
from myScripts.MyNet import Net
from myScripts.ReplayBuffer import ReplayBuffer, ProcessData


class Rudder:
    def __init__(self, mem_dim, nr_procs, obs_space, instr_dim, ac_embed_dim, image_dim, action_space, device):
        self.replay_buffer = ReplayBuffer(nr_procs)
        self.net = Net(image_dim, obs_space, instr_dim, ac_embed_dim, action_space).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=1e-4)
        self.device = device

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
        return loss,returns

    def train_one_episode(self, episode: ProcessData):
        loss, returns = self.feed_network(episode)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss = loss.detach().item()
        returnn = returns.item()
        return loss, returnn

    def train_full_buffer(self):
        episodes = self.replay_buffer.complete_episodes
        for episode in episodes:
            print("train one ep")
            loss, returnn = self.train_one_episode(episode)
            episode.loss = loss
            episode.returnn = returnn
            print('info')

    def add_timestep_data(self, *args):
        self.replay_buffer.add_timestep_data(*args)

        if self.replay_buffer.buffer_full():
            self.train_full_buffer()
