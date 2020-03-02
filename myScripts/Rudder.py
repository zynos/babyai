import torch
from myScripts.MyNet import Net
from myScripts.ReplayBuffer import ReplayBuffer, ProcessData


class Rudder():
    def __init__(self,mem_dim,nr_procs,obs_space,instr_dim,ac_embed_dim,image_dim,action_space,device):
        self.replay_buffer=ReplayBuffer(nr_procs)
        self.net=Net(image_dim,obs_space,instr_dim,ac_embed_dim,action_space).to(device)


    def train_one_episode(self,episode:ProcessData):
        for i in range(len(episode.dones)):
            ts_data=episode.get_timestep_data(i)
            # print("ts",ts_data)
            self.net(ts_data)

    def train_full_buffer(self):
        episodes=self.replay_buffer.complete_episodes
        for episode in episodes:
            print("train one ep")
            self.train_one_episode(episode)


    def add_timestep_data(self,*args):
        self.replay_buffer.add_timestep_data(*args)

        if self.replay_buffer.buffer_full():
            self.train_full_buffer()
