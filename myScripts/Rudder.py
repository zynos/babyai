import torch
from myScripts.ReplayBuffer import ReplayBuffer
class Rudder():
    def __init__(self,mem_dim,nr_procs):
        self.replay_buffer=ReplayBuffer(nr_procs)

    def add_timestep_data(self,*args):
        self.replay_buffer.add_timestep_data(*args)