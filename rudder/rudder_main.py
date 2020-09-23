import numpy as np
import torch

from myScripts.ReplayBuffer import ReplayBuffer, ProcessData
from rudder.rudder_evaluate import calculate_my_flat_loss, transform_data_for_loss_calculation
from rudder.rudder_imitation_learning import RudderImitation


class NonParsedDummyArgs:
    def __init__(self, instr_dim, memory_dim, image_dim, lr):
        self.model = "rudderRLModel"
        self.no_instr = False
        self.no_mem = False
        self.instr_arch = "gru"
        self.arch = 'expert_filmcnn'
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim
        self.env = "BabyAI-PutNextLocal-v0"
        self.log_interval = 1
        self.lr = lr
        self.optim_eps = 1e-5
        self.recurrence = 20

class Rudder:
    def __init__(self, nr_procs, device, frames_per_proc, instr_dim, memory_dim, image_dim, lr):
        dummy_args = NonParsedDummyArgs(instr_dim, memory_dim, image_dim, lr)
        self.il_learn = RudderImitation(None, True, True, dummy_args)
        # these 2 must be updated when replaybuffer full and then after every new insert
        self.il_learn.mean = 0
        self.il_learn.std_dev = 1
        self.replay_buffer = ReplayBuffer(nr_procs, 128, device, frames_per_proc)
        self.first_training_done = False
        self.current_quality = 0
        self.grad_norms = []
        self.grad_norm = -1



        # collect experiences without inferencing and store losses and experiences
        # train rudder full buffer priority replay style (see train full buffer)
        # redistribute reward before ppo loss calculation

