from myScripts.ReplayBuffer import ReplayBuffer
from rudder.rudder_imitation_learning import RudderImitation


class NonParsedDummyArgs:
    def __init__(self, instr_dim, memory_dim, image_dim):
        self.model = None
        self.no_instr = False
        self.no_mem = False
        self.instr_arch = "gru"
        self.arch = 'expert_filmcnn'
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim
        self.env = "BabyAI-PutNextLocal-v0"
        self.log_interval = 1


class Rudder:
    def __init__(self, nr_procs, device, frames_per_proc, instr_dim, memory_dim, image_dim):
        dummy_args = NonParsedDummyArgs(instr_dim, memory_dim, image_dim)
        self.il_learn = RudderImitation(None, True, True, dummy_args)
        self.replay_buffer = ReplayBuffer(nr_procs, None, device, frames_per_proc)

    def add_timestep_data(self, debug, queue_in_rudder, *args):
        complete_episodes = self.replay_buffer.add_timestep_data(*args)

        self.new_add_to_replay_buffer(complete_episodes, debug)
        return

    def new_add_to_replay_buffer(self, complete_episodes, debug=False):
        for ce in complete_episodes:

            self.inference_and_set_metrics(ce)
            if self.replay_buffer.buffer_full():
                idx = self.replay_buffer.new_get_replacement_index(ce)
            else:
                idx = self.replay_buffer.added_episodes
                self.replay_buffer.added_episodes += 1
            if idx != -1:
                self.replay_buffer.new_replace_episode_data(idx, ce)

    def inference_and_set_metrics(self, complete_episode):
        self.il_learn.run_epoch_recurrence_one_batch([complete_episode], False)

    # def inference_and_set_metrics(self, episode: ProcessData):
    #     self.net.train()
    #     with torch.no_grad():
    #         # in train and set metrics rewards are a tensor
    #
    #         # episode.rewards=torch.from_numpy(np.array(episode.rewards)).to(self.device)
    #         try:
    #             # print("episode.rewards",episode.rewards)
    #             episode.rewards = torch.stack(episode.rewards)
    #             # episode.values = torch.stack(episode.values)
    #         except:
    #             pass
    #         assert isinstance(episode.rewards, torch.Tensor)
    #
    #         loss, returns, quality, _, raw_loss = self.feed_network(episode)
    #         loss = loss.detach().item()
    #         returnn = returns.detach().item()
    #         episode.loss = loss
    #         episode.returnn = returnn
    #
    #         return quality, raw_loss
