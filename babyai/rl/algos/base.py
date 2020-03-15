from abc import ABC, abstractmethod
import torch
# import multiprocessing as mp
# import numpy

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector
from myScripts.ReplayBuffer import ReplayBuffer
from myScripts.Rudder import Rudder
from myScripts.asyncTrain import start_background_process


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, aux_info):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        aux_info : list
            a list of strings corresponding to the name of the extra information
            retrieved from the environment for supervised auxiliary losses

        """
        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.aux_info = aux_info

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device="cpu"
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])

        # APEX requires dtype=torch.float16
        # self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device,dtype=torch.float16)
        # self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device,dtype=torch.float16)

        self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        if self.aux_info:
            self.aux_info_collector = ExtraInfoCollector(self.aux_info, shape, self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

        # RUDDER changes
        self.rudder = Rudder(acmodel.memory_dim, self.num_procs, acmodel.obs_space,
                             acmodel.instr_dim, acmodel.memory_dim, acmodel.image_dim,
                             acmodel.action_space, self.device)
        # self.ctx=mp.get_context("spawn")
        # self.queue=self.ctx.Queue()
        # self.async_func=start_background_process
        # self.background_process=self.ctx.Process(target=self.async_func, args=(self.rudder,self.queue,))
        self.p = None
        self.queue_into_rudder = None
        self.queue_back_from_rudder = None

    def update_rudder_and_rescale_rewards(self, update_nr, i, queue_into_rudder, queue_back_from_rudder, embedding,
                                          action, rewards, done,
                                          instr, image):
        ### SYNCHRONOUS
        # embeddings,actions,rewards,dones,instructions,images
        rudder_loss, last_ts_pred, full_pred = 0.0, 0.0, 0.0
        debug = update_nr >= 6 and i >= 15
        debug = False
        self.rudder.add_timestep_data(debug, queue_into_rudder, embedding, action, rewards, done,
                                      instr, image)
        # if debug:
        #     print("after add_timestep_data", i)
        # if update_nr == 6 and i == 14:
        #     print("d")
        # #
        if self.rudder.replay_buffer.buffer_full() and self.rudder.replay_buffer.encountered_different_returns() and i % 39 == 0:
            rudder_loss, last_ts_pred, full_pred = self.rudder.train_full_buffer()
            self.rudder.first_training_done = True
        if self.rudder.first_training_done:
            self.rewards[i] = self.rudder.predict_reward(embedding, action, rewards, done,
                                                         instr,
                                                         image)
        #     # print("rudder rewards")

        ### ASYNCHRONOUS

        # if self.rudder.replay_buffer.buffer_full() and self.queue_into_rudder.empty():
        #     self.queue_into_rudder.put(self.rudder.replay_buffer.get_cloned_copy())
        # # print("qsize", self.queue_into_rudder.qsize())
        # # print("i , self.queue_back_from_rudder.empty()",i, self.queue_back_from_rudder.empty())
        #
        # if self.rudder.replay_buffer.buffer_full() and self.rudder.replay_buffer.encountered_different_returns() and i % 39 == 0:
        #     print("recalc")
        #     self.rudder.recalculate_all_losses()
        #     print("recalc done")
        # if not self.queue_back_from_rudder.empty():
        #     print("wanna get queue food")
        #     self.rudder.first_training_done = queue_back_from_rudder.get()
        #     self.rudder.parallel_train_done = True
        #     assert isinstance(self.rudder.first_training_done, bool)
        # if self.rudder.first_training_done:
        #     self.rewards[i] = self.rudder.predict_reward(embedding, action, rewards, done,
        #                                                  instr, image)
        # assert 0==1

        return rudder_loss, last_ts_pred, full_pred

    def collect_experiences(self, update_nr):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.

        """
        # print("checkpoint 2")
        # if not self.p.is_alive() and not self.p.exitcode:
        #     print("checkpoint 3")
        #     self.rudder.net.share_memory()
        #     self.p.start()
        # print(self.p.exitcode)

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            # print("checkpoint 4")

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']
                embedding = model_results['embedding']

            action = dist.sample()
            # print("checkpoint 5")
            obs, reward, done, env_info = self.env.step(action.cpu().numpy())
            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)
            # print("checkpoint 6")
            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs

            self.memories[i] = self.memory
            self.memory = memory

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            # print("checkpoint 7")
            # RUDDER entry
            rudder_loss, last_ts_pred, full_pred = \
                self.update_rudder_and_rescale_rewards(update_nr, i, self.queue_into_rudder,
                                                       self.queue_back_from_rudder, embedding,
                                                       action, self.rewards[i], done, preprocessed_obs.instr,
                                                       preprocessed_obs.image)

            self.log_probs[i] = dist.log_prob(action)

            if self.aux_info:
                self.aux_info_collector.fill_dictionaries(i, env_info, extra_predictions)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
            # print("checkpoint 8")

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            next_value = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))['value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i + 1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Flatten the data correctly, making sure that
        # each episode's data is a continuous chunk
        # print("checkpoint 9")
        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        # In commments below T is self.num_frames_per_proc, P is self.num_procs,
        # D is the dimensionality

        # T x P x D -> P x T x D -> (P * T) x D
        exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
            "rudder_loss":rudder_loss,
                "rudder_pred_last":last_ts_pred,
        "rudder_pred_sum":full_pred,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]
        # print("checkpoint 10")

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass
