from abc import ABC, abstractmethod

import numpy as np
import torch
# import multiprocessing as mp
# import numpy
import copy
from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector
from myScripts.ReplayBuffer import ReplayBuffer
from rudder.rudder_main import Rudder
from myScripts.asyncTrain import start_background_process


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, aux_info,model_name=None):
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
        self.model_name = "default"
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
        self.obss = np.empty(shape[0],dtype=object)

        # APEX requires dtype=torch.float16
        # self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device,dtype=torch.float16)
        # self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device,dtype=torch.float16)

        self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.dones = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.embeddings = torch.zeros(*shape, self.acmodel.image_dim, device=self.device)
        self.rudder_values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.rudder_rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.rudder_advantages = torch.zeros(*shape, device=self.device)
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
        self.use_rudder = True
        # if self.use_rudder:
        #     assert self.num_frames_per_proc == self.recurrence
        # self.rudder = Rudder(self.num_procs, acmodel.obs_space,
        #                      acmodel.instr_dim, acmodel.memory_dim, acmodel.image_dim,
        #                      acmodel.action_space.n, self.device)
        self.rudder = Rudder(self.num_procs, self.device, 40,
                             acmodel.instr_dim, acmodel.memory_dim, acmodel.image_dim, model_name,lr, self)

        # self.ctx=mp.get_context("spawn")
        # self.queue=self.ctx.Queue()
        # self.async_func=start_background_process
        # self.background_process=self.ctx.Process(target=self.async_func, args=(self.rudder,self.queue,))
        self.p = None
        self.queue_into_rudder = None
        self.queue_back_from_rudder = None

    # def update_rudder_and_rescale_rewards(self, obs, update_nr, i, queue_into_rudder, queue_back_from_rudder, embedding,
    #                                       action, rewards, done,
    #                                       instr, image, value):
    #     ### SYNCHRONOUS
    #     # embeddings,actions,rewards,dones,instructions,images
    #     rudder_loss, last_ts_pred, full_pred = 0.0, 0.0, 0.0
    #     debug = update_nr >= 6 and i >= 15
    #     debug = False
    #     rewards = rewards / 20
    #
    #     self.rudder.add_timestep_data(debug, queue_into_rudder, embedding, action, rewards, done,
    #                                   instr, image, value, obs)
    #     # if debug:
    #     #     print("after add_timestep_data", i)
    #     # if update_nr == 6 and i == 14:
    #     #     print("d")
    #     # #
    #     if self.rudder.first_training_done:
    #         # print(i,image.shape)
    #         #     ret=self.rudder.predict_reward(embedding, action, rewards, done,
    #         #                                                  instr,
    #         #                                                  image)
    #         ret = self.rudder.new_predict_reward(done, i)
    #         if ret is not None:
    #             ret = ret * 20
    #             # ret=torch.clamp(ret,-10.0,10.0)
    #             self.rudder_rewards = ret.transpose(0, 1)
    #     if self.rudder.replay_buffer.buffer_full() and self.rudder.replay_buffer.encountered_different_returns() and i == 39:
    #         rudder_loss, last_ts_pred, full_pred = self.rudder.train_full_buffer()
    #         self.rudder.first_training_done = True
    #         nonzeros = np.count_nonzero(self.rudder.replay_buffer.fast_returns)
    #         percent = nonzeros / self.rudder.replay_buffer.max_size
    #         self.rudder.replay_buffer.nonzero_percent = percent
    #         print('non zero returns and percent', nonzeros, percent)
    #
    #         # print(ret)
    #         # self.rudder_rewards[i] = ret
    #     self.rudder.replay_buffer.init_process_data(self.rudder.replay_buffer.procs_to_init, i)
    #     #     # print("rudder rewards")
    #
    #     return rudder_loss, last_ts_pred, full_pred

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
                rudder_value = model_results['rudder_value']
                logits = model_results['logits']

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
            self.dones[i] = torch.tensor(done, device=self.device, dtype=torch.float)
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            self.rudder_values[i] = rudder_value
            self.embeddings[i] = embedding
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            # print("checkpoint 7")
            # RUDDER entry
            rudder_loss,rudder_aux, last_ts_pred, last_rew_mean = 0, 0, 0, 0
            # if self.use_rudder:
            # rudder_loss, last_ts_pred, last_rew_mean = \
            #     self.update_rudder_and_rescale_rewards(obs,update_nr, i, self.queue_into_rudder,
            #                                            self.queue_back_from_rudder, embedding,
            #                                            action, self.rewards[i], done, preprocessed_obs.instr,
            #                                            preprocessed_obs.image, value)

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
        if self.use_rudder:
            self.rudder.fill_buffer_batch(self.masks.detach().clone(), self.rewards.detach().clone(),
                                          self.values.detach().clone(),
                                          self.actions.detach().clone(), copy.deepcopy(self.obss), self.dones.detach().clone(),
                                          self.embeddings.detach().clone(),update_nr,self.model_name)

            if self.rudder.replay_buffer.buffer_full() and self.rudder.replay_buffer.encountered_different_returns():
                rudder_loss,rudder_aux, rud_grad_norm = self.rudder.train_on_buffer_data()
                self.rudder.grad_norm = rud_grad_norm
                self.rudder_rewards = self.rudder.predict_new_rewards_batch(copy.deepcopy(self.obss), self.masks.detach().clone(),
                                                                            self.rewards.detach().clone(),
                                                                            self.values.detach().clone(),
                                                                            self.actions.detach().clone(),
                                                                            self.dones.detach().clone(),
                                                                            self.embeddings.detach().clone())
                # self.rudder_rewards *= 20

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            next_value = model_results['value']
            next_rudder_value = model_results['rudder_value']

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i + 1] if i < self.num_frames_per_proc - 1 else next_value
            next_rudder_value = self.rudder_values[i + 1] if i < self.num_frames_per_proc - 1 else next_rudder_value

            next_advantage = self.advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0
            next_rudder_advantage = self.rudder_advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0
            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            rudder_delta = self.rudder_rewards[i] + self.discount * next_rudder_value * next_mask - self.rudder_values[
                i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask
            self.rudder_advantages[
                i] = rudder_delta + self.discount * self.gae_lambda * next_rudder_advantage * next_mask

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
        exps.rudder_value = self.rudder_values.transpose(0, 1).reshape(-1)
        exps.rudder_value = torch.clamp(exps.rudder_value, torch.min(exps.value).item(),
                                        torch.max(exps.value).item())
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.rudder_advantage = self.rudder_advantages.transpose(0, 1).reshape(-1)
        rud_orig_val = torch.mean(exps.value)
        rud_rud_val = torch.mean(exps.rudder_value)
        rud_orig_adv = torch.mean(exps.advantage)
        rud_rud_adv = torch.mean(exps.rudder_advantage)
        print("valueX {:.1f} rudValueX {:.1f} advantX {:.1f} rudAdvantX {:.1f}".format(rud_orig_val, rud_rud_val,
                                                                                       rud_orig_adv,
                                                                                       rud_rud_adv))
        # a =a_o(1-qualityv) +a_r * quality
        exps.rudder_advantage = torch.clamp(exps.rudder_advantage, torch.min(exps.advantage).item(),
                                            torch.max(exps.advantage).item())
        if self.use_rudder:
            exps.advantage = exps.advantage * (
                    1 - self.rudder.current_quality) + exps.rudder_advantage * self.rudder.current_quality
        exps.returnn = exps.value + exps.advantage

        exps.rudder_return = exps.rudder_value + exps.rudder_advantage
        exps.rudder_return = torch.clamp(exps.rudder_return, torch.min(exps.returnn).item(),
                                         torch.max(exps.returnn).item())
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        if self.aux_info:
            exps = self.aux_info_collector.end_collection(exps)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        log = {
            "rud_rud_rew": torch.mean(self.rudder_rewards),
            "rud_orig_val": rud_orig_val,
            "rud_rud_val": rud_rud_val,
            "rud_orig_adv": rud_orig_adv,
            "rud_rud_adv": rud_rud_adv,
            "rud_return": torch.mean(exps.rudder_return),
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episodes_done": self.log_done_counter,
            "rudder_loss": rudder_loss,
            "rudder_aux": rudder_aux,
            "rudder_pred_last": last_ts_pred,
            "LastRew_mean": last_rew_mean,
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
