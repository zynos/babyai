from abc import ABC, abstractmethod
import torch

from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector
from myScripts.rudder import Net


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, aux_info,
                 use_rudder=False):
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
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        self.rudder = Net(128, 7, 256).cuda()
        # self.rudder2=Net(128*2,7,128*2).to(self.device)
        self.running_loss = 100.
        self.use_rudder = use_rudder
        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])

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

    def collect_experiences(self):
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
        embeddings = []
        actions = []
        rewards = []
        images = []
        instructs = []

        def lossfunction(predictions, rewards):
            # rewards = torch.tensor(rewards, device=self.device).reshape(1,-1,1)
            # returns = rewards.sum(dim=1)
            # # Main task: predicting return at last timestep
            # main_loss = torch.mean(predictions[:, -1] - returns) ** 2
            # # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
            # aux_loss = torch.mean(predictions[:, :] - returns[..., None]) ** 2
            # # Combine losses
            # loss = main_loss + aux_loss * 0.5
            newPreds = []
            newRews = []
            for j, r in enumerate(rewards):
                idx = (r > 0).nonzero()
                start = 0
                for i in idx:
                    target = torch.zeros(len(r))
                    chunk = predictions[j][start:i + 1]
                    target[:len(chunk)] = chunk.squeeze()
                    newPreds.append(target)
                    target = torch.zeros(len(r))
                    chunk = r[start:i + 1]
                    target[:len(chunk)] = chunk
                    newRews.append(target)
                    start = i + 2
                    # if start>len(r)-1:
                    #     break

            newRews = torch.stack(newRews)
            li = newRews > 0
            newPreds = torch.stack(newPreds)
            diff = newRews[li] - newPreds[li]
            print("predicted rew mean",newPreds.mean())
            l = torch.sum(diff ** 2)
            return l
            return loss

        def create_proc_dict(alls):
            procDict = dict()
            for timeStep in alls:
                for i, process in enumerate(timeStep):
                    try:
                        procDict[i].append(process)
                    except:
                        procDict[i] = [process]
            return procDict

        def do_my_stuff2(images, instrs):
            rewards2 = [torch.tensor(r, device=self.device).float() for r in rewards]
            rews = torch.stack(rewards2).transpose(0, 1)
            rew_mean = rews.mean()
            if rew_mean > 0.15:
                optimizer = torch.optim.Adam(self.rudder2.parameters(), lr=1e-3, weight_decay=1e-5)

                optimizer.zero_grad()
                acts = torch.stack(actions).transpose(0, 1).detach().clone()
                images = torch.stack(images).transpose(0, 1).detach().clone()
                instrs = instrs.detach().clone()
                pred = self.rudder2.forward(images, instrs, acts)
                rewards2 = [torch.tensor(r, device=self.device).float() for r in rewards]
                rews = torch.stack(rewards2).transpose(0, 1)
                rew_mean = rews.mean()
                loss = lossfunction(pred, rews)
                loss.backward()
                self.running_loss = self.running_loss * 0.99 + loss * 0.01
                optimizer.step()
                print("runn loss,loss,rew mean", self.running_loss.item(), loss.item(), rew_mean)

        optimizer = torch.optim.Adam(self.rudder.parameters(), lr=1e-5, weight_decay=1e-5)
        def do_my_stuff():
            rewards2 = [torch.tensor(r, device=self.device).float() for r in rewards]
            rews = torch.stack(rewards2).transpose(0, 1)
            rew_mean = rews.mean()

            if rew_mean > 0.12:
                embs = torch.stack(embeddings).transpose(0, 1)
                acts = torch.stack(actions).transpose(0, 1)
                # optimizer = torch.optim.Adam(self.rudder.parameters(), lr=1e-3)

                optimizer.zero_grad()
                pred = self.rudder(embs, acts)
                loss = lossfunction(pred, rews)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    self.running_loss = self.running_loss * 0.99 + loss * 0.01

                print("runn loss,loss,rew mean", self.running_loss.item(), loss.item(), rew_mean)

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                model_results = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                dist = model_results['dist']
                value = model_results['value']
                memory = model_results['memory']
                extra_predictions = model_results['extra_predictions']
                embed = model_results["embed"]
                embeddings.append(embed.clone().detach())

            action = dist.sample()
            actions.append(action.clone().detach())
            images.append(preprocessed_obs.image)
            instructs.append(preprocessed_obs.instr)
            obs, reward, done, env_info = self.env.step(action.cpu().numpy())
            rewards.append(reward)
            if self.aux_info:
                env_info = self.aux_info_collector.process(env_info)
                # env_info = self.process_aux_info(env_info)

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

        # Add advantage and return to experiences
        if self.use_rudder == True:
            do_my_stuff()
            # do_my_stuff2(images,preprocessed_obs.instr)
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
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass
