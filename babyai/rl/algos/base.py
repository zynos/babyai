import traceback

print("imported base")
import threading
from abc import ABC, abstractmethod
# from myScripts.rudder import train_old_samples
import numpy as np
import torch
# torch.multiprocessing.set_start_method('spawn',force=True)
# import torch.multiprocessing as mp
# from multiprocessing.reduce import ForkingPickler, AbstractReducer
from torch.multiprocessing import Process, Pool
# from multiprocessing import Process
from babyai.rl.format import default_preprocess_obss
from babyai.rl.utils import DictList, ParallelEnv
from babyai.rl.utils.supervised_losses import ExtraInfoCollector
from myScripts.rudder import Rudder
import copy




class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, aux_info,
                 use_rudder=False,rudder_own_net=False,env_max_steps=128):
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
        # mp.set_start_method('spawn')
        self.rudder_loss = -1337.0
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
        self.rudder_own_net = rudder_own_net
        # self.rudder=Rudder(acmodel.instr_dim,7,acmodel.memory_dim,acmodel.image_dim,self.device,env_max_steps,own_net=self.rudder_own_net)
        self.rudder_device="cuda"
        rudder_dict_keys=["reward","image","instr","action","done","embed","timestep"]
        self.rudder=Rudder( self.num_procs,rudder_dict_keys,self.rudder_device,rudder_own_net,acmodel.memory_dim,
                            acmodel.image_dim,acmodel.instr_dim)
        self.parallel_train_func=None
        print("before context")
        self.ctx = None
        # self.p = ctx.Process(target=self.parallel_train_func, args=(self.rudder,))
        # self.pool=self.ctx.Pool(1)#,maxtasksperchild=1)
        self.pool=None
        self.queue=None

        self.my_callback=None
        self.my_error_callback=None
        self.old_rew = None
        self.torch_spawn_context=None
        self.process_running = False
        self.async_status =None
        # self.download_thread = threading.Thread(target=self.rudder.train_old_samples())
        # if self.rudder_own_net:
        #     self.rudder = Net(128 * 2, 7, 128 * 2, device=self.device, own_net=self.rudder_own_net).to(self.device)
        # else:
        #     self.rudder = Net(128, 7, 256, device=self.device,own_net=self.rudder_own_net).to(self.device)

        self.running_loss = 100.
        self.use_rudder = use_rudder
        assert self.num_frames_per_proc % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])

        #APEX requires dtype=torch.float16
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
        dones = []
        old_rew=None

        def start_process():
            self.p.start()

        def start_pool():
            self.async_status=self.pool.map_async(self.parallel_train_func,(self.rudder,))#,callback=self.my_callback,error_callback=self.my_error_callback)
            self.async_status=self.pool.apply_async(self.parallel_train_func,(self.rudder,),callback=self.my_callback,error_callback=self.my_error_callback)
            self.torch_spawn_context=torch.multiprocessing.spawn(self.parallel_train_func, (self.rudder,self.queue,),join=False)

        def read_from_queue():
            # print("try to read from queue")
            if self.queue and not self.queue.empty():
                # print("try to read from queue 2", self.queue)
                loss_ids = self.queue.get(block=True)
                process_async_result(loss_ids)
                # print("second get")
                loss_ids = self.queue.get(block=True)
                # if loss_ids == "end":
                #     print("recieved all")
                # else:
                #     assert 0 == 1

                finish_process_reading()

        def reset_training_bools():
            self.rudder.training_done = True
            self.process_running = False

        def process_async_result(loss_ids):
            for tup in loss_ids:
                self.rudder.replayBuffer.update_sample_loss(tup[0], tup[1])


        def finish_process_reading():
            print("ended the process   ##############################################################################################################################")
            # print("rudder weights:")
            # print(self.rudder.rudder_net.fc_out.weight[0])
            self.p.join()
            reset_training_bools()
            self.queue.close()
            self.queue = self.ctx.Queue()
            self.p = self.ctx.Process(target=self.parallel_train_func, args=(self.rudder, self.queue,))


        def read_from_pool_result():
            if self.async_status and self.async_status.ready():
                print("ended the process   ##############################################################################################################################")
                print("but sadly not with callback")
                loss_ids=self.async_status.get()
                process_async_result(loss_ids)
                reset_training_bools()
                self.pool.close()
                self.async_status=None
                self.pool = self.ctx.Pool(1)#,maxtasksperchild=1)
                # self.rudder.replayBuffer=self.async_status.get()


        def train_asyncronized(use_process):
            self.rudder.rudder_net.share_memory()
            # print("rudder weights:")
            # print(self.rudder.rudder_net.fc_out.weight)
            if not self.process_running:  # self.async_status:
                self.process_running = True
                self.rudder.training_done = False
                if use_process:
                    start_process()
                else:
                    start_pool()
                print("started the process   ############################################################################################################################")
            if self.process_running:
                if use_process:
                    read_from_queue()
                else:
                    read_from_pool_result()






        def do_my_stuff2(action,image,instr,reward,done,embed,i):
            # iterate over every process

            dic=dict()
            dic["reward"]=reward.to(device=self.rudder_device)
            # dic["reward"]=torch.zeros(reward.shape).to(device=self.rudder_device)
            dic["image"]=image.to(device=self.rudder_device)
            # dic["image"]=torch.rand(image.shape).to(device=self.rudder_device)

            # dic["instr"] = torch.zeros(instr.shape,dtype=torch.long).to(device=self.rudder_device)
            dic["instr"] = instr.to(device=self.rudder_device)
            # dic["action"] = torch.zeros(action.shape,dtype=torch.int64).to(device=self.rudder_device)
            dic["action"] = action.to(device=self.rudder_device)
            dic["done"]=done
            # dic["embed"]=torch.zeros(embed.shape).to(device=self.rudder_device)
            dic["embed"] = embed.to(device=self.rudder_device)
            dic["timestep"]=i
            proc_data=self.rudder.add_data(dic,self.process_running)
            del proc_data

            # if torch.sum(torch.stack(proc_data[0]["reward"]))>20:
            #     print("bad")
            if self.rudder.buffer_full() and self.rudder.different_returns():
                # print(reward)
                if  self.rudder.replayBuffer.added_new_sample:
                    train_asyncronized(use_process=True)

                # self.rudder_loss=self.rudder.train_old_samples().item()
                if self.rudder.training_done:
                    rew=self.rudder.predict_reward(dic)
                    if type(self.old_rew) != type(None):
                        redistributed_reward = rew-self.old_rew

                    else:
                        redistributed_reward=0-rew
                    # del self.rewards[i]
                    self.rewards[i]=redistributed_reward.reshape(len(self.rewards[i]),)
                    self.old_rew=rew
                    assert 0==0
                    # print(reward)
            else:
                self.rudder_loss=0.0



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
            embeddings.append(embed.detach().clone())

            action = dist.sample()
            # actions.append(action.clone().detach())
            # images.append(preprocessed_obs.image)
            # instructs.append(preprocessed_obs.instr)
            obs, reward, done, env_info = self.env.step(action.cpu().numpy())
            if np.sum(reward)>0:
                mask=np.array(reward)>0
                r=np.array(reward)[mask]
                d=np.array(done)[mask]
                if False in d:
                    assert "Fuck" == "this"

            # rewards.append(reward)
            # dones.append(done)


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
            bla = self.rewards[i]
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)


            ###### MYSTUFF ########
            myrews=self.rewards[i].detach().clone()
            myIms=preprocessed_obs.image.detach().clone()
            myINstrs=preprocessed_obs.instr.detach().clone()
            myActs=action.detach().clone()
            # do_my_stuff2(action, preprocessed_obs.image,preprocessed_obs.instr, self.rewards[i], done,embed,i)
            do_my_stuff2(myActs, myIms, myINstrs, myrews, done, embed, i)
            # print("after my stuff",i)
            # if i==0:
            #     self.rewards_chache.append(self.rewards[i])
            ###### MYSTUFF ########



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
        # if self.use_rudder == True:
        #     if self.rudder_own_net:
        #         do_my_stuff2(images,preprocessed_obs.instr)
        #     else:
        #         do_my_stuff()
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
        # experiences exps
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
        # exps.myEmbeds=torch.stack(embeddings).detach().clone().transpose(0, 1).to(self.device)


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
            "RUD_L": self.rudder_loss
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass
