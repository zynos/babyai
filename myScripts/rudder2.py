import gc

import torch
from myScripts.preProcess import PreProcess
import numpy as np
from .network import Net


class Rudder():

    def __init__(self, embed_dim, action_dim, n_lstm, image_dim=128, device="cpu", max_steps=128, own_net=True):
        self.network = Net(128 * 2, 7, 128 * 2, image_dim=image_dim, device=device,own_net=own_net).to(device=device)
        self.device = device
        self.pre_replay_buffer = []
        self.rewards = []
        self.timesteps = 0
        self.max_steps = max_steps
        self.own_net = own_net
        self.images = []
        self.instructions = []
        self.actions = []
        self.observations = []
        self.preProcess = PreProcess(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-5, weight_decay=1e-5)
        self.replay_buffer=[]
        self.updates =0
        self.class1=[]
        self.class0=[]
        self.total_episodes=0

    def chunk_pre_replay_buffer(self):
        start=len(self.replay_buffer)*self.max_steps
        end=start+self.max_steps
        print(start,end)
        ims=torch.cat(self.images,dim=1)[:,start:end]
        instrs=self.instructions[self.updates-1]
        acts=torch.cat(self.actions,dim=-1)[:,start:end]
        rews=torch.cat(self.rewards,dim=1)[:,start:end]
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        return  ims,instrs,acts,rews

    def zero_data_after_received_reward(self,ims, instrs, acts,rews):
        idx=rews.max(1)
        for i,line in enumerate(rews):
            values, indices = line.max(0)
            if values.item()>0:
                indices=line.gt(0.0).nonzero()[0].item()+1
                ims[i][indices:]*=0
                instrs[i][indices:] *= 0
                acts[i][indices:] *= 0
                line[indices:] *= 0
        return  ims, instrs, acts



    def pre_replay_buffer_to_batch(self):
        ims, instrs, acts,rews = self.chunk_pre_replay_buffer()
        ims, instrs, acts =self.zero_data_after_received_reward(ims, instrs, acts,rews)
        batch = ims, instrs, acts
        self.replay_buffer.append((batch,rews))
        pred=self.network.forward(ims, instrs, acts)
        loss,_=self.lossfunction(pred,rews)
        print("loss",loss.item())
        print("pred",list("{:.2f}".format(p.item()) for p in pred.squeeze()[0][:20]))
        print("rew ", list("{:.2f}".format(p.item()) for p in rews.squeeze()[0][:20]))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return batch

    def extend_pre_replay_buffer(self, net_inputs):
        if self.own_net:
            # images = []
            # instructions = []
            # actions = []
            # observations = []
            if self.own_net:
                ims, instrs, acts = net_inputs
                self.images.append(ims)
                self.instructions.append(instrs)
                self.actions.append(acts)
            else:
                obs, acts = net_inputs
                self.observations.append(obs)
                self.actions.append(acts)

    def extend_replay_buffers2(self, rewards, *net_inputs):
        self.timesteps += rewards.shape[1]
        self.updates+=1
        # self.pre_replay_buffer.append(net_inputs)
        self.rewards.append(rewards)
        self.extend_pre_replay_buffer(net_inputs)
        if self.timesteps >= self.max_steps:
            # put content of pre replay buffer together to one batch
            batch = self.pre_replay_buffer_to_batch()
            self.timesteps=0

    def clear_buffers(self):
        self.rewards=[]
        self.images=[]
        self.instructions=[]
        self.actions=[]



    def get_buffer_statistics(self):
        def get_mean_rew_and_mean_len(name,lis):
            rews=[el[-1] for el in lis]
            mean_rew = np.mean(rews)
            mean_len = np.mean([len(el[-2]) for el in lis])
            total = len(lis)
            max= np.max(rews)
            min = np.min(rews)
            print("class {}  mean episode len {:.2f}, total elements {}  rew mean {:.2} min {:.2} max {:.2}".
                  format(name,mean_len,total,mean_rew,min,max))
            return mean_rew,mean_len,len(lis)

        get_mean_rew_and_mean_len("0",self.class0)
        if len(self.class1)>0:
            get_mean_rew_and_mean_len("1", self.class1)
        else:
            print("class 1 is empty")


        #class 0





    def extend_replay_buffers(self, reward, im, instr, act, done):
        if self.own_net:
            self.rewards.append(reward[0])
            self.images.append(im)
            self.instructions.append(instr)
            self.actions.append(act[0])
            if done[0]:
                self.total_episodes +=1
                if sum(self.rewards)>0:
                    #add to class one
                    self.class1.append((self.images,self.instructions,self.actions,self.rewards,np.mean(self.rewards)))
                else:
                    self.class0.append((self.images, self.instructions, self.actions, self.rewards,np.mean(self.rewards)))
                self.clear_buffers()
                if self.total_episodes%100==0:
                    self.get_buffer_statistics()


        else:
            obs, acts = net_inputs
            self.observations.append(obs)
            self.actions.append(acts)

    def lossfunction(self, predictions, rewards):
        # from https://github.com/widmi/rudder-a-practical-tutorial/blob/master/tutorial.ipynb
        returns = rewards.sum(dim=1)
        predictions = predictions.squeeze(2)
        # Main task: predicting return at last timestep
        main_loss = torch.mean(predictions[:, -1] - returns) ** 2
        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        aux_loss = torch.mean(predictions[:, :] - returns[..., None]) ** 2
        # Combine losses
        loss = main_loss + aux_loss * 0.5
        return loss, (main_loss, aux_loss)


