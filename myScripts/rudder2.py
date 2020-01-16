import torch
from myScripts.preProcess import PreProcess

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


    def chunk_pre_replay_buffer(self):
        start=len(self.replay_buffer)*self.max_steps
        end=start+self.max_steps
        print(start,end)
        ims=torch.cat(self.images,dim=1)[:,start:end]
        instrs=self.instructions[self.updates-1]
        acts=torch.cat(self.actions,dim=-1)[:,start:end]
        rews=torch.cat(self.rewards,dim=1)[:,start:end]
        return  ims,instrs,acts,rews

    def zero_data_after_received_reward(self,ims, instrs, acts,rews):
        idx=rews.max(1)
        for i,line in enumerate(rews):
            values, indices = line.max(0)
            if values.item()>0:
                ims[i][indices:]*=0
                instrs[i][indices:] *= 0
                acts[i][indices:] *= 0
        return  ims, instrs, acts



    def pre_replay_buffer_to_batch(self):
        ims, instrs, acts,rews = self.chunk_pre_replay_buffer()
        ims, instrs, acts =self.zero_data_after_received_reward(ims, instrs, acts,rews)
        batch = ims, instrs, acts
        self.replay_buffer.append((batch,rews))
        pred=self.network.forward(ims, instrs, acts)
        loss,_=self.lossfunction(pred,rews)
        print(loss.item())
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

    def extend_replay_buffers(self, rewards, *net_inputs):
        self.timesteps += rewards.shape[1]
        self.updates+=1
        # self.pre_replay_buffer.append(net_inputs)
        self.rewards.append(rewards)
        self.extend_pre_replay_buffer(net_inputs)
        if self.timesteps >= self.max_steps:
            # put content of pre replay buffer together to one batch
            batch = self.pre_replay_buffer_to_batch()
            self.timesteps=0

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


