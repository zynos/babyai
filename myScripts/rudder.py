import torch
from myScripts.network import Net
from myScripts.replayBuffer import LessonReplayBuffer
import numpy as np
from .preReplayBuffer import preReplayBuffer
from collections import Counter




class Rudder():
    # has   preReplayBuffer
    #       rudder lstm
    #       replay buffer
    def __init__(self,nr_procs:int,buffer_dict_fields:list,device,own_net,embed_mem_dim,image_dim,instr_dim):
        self.rudder_train_epochs = 8
        self.preReplayBuffer=preReplayBuffer(nr_procs,buffer_dict_fields)
        self.nr_procs=nr_procs
        self.device=device
        self.rudder_net=Net(instr_dim, embed_mem_dim,7, 128 * 2, image_dim, device=device,own_net=own_net).to(device=device)
        self.optimizer = torch.optim.Adam(self.rudder_net.parameters(), lr=1e-4, weight_decay=1e-4)
        self.replayBuffer=LessonReplayBuffer(512, buffer_dict_fields)
        self.reward_scale = 20
        self.quality_threshold = 0.8



    def preprocess_batch(self, sample,batch=False):
        ims, instrs, acts,embeds= sample["image"], sample["instr"], sample["action"],sample["embed"]
        if not batch:
            ims=torch.stack(ims).unsqueeze(0)
            # instrs=torch.tensor(instrs[0],device=self.device).unsqueeze(0)
            instrs = torch.stack(instrs).unsqueeze(0)
            acts=torch.stack(acts).unsqueeze(0)
            embs=torch.stack(embeds).unsqueeze(0)
        else:
            ims=ims.unsqueeze(0)
            instrs=instrs.unsqueeze(0)
            acts = acts.unsqueeze(0)
        return ims,instrs,acts,embs

    def feed_rudder(self,sample,batch=False):
        ims, instrs, acts,embeds = self.preprocess_batch(sample,batch)
        pred = self.rudder_net.forward(ims, instrs, acts,embeds,batch)
        return pred

    def inference_rudder(self,sample):
        rews = sample["reward"]
        rews = torch.tensor(rews, device=self.device).unsqueeze(0)
        pred = self.feed_rudder(sample)
        loss, _ = self.lossfunction(pred, rews)
        return pred,loss

    def train_rudder(self,sample):
        self.optimizer.zero_grad()
        rews = sample["reward"]
        rews = torch.tensor(rews, device=self.device).unsqueeze(0)
        pred=self.feed_rudder(sample)
        loss, tup = self.lossfunction(pred, rews)
        loss.backward()
        self.optimizer.step()
        # print("loss", loss.item())
        return loss,pred,tup[0]
    def buffer_full(self):
        return self.replayBuffer.buffersize>=self.replayBuffer.max_buffersize

    def predict_reward(self,proc_buffer_data):
        rews=[]
        for proc_id,sequence in proc_buffer_data.items():
            with torch.no_grad():
                pred=self.feed_rudder(sequence,False)
                try:
                    rews.append(pred.squeeze()[-1])
                except:
                    rews.append(pred.squeeze(1)[-1].squeeze())
            assert np.sum(sequence["reward"])<20

        return torch.stack(rews)

    def train_old_samples(self):
        end=False
        ids=[]
        while not end:
            qualitys=[]
            for i in range(self.rudder_train_epochs):
                sample=self.replayBuffer.get_sample()
                loss,pred,quality=self.train_rudder(sample)
                self.replayBuffer.update_sample_loss(loss,sample["id"])
                ids.append(sample["id"])
                print("loss {}, quality {}, sample {} ".format(loss.item(),quality,sample["id"]))
                qualitys.append(quality>=0)
            if False in qualitys:
                end=False
            else:
                end=True
        idc=Counter(ids)
        # self.rudder_net.lstm1.plot_internals(filename=None, show_plot=True, mb_index=0, fdict=dict(figsize=(8, 8), dpi=100))
        return loss

    def add_data(self,sample:dict):
        processes_data=dict()
        for process_id in range(self.nr_procs):
            tmp = self.preReplayBuffer.replay_buffer_dict[process_id]
            if tmp["reward"] and tmp["reward"][0] > 0:
                print("shits go down")
            timesteps=self.preReplayBuffer.add_timestep_data(sample,process_id)
            processes_data[process_id]=timesteps
            data_for_rudder=self.preReplayBuffer.send_to_rudder
            if len(data_for_rudder)>0:
                for i,batch in enumerate(data_for_rudder):
                    with torch.no_grad():
                        pred,loss=self.inference_rudder(batch)
                        batch["loss"]=loss.item()
                        self.replayBuffer.consider_adding_sample(batch)

                self.preReplayBuffer.send_to_rudder=[]
        return processes_data

    def lossfunction(self, predictions, rewards):
        # from https://github.com/widmi/rudder-a-practical-tutorial/blob/master/tutorial.ipynb
        returns = rewards.sum(dim=1)
        predictions = predictions.squeeze(2)
        # Main task: predicting return at last timestep
        diff=predictions[:, -1] - returns
        main_loss = torch.mean(diff) ** 2
        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        aux_loss = torch.mean(predictions[:, :] - returns[..., None]) ** 2
        # Combine losses
        loss = main_loss + aux_loss * 0.5
        with torch.no_grad():
            quality=1-(torch.abs(diff)/self.reward_scale) *(1/(1-self.quality_threshold))
            # print("quality",quality)
        return loss, (quality,main_loss, aux_loss)




