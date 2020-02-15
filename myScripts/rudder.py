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
        self.rudder_train_samples_per_epochs = 8
        self.preReplayBuffer=preReplayBuffer(nr_procs,buffer_dict_fields)
        self.nr_procs=nr_procs
        self.device=device
        self.rudder_net=Net(instr_dim, embed_mem_dim,7,128 , image_dim, device=device,own_net=own_net).to(device=device)
        self.optimizer = torch.optim.Adam(self.rudder_net.parameters(), lr=1e-4, weight_decay=1e-4)
        self.replayBuffer=LessonReplayBuffer(64, buffer_dict_fields)
        self.reward_scale = 20
        self.quality_threshold = 0.8
        self.current_loss = 0


    def preprocess_batch(self, sample,batch=False):
        ims, instrs, acts,embs= sample["image"], sample["instr"], sample["action"],sample["embed"]
        # if not batch:
        #     ims=torch.stack(ims).unsqueeze(0)
        #     # instrs=torch.tensor(instrs[0],device=self.device).unsqueeze(0)
        #     instrs = torch.stack(instrs).unsqueeze(0)
        #     acts=torch.stack(acts).unsqueeze(0)
        #     embs=torch.stack(embeds).unsqueeze(0)
        # else:
        #     ims=ims.unsqueeze(0)
        #     instrs=instrs.unsqueeze(0)
        #     acts = acts.unsqueeze(0)
        return ims,instrs,acts,embs

    def feed_rudder(self,sample,batch=False):
        ims, instrs, acts,embeds = self.preprocess_batch(sample,batch)
        pred = self.rudder_net.forward(ims, instrs, acts,embeds,batch)
        return pred

    def inference_rudder(self,sample):
        with torch.no_grad():
            rews = sample["reward"]
            rews = torch.tensor(rews, device=self.device).unsqueeze(0)
            pred = self.feed_rudder(sample)
            loss, _ = self.lossfunction(pred, rews)
            # loss = loss.clone().detach()
            # pred = pred.clone().detach()
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            return pred,loss.detach()

    def train_rudder(self,sample):
        self.optimizer.zero_grad()
        rews = sample["reward"]
        rews = torch.tensor(rews, device=self.device).unsqueeze(0)
        pred=self.feed_rudder(sample)
        loss, tup = self.lossfunction(pred, rews)
        loss.backward()
        self.optimizer.step()
        # print("loss", loss.item())
        loss=loss.clone().detach()
        del pred
        return loss,tup[0]
    def different_returns(self):
        s=self.replayBuffer.get_return_set()
        if len(s)>1:
            return True
        return False

    def buffer_full(self):
        return self.replayBuffer.buffersize>=self.replayBuffer.max_buffersize

    def predict_reward(self,proc_buffer_data):
        rews=[]
        for proc_id,sequence in proc_buffer_data.items():
            with torch.no_grad():
                pred=self.feed_rudder(sequence,False)
                pred = pred.clone().detach()

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
            for i in range(self.rudder_train_samples_per_epochs):
                sample=self.replayBuffer.get_sample()
                loss,quality=self.train_rudder(sample)
                self.replayBuffer.update_sample_loss(loss,sample["id"])
                ids.append(sample["id"])
                print("loss {}, quality {}, sample {} ".format(loss.item(),quality,sample["id"]))
                qualitys.append((quality>=0).item())
            print("qualities",qualitys)
            if False in qualitys:
                end=False
            else:
                end=True
        idc=Counter(ids)
        self.current_loss=loss
        torch.cuda.empty_cache()
        self.replayBuffer.added_new_sample = False
        # self.rudder_net.lstm1.plot_internals(filename=None, show_plot=True, mb_index=0, fdict=dict(figsize=(8, 8), dpi=100))
        return loss

    def add_data(self,sample:dict):
        processes_data=dict()
        for process_id in range(self.nr_procs):
            # tmp = self.preReplayBuffer.replay_buffer_dict[process_id]
            # if tmp["reward"] and tmp["reward"][0] > 0:
            #     print("shits go down")
            timesteps=self.preReplayBuffer.add_timestep_data(sample,process_id)
            processes_data[process_id]=timesteps
            if len(self.preReplayBuffer.send_to_rudder)>0:
                sort=sorted(self.preReplayBuffer.send_to_rudder, key = lambda i: len(i['timestep']),reverse = True)
                if len(sort)>1:
                    assert 0==0
                for i,batch in enumerate(sort):
                    with torch.no_grad():
                        pred,loss=self.inference_rudder(batch)
                        batch["loss"]=loss.item()
                        self.replayBuffer.consider_adding_sample(batch)
                        del loss
                        torch.cuda.empty_cache()

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
            main_loss=main_loss.clone().detach()
            aux_loss=aux_loss.clone().detach()
        return loss, (quality,main_loss, aux_loss)




