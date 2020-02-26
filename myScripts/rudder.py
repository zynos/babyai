print("imported rudder")

import torch
# torch.backends.cudnn.benchmark = True
from myScripts.network import Net
from myScripts.replayBuffer import LessonReplayBuffer
import numpy as np
from torch.multiprocessing import Pool
from .preReplayBuffer import preReplayBuffer
from collections import Counter

# def train_old_samples(rudder):
#     end = False
#     ids = []
#     new_sample_losses=[]
#     while not end:
#         qualitys = []
#         for i in range(rudder.rudder_train_samples_per_epochs):
#             sample = rudder.replayBuffer.get_sample()
#             loss, quality =  rudder.train_rudder(sample)
#             new_sample_losses.append((loss.detach() , sample["id"]))
#             # rudder.replayBuffer.update_sample_loss(loss, sample["id"])
#             ids.append(sample["id"])
#             # print("loss {}, quality {}, sample {} ".format(loss.item(), quality.item(), sample["id"]))
#             qualitys.append((quality >= 0).item())
#         print("loss, qualities",loss.item(), qualitys)
#         if False in qualitys:
#             end = False
#         else:
#             end = True
#     idc = Counter(ids)
#     rudder.current_loss = loss
#     torch.cuda.empty_cache()
#     rudder.replayBuffer.added_new_sample = False
#     rudder.training_done = True
#     # self.rudder_net.lstm1.plot_internals(filename=None, show_plot=True, mb_index=0, fdict=dict(figsize=(8, 8), dpi=100))
#     # ret=copy.deepcopy(rudder.replayBuffer)
#     return new_sample_losses
    # return loss


class Rudder():
    # has   preReplayBuffer
    #       rudder lstm
    #       replay buffer
    def __init__(self,nr_procs,buffer_dict_fields,device,own_net,embed_mem_dim,image_dim,instr_dim):
        self.rudder_train_samples_per_epochs = 8
        self.preReplayBuffer=preReplayBuffer(nr_procs,buffer_dict_fields)
        self.nr_procs=nr_procs
        self.device=device
        self.rudder_net=Net(instr_dim, embed_mem_dim,7,32 , image_dim, device=device,own_net=own_net).to(device=device)
        self.optimizer = torch.optim.Adam(self.rudder_net.parameters(), lr=1e-3, weight_decay=1e-4)
        self.replayBuffer=LessonReplayBuffer(128, buffer_dict_fields)
        self.reward_scale = 20
        self.quality_threshold = 0.8
        # self.quality_threshold = -8
        self.current_loss = 0
        self.training_done = False
        # self.parallel_train_func=train_old_samples
        self.torch_spawn_context = None
        self.last_hidden=[None]*nr_procs

        # self.rudder_net.share_memory()
        # if self.torch_spawn_context == None:
        #     self.torch_spawn_context=torch.multiprocessing.spawn(self.parallel_train_func,args=(self.rudder_net,),join=False)

        # with Pool(1) as p:
        #     print(p.map(self.parallel_train_func, [self.rudder_net]))

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

    def pre_process_images(self, image):
        if not isinstance(image, tuple) and image.ndim==3:
            # [7 7 3] to [3 7 7]
            it= image.transpose(0,2).unsqueeze(0)
            x = self.rudder_net.image_conv(it)
            return x.squeeze(2).squeeze(2).unsqueeze(0)
        all_ims = []
        for idx, i in enumerate(image):
            # it = torch.transpose(torch.transpose(i, 1, 3), 2, 3)
            it = torch.transpose(i, 0, 2).unsqueeze(0)

            x = self.rudder_net.image_conv(it)
            all_ims.append(x.squeeze(2).squeeze(2))
        type(image)
        all_ims = torch.stack(all_ims)
        return all_ims

    def pre_process_instructions(self, instr):
        if not isinstance(instr, tuple) and instr.ndim == 1:
            return self.rudder_net.preProcess.get_gru_embedding(instr.unsqueeze(0)).unsqueeze(0)

        all_instrs = []
        for el in instr:
            el = el.unsqueeze(0)
            all_instrs.append(self.rudder_net.preProcess.get_gru_embedding(el))
        all_instrs = torch.stack(all_instrs)
        return all_instrs

    def create_input(self, image, instr, actions, embeds, batch):
        all_ims = torch.transpose(self.pre_process_images(image), 0, 1)
        all_instrs = torch.transpose(self.pre_process_instructions(instr), 0, 1)
        if  isinstance(actions, tuple):
            actions = torch.stack(actions)
        else:
            actions=actions.unsqueeze(0)
        if isinstance(embeds, tuple):
            embeds = torch.stack(embeds)
        else:
            embeds = embeds.unsqueeze(0)
        embeds=embeds.unsqueeze(0)
        one_hot = torch.nn.functional.one_hot(actions, 7).float().unsqueeze(0) #dangerous
        # x = x.reshape(x.shape[0], -1)
        input = torch.cat([all_ims, all_instrs, one_hot, embeds], dim=-1)
        if batch:
            input = input.transpose(0, 1)
        return input

    def feed_rudder(self,sample,hidden=None,batch=False):
        ims, instrs, acts,embeds = self.preprocess_batch(sample,batch)
        input=self.create_input(ims, instrs, acts,embeds,False)
        pred,hidden = self.rudder_net.forward(input,hidden)
        return pred,hidden

    def inference_rudder(self,sample):
        with torch.no_grad():
            rews = sample["reward"]
            rews = torch.tensor(rews, device=self.device).unsqueeze(0)
            pred,hidden = self.feed_rudder(sample)
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
        pred,hidden=self.feed_rudder(sample)
        loss, tup = self.lossfunction(pred, rews)
        loss.backward()
        self.optimizer.step()
        # print("loss", loss.item())
        loss=loss.detach().clone()
        del pred
        return loss,tup[0]
    def different_returns(self):
        s=self.replayBuffer.get_return_set()
        if len(s)>1:
            return True
        return False

    def buffer_full(self):
        return self.replayBuffer.buffersize>=self.replayBuffer.max_buffersize

    def predict_reward(self,proc_data):
        rews = []
        #input dict with [process nr and image, instr etc
        for proc_id in range(self.nr_procs):
            sample=dict()
            for key, value in proc_data.items():
                if key=="timestep":
                    sample[key] = value
                else:
                    sample[key]=value[proc_id]
            with torch.no_grad():
                pred, hidden = self.feed_rudder(sample, self.last_hidden[proc_id])
                if sample["done"]==True:
                    self.last_hidden[proc_id]=None
                else:
                    self.last_hidden[proc_id] = hidden
                try:
                    rews.append(pred.squeeze()[-1])
                except:
                    rews.append(pred.squeeze(1)[-1].squeeze())
            # assert np.sum(sample["reward"])<20

        return torch.stack(rews)

    # def predict_reward(self,proc_buffer_data):
    #     rews=[]
    #     for proc_id,sequence in proc_buffer_data.items():
    #         print("proc",proc_id)
    #         with torch.no_grad():
    #             pred,hidden=self.feed_rudder(sequence,self.last_hidden[proc_id])
    #             # pred = pred.clone().detach()
    #             self.last_hidden[proc_id]=hidden
    #             try:
    #                 rews.append(pred.squeeze()[-1])
    #             except:
    #                 rews.append(pred.squeeze(1)[-1].squeeze())
    #         assert np.sum(sequence["reward"])<20
    #
    #     return torch.stack(rews)

    # def train_old_samples(self):
    #     end=False
    #     ids=[]
    #     while not end:
    #         qualitys=[]
    #         for i in range(self.rudder_train_samples_per_epochs):
    #             sample=self.replayBuffer.get_sample()
    #             loss,quality=self.train_rudder(sample)
    #             self.replayBuffer.update_sample_loss(loss,sample["id"])
    #             ids.append(sample["id"])
    #             print("loss {}, quality {}, sample {} ".format(loss.item(),quality,sample["id"]))
    #             qualitys.append((quality>=0).item())
    #         print("qualities",qualitys)
    #         if False in qualitys:
    #             end=False
    #         else:
    #             end=True
    #     idc=Counter(ids)
    #     self.current_loss=loss
    #     torch.cuda.empty_cache()
    #     self.replayBuffer.added_new_sample = False
    #     self.training_done=True
    #     # self.rudder_net.lstm1.plot_internals(filename=None, show_plot=True, mb_index=0, fdict=dict(figsize=(8, 8), dpi=100))
    #     return loss

    def add_data(self,sample:dict,training_running):
        processes_data=dict()
        for process_id in range(self.nr_procs):
            # tmp = self.preReplayBuffer.replay_buffer_dict[process_id]
            # if tmp["reward"] and tmp["reward"][0] > 0:
            #     print("shits go down")
            timesteps=self.preReplayBuffer.add_timestep_data(sample,process_id)
            processes_data[process_id]=timesteps
            if not training_running:
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




