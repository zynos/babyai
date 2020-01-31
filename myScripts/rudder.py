import torch
from myScripts.network import Net
from myScripts.replayBuffer import LessonReplayBuffer

from .preReplayBuffer import preReplayBuffer





class Rudder():
    # has   preReplayBuffer
    #       rudder lstm
    #       replay buffer
    def __init__(self,nr_procs:int,buffer_dict_fields:list,device,own_net):
        self.preReplayBuffer=preReplayBuffer(nr_procs,buffer_dict_fields)
        self.nr_procs=nr_procs
        self.device=device
        self.rudder_net=Net(128 * 2, 7, 128 * 2, image_dim=128, device=device,own_net=own_net).to(device=device)
        self.optimizer = torch.optim.Adam(self.rudder_net.parameters(), lr=1e-5, weight_decay=1e-5)
        self.replayBuffer=LessonReplayBuffer(200, buffer_dict_fields)



    def preprocess_batch(self, sample,batch=False):
        ims, instrs, acts= sample["image"], sample["instr"], sample["action"]
        if not batch:
            ims=torch.stack(ims).unsqueeze(0)
            # instrs=torch.tensor(instrs[0],device=self.device).unsqueeze(0)
            instrs = torch.stack(instrs).unsqueeze(0)
            acts=torch.stack(acts).unsqueeze(0)
        else:
            ims=ims.unsqueeze(0)
            instrs=instrs.unsqueeze(0)
            acts = acts.unsqueeze(0)
        return ims,instrs,acts

    def feed_rudder(self,sample,batch=False):
        ims, instrs, acts = self.preprocess_batch(sample,batch)
        pred = self.rudder_net.forward(ims, instrs, acts,batch)
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
        loss, _ = self.lossfunction(pred, rews)
        loss.backward()
        self.optimizer.step()
        # print("loss", loss.item())
        return loss,pred
    def buffer_full(self):
        return self.replayBuffer.buffersize>=self.replayBuffer.max_buffersize

    def predict_reward(self,sample):
        with torch.no_grad():
            pred=self.feed_rudder(sample,True)
            return pred

    def train_old_sample(self):
        sample=self.replayBuffer.get_sample()
        loss,pred=self.train_rudder(sample)
        self.replayBuffer.update_sample_loss(loss,sample["id"])
        return loss

    def add_data(self,sample:dict):
        for process_id in range(self.nr_procs):
            self.preReplayBuffer.add_timestep_data(sample,process_id)
            data_for_rudder=self.preReplayBuffer.send_to_rudder
            if len(data_for_rudder)>0:
                for i,batch in enumerate(data_for_rudder):
                    with torch.no_grad():
                        pred,loss=self.inference_rudder(batch)
                        batch["loss"]=loss.item()
                        self.replayBuffer.consider_adding_sample(batch)

                self.preReplayBuffer.send_to_rudder=[]

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




