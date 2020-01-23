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
        self.replayBuffer=LessonReplayBuffer(20, buffer_dict_fields)

    def preprocess_batch(self,batch):
        ims, instrs, acts= batch["image"], batch["instr"], batch["action"]
        ims=torch.stack(ims).unsqueeze(0)
        instrs=torch.tensor(instrs[0],device=self.device).unsqueeze(0)
        acts=torch.stack(acts).unsqueeze(0)
        return ims,instrs,acts

    def train_rudder(self,batch):
        ims, instrs, acts=self.preprocess_batch(batch)
        rews=batch["reward"]
        rews = torch.tensor(rews, device=self.device).unsqueeze(0)
        pred = self.rudder_net.forward(ims, instrs, acts)
        loss, _ = self.lossfunction(pred, rews)
        print("loss", loss.item())
        return loss

    def add_data(self,sample:dict):
        for process_id in range(self.nr_procs):
            self.preReplayBuffer.add_timestep_data(sample,process_id)
            data_for_rudder=self.preReplayBuffer.send_to_rudder
            if len(data_for_rudder)>0:
                data_for_replay_buffer=dict()
                for i,batch in enumerate(data_for_rudder):
                    with torch.no_grad():
                        loss=self.train_rudder(batch)
                        batch["loss"]=loss.item()
                        self.replayBuffer.consider_adding_sample(batch)

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




