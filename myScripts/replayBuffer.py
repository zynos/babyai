import numpy as np
import torch
from scipy.stats import rankdata
import copy

class LessonReplayBuffer():
    def __init__(self, max_buffersize, dict_fields: list,rnd_gen=None):
        self.use_list = True
        self.max_buffersize=max_buffersize
        self.replaced_indices=set()
        if self.use_list:
            self.replay_buffer_list=[]
        else:
            self.replay_buffer_dict=dict()
        self.temperature=1.
        self.buffersize=0
        self.added_new_sample = True
        if rnd_gen is None:
            rnd_gen = np.random.RandomState()
        self.rnd_gen = rnd_gen

    def get_return_set(self):
        if self.use_list:
            returns = [np.sum(v["reward"]).item() for v in self.replay_buffer_list]
        else:
            returns=[np.sum(v["reward"]).item() for k,v in self.replay_buffer_dict.items()]
        return set(returns)

    def get_mean_return(self):
        if self.use_list:
            returns = [np.sum(v["reward"]) for v in self.replay_buffer_list]
        else:
            returns=[np.sum(v["reward"]) for k,v in self.replay_buffer_dict.items()]
        return np.sum(returns)/len(returns)


    def add_sample(self, sample):
        """Add sample to buffer, assign id"""
        assert np.sum(sample["reward"])<20
        print("add sample",self.buffersize)
        if self.use_list:
            self.replay_buffer_list.append(sample)
        else:
            self.replay_buffer_dict[self.buffersize]=sample
        self.buffersize+=1

    def get_losses(self):
        """Get all losses in buffer"""
        if self.use_list:
            keys_losses = [(i, self.replay_buffer_dict[i]["loss"]) for i, e in self.replay_buffer_dict.keys()]
        else:
            keys_losses = [(k, self.replay_buffer_dict[k]["loss"]) for k in self.replay_buffer_dict.keys()]
        return keys_losses

    def replace_entry(self,id,sample):
        assert np.sum(sample["reward"]) < 20
        # del self.replay_buffer[id]
        # print(torch.cuda.memory_summary)

        if self.use_list:
            # id=np.random.randint(len(self.replay_buffer_list))
            # print("replace",id)
            # old=self.replay_buffer_list[id]
            # del old
            # torch.cuda.empty_cache()
            #        rudder_dict_keys=["reward","image","instr","action","done","embed","timestep"]

            self.replay_buffer_list[id]=None
            sample2=dict()
            sample2["loss"]=sample["loss"]
            sample2["reward"]=sample["reward"]
            # sample2["image"] = sample["image"] # leak
            # sample2["instr"] = sample["instr"] # no leak or mini mini leak
            # sample2["embed"] = torch.rand_like(torch.stack(sample["embed"]))
            sample2["embed"] = sample["embed"] # mega leak
            assert sample2["embed"] != None
            del sample["embed"]
            torch.cuda.empty_cache()
            # sample2["done"]=sample["done"]
            self.replay_buffer_list[id]=sample2
            del sample
            torch.cuda.empty_cache()
            assert self.replay_buffer_list[id] != None
        else:
            new_dict=dict()
            for k,v in self.replay_buffer_dict.items():
                new_dict[k]=v
                del v
                torch.cuda.empty_cache()
            new_dict[id]=sample
            del sample
            torch.cuda.empty_cache()
            del self.replay_buffer_dict
            torch.cuda.empty_cache()
            # s=torch.cuda.memory_summary(device=0)
            # print(s)
            self.replay_buffer_dict=new_dict
            assert self.replay_buffer_dict[id] != None


        # self.replay_buffer[id]=sample


    def consider_adding_sample(self, sample: dict):
        """ Show sample to buffer; Buffer decides whether to add it or not based on sample loss;

        Sample must at least contain the key 'loss'; For usage with RUDDER example code, see LessonReplayBuffer class
        docstring;
        """
        assert np.sum(sample["reward"]) < 20
        if self.buffersize < self.max_buffersize:
            # Add sample if buffer is not full
            self.add_sample(sample)
        else:
            # Replace sample with lowest loss in buffer if new sample loss is higher
            keys_ranks = self.get_ranks(sample)
            sample_rank=keys_ranks[-1][1]

            low_sample_ind = np.argmin([b[1] for b in keys_ranks])
            low_sample_rank=keys_ranks[low_sample_ind][1]
            if sample_rank > low_sample_rank:
                id =keys_ranks[low_sample_ind][0]
                # print("replace {} with {} id {}".format(low_sample_rank,sample_rank,id))
                # id = np.random.randint(len(self.replay_buffer_list))
                self.replaced_indices.add(id)
                self.replace_entry(id,sample)
                self.added_new_sample=True
                diff=set(range(self.max_buffersize))-self.replaced_indices
                if len(diff)==1:
                    single_id=next(iter(diff))
                    print("id, loss , return",single_id,self.replay_buffer_list[single_id]["loss"],
                          self.replay_buffer_list[single_id]["reward"][-1])
                if len(diff)==0:
                    self.replaced_indices=set()
                print("never replaced:",diff)
            else:
                self.added_new_sample = False

    def softmax(self, x):
        e_logits = np.exp((x-np.max(x))/self.temperature)
        return  e_logits/ e_logits.sum()

    def get_ranks(self,in_sample=None):
        def append_stuff():
            try:
                return_distances.append(np.abs(mean_ret.cpu() - np.sum(sample["reward"])))
            except:
                return_distances.append(np.abs(mean_ret - np.sum(sample["reward"])))
            losses.append(sample["loss"])
            ids.append(id)
        mean_ret=self.get_mean_return()
        return_distances=[]
        losses=[]
        ids=[]
        if self.use_list:
            for id,sample in enumerate(self.replay_buffer_list):
                append_stuff()
        else:
            for id,sample in self.replay_buffer_dict.items():
                append_stuff()


        if in_sample:
            losses.append(in_sample["loss"])
            # return_distances.append(np.abs(mean_ret.cpu()-np.sum(in_sample["reward"])))
            try:
                return_distances.append(np.abs(mean_ret.cpu()-np.sum(in_sample["reward"])))
            except:
                return_distances.append(np.abs(mean_ret - np.sum(in_sample["reward"])))
            #has no id yet
            ids.append(-1)

        rd_ranks=rankdata(return_distances)
        losses_ranks=rankdata(losses)
        combined_ranks=rd_ranks+losses_ranks
        ret=list(zip(ids,combined_ranks))
        return ret



    def get_sample(self):
        """Randomly sample episode or game with from buffer; Sampling probabilities are softmax values of losses in
        buffer;"""
        # keys_losses = self.get_losses()
        keys_losses = self.get_ranks()

        losses = [b[1] for b in keys_losses]
        ids = [b[0] for b in keys_losses]
        losses=np.array(losses,dtype=float)
        probs = self.softmax(losses)
        sample_id = self.rnd_gen.choice(ids, p=probs)
        if self.use_list:
            sample = self.replay_buffer_list[sample_id]
        else:
            sample = self.replay_buffer_dict[sample_id]
        sample['id'] = sample_id
        return sample

    def update_sample_loss(self, loss, id):
        """Update loss of sample id"""
        if self.use_list:
            self.replay_buffer_list[id]['loss'] = loss
        else:
            self.replay_buffer_dict[id]['loss'] = loss