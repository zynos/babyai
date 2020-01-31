import numpy as np
import torch


class LessonReplayBuffer():
    def __init__(self, max_buffersize, dict_fields: list,rnd_gen=None):
        self.max_buffersize=max_buffersize
        self.replay_buffer=dict()
        self.temperature=1.
        self.buffersize=0
        if rnd_gen is None:
            rnd_gen = np.random.RandomState()
        self.rnd_gen = rnd_gen

    def get_mean_return(self):
        returns=[np.sum(v["reward"]) for k,v in self.replay_buffer]
        return np.sum(returns)/len(returns)


    def add_sample(self, sample):
        """Add sample to buffer, assign id"""
        print("add sample",self.buffersize)
        self.replay_buffer[self.buffersize]=sample
        self.buffersize+=1

    def get_losses(self):
        """Get all losses in buffer"""
        keys_losses = [(k, self.replay_buffer[k]["loss"]) for k in self.replay_buffer.keys()]
        return keys_losses

    def replace_entry(self,id,sample):
        self.replay_buffer[id]=sample

    def consider_adding_sample(self, sample: dict):
        """ Show sample to buffer; Buffer decides whether to add it or not based on sample loss;

        Sample must at least contain the key 'loss'; For usage with RUDDER example code, see LessonReplayBuffer class
        docstring;
        """
        if self.buffersize < self.max_buffersize:
            # Add sample if buffer is not full
            self.add_sample(sample)
        else:
            # Replace sample with lowest loss in buffer if new sample loss is higher
            keys_losses = self.get_losses()
            low_sample_ind = np.argmin([b[1] for b in keys_losses])
            if sample['loss'] > keys_losses[low_sample_ind][1]:
                id =keys_losses[low_sample_ind][0]
                print("replace {} with {}".format(self.replay_buffer[id]["loss"],sample["loss"]))
                self.replace_entry(id,sample)

    def softmax(self, x):
        e_logits = np.exp((x-np.max(x))/self.temperature)
        return  e_logits/ e_logits.sum()

    def get_sample(self):
        """Randomly sample episode or game with from buffer; Sampling probabilities are softmax values of losses in
        buffer;"""
        keys_losses = self.get_losses()
        losses = [b[1] for b in keys_losses]
        ids = [b[0] for b in keys_losses]
        losses=np.array(losses,dtype=float)
        probs = self.softmax(losses)
        sample_id = self.rnd_gen.choice(ids, p=probs)
        sample = self.replay_buffer[sample_id]
        sample['id'] = sample_id
        return sample

    def update_sample_loss(self, loss, id):
        """Update loss of sample id"""
        self.replay_buffer[id]['loss'] = loss