import numpy as np
class preReplayBuffer():
    def __init__(self,nr_procs,dict_fields: list):
        self.replay_buffer_dict=dict()
        self.total_episodes=0
        self.dict_fields = dict_fields
        self.send_to_rudder=[]

        for process_id in range(nr_procs):
            self.init_dict(process_id)


    def init_dict(self,process_id):
        self.replay_buffer_dict[process_id] = dict()
        for field in self.dict_fields:
            self.replay_buffer_dict[process_id][field] = []



    def add_timestep_data(self,sample,process_id):
        # sample keys: rewards, ims, instrs, acts, dones etc.
        assert np.sum(self.replay_buffer_dict[process_id]["reward"]) < 20
        tmp = self.replay_buffer_dict[process_id]
        if tmp["reward"] and tmp["reward"][0] > 0:

            print("shits go down")
        if sample["timestep"]==0 and sample["reward"][process_id]>0:
            print("begin")
        if sample["reward"][process_id] > 0 and sample["done"][process_id] == False:
            print("WTF lolol")
        for key, value in sample.items():
            if key=="timestep":
                self.replay_buffer_dict[process_id][key].append(value)
            else:
                self.replay_buffer_dict[process_id][key].append(value[process_id])
        tmp =self.replay_buffer_dict[process_id]
        if tmp["reward"][0]>0:
            print("shits go down")
        assert np.sum(self.replay_buffer_dict[process_id]["reward"]) < 20
        if sample["done"][process_id]==True:
            self.send_to_rudder.append(self.replay_buffer_dict[process_id])

            self.init_dict(process_id)
        return tmp


