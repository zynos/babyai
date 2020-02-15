import numpy as np
class preReplayBuffer():
    def __init__(self,nr_procs,dict_fields: list):
        self.replay_buffer_dict=dict()
        self.replay_buffer_list=[None]*nr_procs
        self.use_list=True
        self.total_episodes=0
        self.dict_fields = dict_fields
        self.send_to_rudder=[]

        for process_id in range(nr_procs):
            self.init_buffer_at(process_id)

    def init_buffer_at(self, process_id):
        if self.use_list:
            self.replay_buffer_list[process_id] = dict()
            for field in self.dict_fields:
                self.replay_buffer_list[process_id][field] = []

        else:
            if process_id in self.replay_buffer_dict.keys():
                del self.replay_buffer_dict[process_id]
            self.replay_buffer_dict[process_id] = dict()
            for field in self.dict_fields:
                self.replay_buffer_dict[process_id][field] = []

    def check_assertions(self,process_id):
        if self.use_list:
            assert np.sum(self.replay_buffer_list[process_id]["reward"]) < 20
        else:
            assert np.sum(self.replay_buffer_dict[process_id]["reward"]) < 20


    def lists_to_tuples(self,process_id):
        for key, value in self.replay_buffer_list[process_id].items():
            if isinstance(value, list):
                self.replay_buffer_list[process_id][key]=tuple(value)


    def add_timestep_data(self,sample,process_id):
        # sample keys: rewards, ims, instrs, acts, dones etc.
        self.check_assertions(process_id)
        for key, value in sample.items():
            if key=="timestep":
                if self.use_list:
                    self.replay_buffer_list[process_id][key].append(value)
                else:
                    self.replay_buffer_dict[process_id][key].append(value)
            else:
                if self.use_list:
                    self.replay_buffer_list[process_id][key].append(value[process_id])
                else:
                    self.replay_buffer_dict[process_id][key].append(value[process_id])
        tmp=self.replay_buffer_list[process_id]
        self.check_assertions(process_id)
        if sample["done"][process_id]==True:
            self.lists_to_tuples(process_id)

            if self.use_list:
                self.send_to_rudder.append(self.replay_buffer_list[process_id])
            else:
                self.send_to_rudder.append(self.replay_buffer_dict[process_id])

            self.init_buffer_at(process_id)
        return tmp


