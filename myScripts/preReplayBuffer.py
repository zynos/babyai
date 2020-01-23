
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
        for key, value in sample.items():
            self.replay_buffer_dict[process_id][key].append(value[process_id])
        if sample["done"][process_id]==True:
            self.send_to_rudder.append(self.replay_buffer_dict[process_id])
            self.init_dict(process_id)


