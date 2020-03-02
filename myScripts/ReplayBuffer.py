import torch

class ProcessData():
    def __init__(self):
        # input proc_id * feature
        self.actions =[]
        self.rewards =[]
        self.dones =[]
        self.embeddings =[]
        self.images =[]
        self.instructions = []

    def add_single_timestep(self,embedding, action, reward, done, instruction, image):
        self.embeddings.append(embedding)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.images.append(image)
        self.instructions.append(instruction)

    def get_timestep_data(self,timestep):

        return {key:value[timestep] for key, value in self.__dict__.items()
                if not key.startswith('__') and not callable(key)}


    def self_destroy(self):
        del self.actions
        torch.cuda.empty_cache()
        del self.rewards
        torch.cuda.empty_cache()
        del self.dones
        torch.cuda.empty_cache()
        del self.embeddings
        torch.cuda.empty_cache()
        del self.images
        torch.cuda.empty_cache()
        del self.instructions
        torch.cuda.empty_cache()


class ReplayBuffer:
    def __init__(self,nr_procs):
        self.nr_procs=nr_procs
        self.max_size=128
        self.added_episodes=0
        self.proc_data_buffer= [ProcessData() for _ in range(self.nr_procs)]
        self.complete_episodes=[None]*self.nr_procs
    def buffer_full(self):
        return self.added_episodes==self.max_size

    def detach_and_clone(self,*args):
        res=[]
        for a in args:
            try:
                tensor=a.detach().clone()
                res.append(tensor)
            except:
                res.append(a)
        return res

    def add_complete_episodes_to_buffer(self, complete_episodes):
        for ce in complete_episodes:
            if self.added_episodes<self.max_size:
                # print(ce.rewards[-1])
                self.complete_episodes[self.added_episodes]=ce
                self.added_episodes+=1
                # print("added ",self.added_episodes)

            else:
                # print("full")
                self.added_episodes =0
        del complete_episodes


    def add_data_to_proc(self,data_list):
        # data is embeddings,actions,rewards,dones,instructions,images
        complete_episodes=[]
        procs_to_init=[]
        for proc_id in range(self.nr_procs):
            el=[data[proc_id] for data in data_list]
            self.proc_data_buffer[proc_id].add_single_timestep(*el)
            if self.proc_data_buffer[proc_id].dones[-1]==True:
                procs_to_init.append(proc_id)
                complete_episodes.append(self.proc_data_buffer[proc_id])
                # self.proc_data_buffer[proc_id]=ProcessData()
        self.add_complete_episodes_to_buffer(complete_episodes)
        self.init_process_data(procs_to_init)
        del data_list

    def init_process_data(self,procs_to_init):
        for p_id in procs_to_init:
            # self.proc_data_buffer[p_id].self_destroy()
            # self.proc_data_buffer[p_id]=None
            self.proc_data_buffer[p_id] = ProcessData()
            # print("init",p_id)



    def add_timestep_data(self,embeddings,actions,rewards,dones,instructions,images):
        result=self.detach_and_clone(embeddings,actions,rewards,dones,instructions,images)
        self.add_data_to_proc(result)
        del result

