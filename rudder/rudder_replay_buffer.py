class RudderReplayBuffer:
    def __init__(self,nr_procs,frames_per_proc):
        self.nr_procs = nr_procs
        self.frames_per_proc = frames_per_proc

    def store_sequence_chunk(self):


