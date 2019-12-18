from torch import nn


class PreProcess:

    def __init__(self,device):
        self.device=device
        self.obs_space_instr = 100
        self.lang_model = 'gru'
        self.instr_dim=128
        self.word_embedding = nn.Embedding(self.obs_space_instr, self.instr_dim).to(self.device)
        self.final_instr_dim = self.instr_dim
        self.instr_rnn = nn.GRU(
            self.instr_dim, self.instr_dim, batch_first=True,
            bidirectional=(self.lang_model in ['bigru', 'attgru'])).to(self.device)



    def get_gru_embedding(self, instr):
        if self.lang_model == 'gru':
            _, hidden = self.instr_rnn(self.word_embedding(instr))
            return hidden[-1]
    def get_embedding(self,instr):
        return self.word_embedding(instr)

