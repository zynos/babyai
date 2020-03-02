import torch
from torch import nn

class Net(nn.Module):
    def __init__(self,image_dim,obs_space,instr_dim,ac_embed_dim,action_space):
        super(Net, self).__init__()
        self.action_space=action_space.n
        self.image_dim=image_dim
        self.instr_dim=instr_dim
        self.ac_embed_dim=ac_embed_dim
        self.combined_input_dim=action_space.n+ac_embed_dim+instr_dim+image_dim
        self.rudder_lstm_out=128
        self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)


        self.instr_rnn = nn.GRU(
            self.instr_dim, self.instr_dim,
            batch_first=True,
            bidirectional=False)


        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=image_dim, kernel_size=(2, 2)),
            nn.ReLU()
        )
        self.lstm=nn.LSTM(self.combined_input_dim,self.rudder_lstm_out,batch_first=True)

    def extract_dict_values(self,dic):
        image = dic["images"].transpose(0,2).unsqueeze(0)
        instruction = dic["instructions"].unsqueeze(0)
        action=dic["actions"].unsqueeze(0)
        embedding=dic["embeddings"].unsqueeze(0).unsqueeze(0)
        return image,instruction,action,embedding

    def forward(self, dic):
        image, instruction, action, embedding=self.extract_dict_values(dic)

        image=self.image_conv(image).squeeze(2).squeeze(2).unsqueeze(0)
        instruction=self.instr_rnn(self.word_embedding(instruction))[1][-1].unsqueeze(0)
        action=torch.nn.functional.one_hot(action, num_classes=self.action_space).float().unsqueeze(0)
        x=torch.cat([image,instruction,action,embedding],dim=-1)

        x,hidden=self.lstm(x)


        return x,hidden