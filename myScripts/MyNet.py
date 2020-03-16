import torch
from torch import nn

class Net(nn.Module):
    def __init__(self,image_dim,obs_space,instr_dim,ac_embed_dim,action_space):
        super(Net, self).__init__()
        self.action_space=action_space.n
        self.image_dim=image_dim
        self.instr_dim=instr_dim
        self.ac_embed_dim=ac_embed_dim
        self.rudder_lstm_out=128
        self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
        self.compressed_embedding=128
        self.combined_input_dim=action_space.n+self.compressed_embedding+instr_dim+image_dim
        self.embedding_reducer=nn.Linear(ac_embed_dim,self.compressed_embedding)

        self.linear_out=nn.Linear(self.rudder_lstm_out,1)
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


    def extract_dict_values(self,dic,batch):
        if batch:
            image = torch.transpose(dic["images"], 1, 3)
            instruction = dic["instructions"]
            action = dic["actions"]
            embedding = dic["embeddings"]

        else:
            image = dic["images"].transpose(0,2).unsqueeze(0)
            instruction = dic["instructions"].unsqueeze(0)
            action=dic["actions"].unsqueeze(0)
            embedding=dic["embeddings"].unsqueeze(0).unsqueeze(0)
        return image,instruction,action,embedding

    def forward(self, dic,hidden,batch=False):
        image, instruction, action, embedding=self.extract_dict_values(dic,batch)
        if batch:
            image = self.image_conv(image).squeeze(2).squeeze(2)
            instruction = self.instr_rnn(self.word_embedding(instruction))[1][-1]
            action = torch.nn.functional.one_hot(action, num_classes=self.action_space).float()
            x = torch.cat([image, instruction, action, embedding], dim=-1).unsqueeze(0).transpose(0,1)
        else:
            image=self.image_conv(image).squeeze(2).squeeze(2).unsqueeze(0)
            instruction=self.instr_rnn(self.word_embedding(instruction))[1][-1].unsqueeze(0)
            action=torch.nn.functional.one_hot(action, num_classes=self.action_space).float().unsqueeze(0)
            compressed_embedding=self.embedding_reducer(embedding)
            x=torch.cat([image,instruction,action,compressed_embedding],dim=-1)
        batch_size=x.shape[0]
        # self.init_hidden(batch_size)
        if hidden:
            x, hidden = self.lstm(x)
        else:
            x,hidden=self.lstm(x,hidden)
        x=self.linear_out(x.squeeze(1))

        return x,hidden