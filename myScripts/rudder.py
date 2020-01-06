import torch
from torch import nn
from widis_lstm_tools.nn import LSTMLayer
from myScripts.preProcess import PreProcess


class Net(torch.nn.Module):
    def __init__(self, embed_dim, action_dim, n_lstm, image_dim=128,device="cpu"):
        super(Net, self).__init__()
        self.device = device
        self.preProcess = PreProcess(self.device)

        # This will create an LSTM layer where we will feed the concatenate
        self.lstm1 = LSTMLayer(
            in_features=embed_dim + action_dim, out_features=n_lstm, inputformat='NLC',
            # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
            w_ci=(torch.nn.init.xavier_normal_, False),
            # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
            w_ig=(False, torch.nn.init.xavier_normal_),
            # output gate: disable all connection (=no forget gate) and disable bias
            w_og=False, b_og=False,
            # forget gate: disable all connection (=no forget gate) and disable bias
            w_fg=False, b_fg=False,
            # LSTM output activation is set to identity function
            a_out=lambda x: x
        )
        self.lstm2 = LSTMLayer(
            in_features=n_lstm*2, out_features=n_lstm, inputformat='NLC',
            # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
            w_ci=(torch.nn.init.xavier_normal_, False),
            # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
            w_ig=(False, torch.nn.init.xavier_normal_),
            # output gate: disable all connection (=no forget gate) and disable bias
            w_og=False, b_og=False,
            # forget gate: disable all connection (=no forget gate) and disable bias
            w_fg=False, b_fg=False,
            # LSTM output activation is set to identity function
            a_out=lambda x: x
        )
        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=image_dim, kernel_size=(2, 2)),
            nn.ReLU()
        )

        # After the LSTM layer, we add a fully connected output layer
        self.myLstm1=torch.nn.LSTM(embed_dim + action_dim,n_lstm*2)
        self.myLstm2=torch.nn.LSTM(n_lstm*2,n_lstm)
        self.myGRU=torch.nn.GRU(embed_dim + action_dim,n_lstm*2)
        self.myGRU2 = torch.nn.GRU(n_lstm * 2, n_lstm)
        self.fc_out = torch.nn.Linear(n_lstm, 1)

    def forward(self, observations, actions):
        # observations=torch.stack(observations)
        # actions=torch.stack(actions)

        one_hot = torch.nn.functional.one_hot(actions, 7).float()  # size=(4,7,n)
        # Process input sequence by LSTM
        # input=torch.unsqueeze(torch.cat([observations, one_hot],dim=-1),0).cuda()
        input = torch.cat([observations, one_hot], dim=-1)
        lstm_out, *_ = self.lstm1(input,
                                  return_all_seq_pos=True  # return predictions for all sequence positions
                                  )
        # lstm_out, *_ = self.lstm2(lstm_out,
        #                           return_all_seq_pos=True  # return predictions for all sequence positions
        #                           )
        # lstm_out,_=self.myLstm1(input)
        # lstm_out,_ = self.myLstm2(lstm_out)
        net_out = self.fc_out(lstm_out)
        return net_out

    def forward2(self, image, instr, actions):
        all_ims = []
        all_instrs = []

        for idx, i in enumerate(image):
            it = torch.transpose(torch.transpose(i, 1, 3), 2, 3)
            x = self.image_conv(it)
            all_ims.append(x.squeeze())

        all_ims = torch.stack(all_ims)
        all_instrs = self.preProcess.get_gru_embedding(instr)
        all_instrs = all_instrs.unsqueeze(1).repeat(1, 40, 1)
        one_hot = torch.nn.functional.one_hot(actions, 7).float()
        # x = x.reshape(x.shape[0], -1)
        input = torch.cat([all_ims, all_instrs, one_hot], dim=-1)
        lstm_out, *_ = self.lstm1(input,
                                  return_all_seq_pos=True  # return predictions for all sequence positions
                                  )

        net_out = self.fc_out(lstm_out)
        return net_out

        return x
