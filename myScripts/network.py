import torch
from myScripts.preProcess import PreProcess
from torch import nn
from widis_lstm_tools.nn import LSTMLayer


class Net(torch.nn.Module):
    def __init__(self, instr_embed_dim,ppo_embed_dim, action_dim, n_lstm, image_dim=128,device="gpu",own_net=True):
        super(Net, self).__init__()
        self.device = device
        self.own_net =own_net
        self.preProcess = PreProcess(self.device)
        self.input_length=instr_embed_dim + action_dim+image_dim+ppo_embed_dim

        # This will create an LSTM layer where we will feed the concatenate
        self.lstm1 = LSTMLayer(
            in_features=self.input_length, out_features=n_lstm, inputformat='NLC',
            # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
            w_ci=(torch.nn.init.xavier_normal_, False),
            # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
            w_ig=(False, torch.nn.init.xavier_normal_),
            # output gate: disable all connection (=no forget gate) and disable bias
            # w_og=(False, torch.nn.init.xavier_normal_),
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
        self.image_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=image_dim, kernel_size=(2, 2)),
            nn.ReLU()
        )

        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=image_dim, kernel_size=(2, 2)),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_length, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)


        # After the LSTM layer, we add a fully connected output layer
        self.myLstm1=torch.nn.LSTM(instr_embed_dim + action_dim,n_lstm*2)
        self.myLstm2=torch.nn.LSTM(n_lstm*2,n_lstm)
        self.myGRU=torch.nn.GRU(instr_embed_dim + action_dim,n_lstm*2)
        self.myGRU2 = torch.nn.GRU(n_lstm * 2, n_lstm)
        self.fc_out = torch.nn.Linear(n_lstm, 1)
        self.fc_out_trans = torch.nn.Linear(self.input_length, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-5)

    def forward_own_net2(self, image, instr, actions, embeds, batch):
        all_ims = torch.transpose(self.pre_process_images(image), 0, 1)
        all_instrs = torch.transpose(self.pre_process_instructions(instr), 0, 1)
        actions = torch.stack(actions)
        embeds = torch.stack(embeds).unsqueeze(0)
        one_hot = torch.nn.functional.one_hot(actions, 7).float().unsqueeze(0)
        input = torch.cat([all_ims, all_instrs, one_hot,embeds], dim=-1)
        # lstm_out, *_ = self.lstm1(input,
        #                           return_all_seq_pos=True  # return predictions for all sequence positions
        #                           )
        tr_out=self.transformer_encoder(input)
        net_out=self.fc_out_trans(tr_out)
        # lstm_out, *_ = self.lstm2(lstm_out,
        #                           return_all_seq_pos=True  # return predictions for all sequence positions
        #                           )
        # lstm_out,_=self.myLstm1(input)
        # lstm_out,_ = self.myLstm2(lstm_out)

        # net_out = self.fc_out(lstm_out)
        return net_out
    def pre_process_images(self,image):
        # if len(image)<2:
        #     image=[image]
        all_ims = []
        for idx, i in enumerate(image):
            # it = torch.transpose(torch.transpose(i, 1, 3), 2, 3)
            it = torch.transpose(i,0,2).unsqueeze(0)

            x = self.image_conv(it)
            all_ims.append(x.squeeze(2).squeeze(2))

        all_ims = torch.stack(all_ims)
        return all_ims

    def pre_process_instructions(self,instr):
        all_instrs = []
        for el in instr:
            el=el.unsqueeze(0)
            all_instrs.append(self.preProcess.get_gru_embedding(el))
        all_instrs=torch.stack(all_instrs)
        return all_instrs
    def forward_own_net(self, image, instr, actions,embeds,batch):
        all_ims=torch.transpose(self.pre_process_images(image),0,1)
        all_instrs=torch.transpose(self.pre_process_instructions(instr),0,1)
        actions=torch.stack(actions)
        embeds=torch.stack(embeds).unsqueeze(0)
        one_hot = torch.nn.functional.one_hot(actions, 7).float().unsqueeze(0)
        # x = x.reshape(x.shape[0], -1)
        input = torch.cat([all_ims, all_instrs, one_hot,embeds], dim=-1)
        if batch:
            input=input.transpose(0, 1)
        lstm_out, bla = self.lstm1(input,
                                  return_all_seq_pos=True  # return predictions for all sequence positions
                                  )
        # lstm_out, *_ = self.lstm2(lstm_out,
        #                           return_all_seq_pos=True  # return predictions for all sequence positions
        #                           )
        # lstm_out,_=self.myLstm1(input)
        # lstm_out,_ = self.myLstm2(lstm_out)

        net_out = self.fc_out(lstm_out)
        return net_out

    def forward(self, *argv):
        # if len(self.replay_buffer) > 0:
        #     self.train_one_old_sample_from_replay()

        if self.own_net:
            pred = self.forward_own_net(*argv)
        else:
            pred = self.forward_no_own_net(*argv)
        # with torch.no_grad():
        #     loss, _ = self.lossfunction(pred, self.replay_rewards[-1])
        # self.extend_replay_buffers(argv, loss)
        # print(["{:.3f}".format(p) for p in self.losses_and_mean_dists])
        return pred