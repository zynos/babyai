import torch
from torch import nn
from widis_lstm_tools.nn import LSTMLayer
from myScripts.preProcess import PreProcess


class Net(torch.nn.Module):
    def __init__(self, embed_dim, action_dim, n_lstm, image_dim=128,device="gpu",own_net=True):
        super(Net, self).__init__()
        self.device = device
        self.preProcess = PreProcess(self.device)
        self.own_net = own_net
        self.replay_buffer = []
        self.replay_rewards = []
        self.losses_and_mean_dists=[]
        self.dones=[]


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

        encoder_layer = nn.TransformerEncoderLayer(d_model=128+action_dim, nhead=5)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)


        # After the LSTM layer, we add a fully connected output layer
        self.myLstm1=torch.nn.LSTM(embed_dim + action_dim,n_lstm*2)
        self.myLstm2=torch.nn.LSTM(n_lstm*2,n_lstm)
        self.myGRU=torch.nn.GRU(embed_dim + action_dim,n_lstm*2)
        self.myGRU2 = torch.nn.GRU(n_lstm * 2, n_lstm)
        self.fc_out = torch.nn.Linear(n_lstm, 1)
        self.fc_out_trans = torch.nn.Linear(embed_dim + action_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-5)


    def forward_no_own_net(self, observations, actions):
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
        # transfomer_out = self.transformer_encoder(input)
        # net_out= self.fc_out_trans(transfomer_out)
        net_out = self.fc_out(lstm_out)
        return net_out

    def forward_own_net(self, image, instr, actions):
        all_ims = []
        all_instrs = []

        for idx, i in enumerate(image):
            it = torch.transpose(torch.transpose(i, 1, 3), 2, 3)
            x = self.image_conv(it)
            all_ims.append(x.squeeze())

        all_ims = torch.stack(all_ims)
        all_instrs = self.preProcess.get_gru_embedding(instr)
        all_instrs = all_instrs.unsqueeze(1).repeat(1, all_ims.shape[1], 1)
        one_hot = torch.nn.functional.one_hot(actions, 7).float()
        # x = x.reshape(x.shape[0], -1)
        input = torch.cat([all_ims, all_instrs, one_hot], dim=-1)
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

    def do_optimization(self,pred,i):
        # preds.append(pred)

        loss, (l, aux) = self.lossfunction(pred, self.replay_rewards[i])
        self.rudder_loss = loss.item()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def calc_reward_mean_dist(self):
        current=torch.mean(self.replay_rewards[-1])
        mean = torch.mean(torch.stack(self.replay_rewards[:-1]))
        return torch.abs(mean-current)

    def extend_replay_buffers(self,argv,loss):
        limit=100
        if len(self.replay_buffer)>=limit:
            mean_dist = self.calc_reward_mean_dist()
            score = mean_dist + loss.item()
            minimum=min(self.losses_and_mean_dists)
            if score>minimum:
                idx=self.losses_and_mean_dists.index(minimum)
                self.replay_buffer[idx]=argv
                self.replay_rewards[idx]=self.replay_rewards[-1]
                self.losses_and_mean_dists[idx]=score.item()
            self.replay_rewards=self.replay_rewards[:-1]

        else:
            if len(self.replay_buffer)==0:
                self.losses_and_mean_dists.append(loss.item())
            else:
                self.losses_and_mean_dists.append(loss.item()+self.calc_reward_mean_dist().item())
            self.replay_buffer.append(argv)
        assert len(self.losses_and_mean_dists)<limit+1
        assert len(self.replay_rewards) < limit+1
        assert len(self.replay_buffer) < limit+1

    def sample_from_replay_buffer(self):
        probs=torch.nn.functional.softmax(torch.tensor(self.losses_and_mean_dists))
        m = torch.distributions.Categorical(probs)
        idx = m.sample().item()
        print(idx)
        return self.replay_buffer[idx],idx



    def train_one_old_sample_from_replay(self):
        batch, i = self.sample_from_replay_buffer()
        if self.own_net:
            pred = self.forward_own_net(*batch)
            loss = self.do_optimization(pred, i)
        else:
            pred = self.forward_no_own_net(*batch)
            loss = self.do_optimization(pred, i)

    def forward(self, *argv):
        if len(self.replay_buffer)>0:
            self.train_one_old_sample_from_replay()

        if self.own_net:
            pred = self.forward_own_net(*argv)
        else:
            pred = self.forward_no_own_net(*argv)
        with torch.no_grad():
            loss,_=self.lossfunction(pred,self.replay_rewards[-1])
        self.extend_replay_buffers(argv,loss)
        # print(["{:.3f}".format(p) for p in self.losses_and_mean_dists])
        return pred




    def lossfunction(self,predictions,rewards):
        # from https://github.com/widmi/rudder-a-practical-tutorial/blob/master/tutorial.ipynb
        returns = rewards.sum(dim=1)
        predictions=predictions.squeeze(2)
        # Main task: predicting return at last timestep
        main_loss = torch.mean(predictions[:, -1] - returns) ** 2
        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        aux_loss = torch.mean(predictions[:, :] - returns[..., None]) ** 2
        # Combine losses
        loss = main_loss + aux_loss * 0.5
        return loss,(main_loss,aux_loss)



    def lossfunction2(self,predictions,rewards):
        predictions=predictions.squeeze()
        # print("rews",list(rewards[0][-10:]))
        # print("pred", list(predictions[0][-10:]))
        # main loss: estimate correct end reward
        # we get rewards like [0,0,0,0.9,0,0,0,0,0,0.89 ...]
        idx= rewards>0
        if rewards.max() > 0:
            main=(rewards[idx]-predictions[idx])**2
        else:
            return torch.tensor(0,device=self.device),(torch.tensor(0,device=self.device),torch.tensor(0,device=self.device))
            main=torch.tensor(0,device=self.device)
        aux_rewards=[]
        for r in rewards:
            idx=(r > 0).nonzero()
            new_pred_part=[]
            start=0
            for i in idx:
                lis=r[i].repeat(i - start + 1)
                new_pred_part.append(lis)
                start=i
            if r.max()>0:
                new_pred_part=torch.cat(new_pred_part)
            else:
                new_pred_part=torch.zeros(len(r),device=self.device)
            new_pred_part=torch.cat([new_pred_part,torch.zeros(len(r),device=self.device)])
            aux_rewards.append(new_pred_part[:40])
        aux_rewards=torch.stack(aux_rewards)
        aux=torch.mean((aux_rewards-predictions)**2)
        main=torch.mean(main)
        loss=main+0.5*aux
        return loss,(main,aux)






    def lossfunction1(self,predictions, rewards):
        # rewards = torch.tensor(rewards, device=self.device).reshape(1,-1,1)
        # returns = rewards.sum(dim=1)
        # # Main task: predicting return at last timestep
        # main_loss = torch.mean(predictions[:, -1] - returns) ** 2
        # # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        # aux_loss = torch.mean(predictions[:, :] - returns[..., None]) ** 2
        # # Combine losses
        # loss = main_loss + aux_loss * 0.5
        newPreds = []
        newRews = []
        auxRews = []
        for j, r in enumerate(rewards):
            idx = (r > 0).nonzero()
            start = 0
            for i in idx:
                target = torch.zeros(len(r))
                aux_target = torch.zeros(len(r))
                chunk = predictions[j][start:i + 1]
                target[:len(chunk)] = chunk.squeeze()
                newPreds.append(target)
                target = torch.zeros(len(r))
                aux_chunk = r[i].repeat(i - start + 1)
                aux_target[:len(aux_chunk)] = aux_chunk
                auxRews.append(aux_target)
                chunk = r[start:i + 1]
                target[:len(chunk)] = chunk
                newRews.append(target)
                start = i + 2
                # if start>len(r)-1:
                #     break [0,0,1] [1,1,1] [0,1,1]
        if len(newRews)==0:
            diff= torch.mean((predictions.squeeze()-rewards)**2)
            loss = diff/ len(rewards[0])
            aux = diff
            return loss + 0.5 * aux, (loss, aux)
        newRews = torch.stack(newRews)
        auxRews = torch.stack(auxRews)
        li = newRews > 0
        newPreds = torch.stack(newPreds)
        diff = newRews[li] - newPreds[li]
        # print("predicted rew mean",newPreds.mean().item())

        aux = torch.mean((auxRews - newPreds) ** 2)
        l = torch.mean(diff ** 2)
        return l + 0.5 * aux, (l, aux)
