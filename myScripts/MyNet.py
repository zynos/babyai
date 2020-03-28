import torch
from torch import nn
import torch.nn.functional as F

# from widis_lstm_tools.nn import LSTMLayer

def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
class ExpertControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, image_dim, obs_space, instr_dim, ac_embed_dim, action_space):
        super(Net, self).__init__()
        self.action_space = action_space.n
        self.image_dim = image_dim
        self.instr_dim = instr_dim
        self.ac_embed_dim = ac_embed_dim
        self.rudder_lstm_out = 128
        self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
        self.compressed_embedding = 128
        # self.combined_input_dim = action_space.n + self.compressed_embedding + instr_dim + image_dim
        # embed only
        self.combined_input_dim = action_space.n + ac_embed_dim +instr_dim

        self.embedding_reducer = nn.Linear(ac_embed_dim, self.compressed_embedding)
        self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.linear_out = nn.Linear(self.rudder_lstm_out, 1)
        self.instr_rnn = nn.GRU(
            self.instr_dim, self.instr_dim,
            batch_first=True,
            bidirectional=False)

        num_module = 2
        self.controllers = []
        for ni in range(num_module):
            if ni < num_module - 1:
                mod = ExpertControllerFiLM(
                    in_features=self.instr_dim,
                    out_features=128, in_channels=128, imm_channels=128)
            else:
                mod = ExpertControllerFiLM(
                    in_features=self.instr_dim, out_features=self.image_dim,
                    in_channels=128, imm_channels=128)
            self.controllers.append(mod)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.combined_input_dim, nhead=1)
        self.transformer_input_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.combined_input_dim * 2, nhead=1)
        self.transformer_combined_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc_out_trans = torch.nn.Linear(self.combined_input_dim, 1)

        self.image_conv_old = nn.Sequential(
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
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.lstm = nn.LSTM(self.combined_input_dim, self.rudder_lstm_out, batch_first=True)

        # self.widi_lstm = LSTMLayer(
        #     in_features=self.combined_input_dim, out_features=self.rudder_lstm_out, inputformat='NLC',
        #     # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
        #     w_ci=(torch.nn.init.xavier_normal_, False),
        #     # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
        #     w_ig=(False, torch.nn.init.xavier_normal_),
        #     # output gate: disable all connection (=no forget gate) and disable bias
        #     w_og=False, b_og=False,
        #     # forget gate: disable all connection (=no forget gate) and disable bias
        #     w_fg=False, b_fg=False,
        #     # LSTM output activation is set to identity function
        #     a_out=lambda x: x
        # )

    def extract_process_data(self, dic):
        try:
            image = torch.transpose(torch.stack(dic.images), 1, 3)
            instruction = torch.stack(dic.instructions)
            action = torch.stack(dic.actions)
            embedding = torch.stack(dic.embeddings)
        except:
            image = torch.transpose(dic.images, 1, 3)
            instruction = dic.instructions
            action = dic.actions
            embedding = dic.embeddings

        return image, instruction, action, embedding

    def extract_dict_values(self, dic):
        try:
            image = dic["images"].transpose(0, 2).unsqueeze(0)
            instruction = dic["instructions"].unsqueeze(0)
            action = dic["actions"].unsqueeze(0)
            embedding = dic["embeddings"].unsqueeze(0).unsqueeze(0)
        except:
            image, instruction, action, embedding = self.extract_process_data(dic)
        return image, instruction, action, embedding

    def film_and_action_forward(self,instruction,image,action):
        instruction = self.instr_rnn(self.word_embedding(instruction))[1][-1]
        x = self.image_conv(image)
        for controler in self.controllers:
            x = controler(x, instruction)
        x = F.relu(self.film_pool(x))
        x = x.squeeze(2).squeeze(2)
        action = torch.nn.functional.one_hot(action, num_classes=self.action_space).float()
        return x,action

    def prepare_batch_input(self,image, instruction, action, embedding):
        # image = self.image_conv(image).squeeze(2).squeeze(2)
        # overloaded = torch.cat([instruction, embedding], dim=-1)
        x,action = self.film_and_action_forward()
        # x = torch.cat([image, instruction, action, embedding], dim=-1).unsqueeze(0)
        # embedding only
        x = torch.cat([x, embedding,action], dim=-1).unsqueeze(0)
        return x

    def prepare_input(self, dic, batch, use_transformer):
        # if batch:
        #     image, instruction, action, embedding = self.extract_process_data(dic)
        #     # if use_transformer:
        #     #     image, instruction, action, embedding = self.extract_process_data(dic)
        #     # else:
        #     #     image, instruction, action, embedding = self.extract_dict_values(dic)
        # else:
        #     image, instruction, action, embedding = self.extract_dict_values(dic)
        image, instruction, action, embedding = self.extract_process_data(dic)
        if batch:
            x=self.prepare_batch_input(image, instruction, action, embedding)
        else:
            image = self.image_conv(image).squeeze(2).squeeze(2).unsqueeze(0)
            instruction = self.instr_rnn(self.word_embedding(instruction))[1][-1].unsqueeze(0)
            action = torch.nn.functional.one_hot(action, num_classes=self.action_space).float().unsqueeze(0)
            compressed_embedding = self.embedding_reducer(embedding).unsqueeze(0)
            x = torch.cat([image, instruction, action, compressed_embedding], dim=-1)
        return x

    def create_parallel_input(self,episodes):
        film_out=[]
        actions=[]
        embeddings=[]
        for episode in episodes:
            image, instruction, action, embedding = self.extract_process_data(episode)
            # images.append(image)
            x,action = self.film_and_action_forward(instruction,image,action)
            film_out.append(x)
            # instructions.append(instruction)
            actions.append(action)
            embeddings.append(embedding)
        film_out=torch.nn.utils.rnn.pad_sequence(film_out,True)
        # instructions = torch.nn.utils.rnn.pad_sequence(instructions, True)
        actions = torch.nn.utils.rnn.pad_sequence(actions, True)
        embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, True)

        x = torch.cat([film_out, embeddings, actions], dim=-1)
        # embedding only
        return x



    def forward(self, dic, hidden, batch=False, use_transformer=False):
        if isinstance(dic,list):
            x = self.create_parallel_input(dic)
        else:
            x = self.prepare_input(dic, batch, use_transformer)
        batch_size = x.shape[0]
        # self.init_hidden(batch_size)
        if use_transformer:
            x = x.transpose(0, 1)
            x = self.transformer_input_encoder(x)
            x = x.transpose(0, 1)
            # comb=torch.cat([hidden,new_hidden],dim=-1)

            # out=self.transformer_combined_encoder(comb)

            x = self.fc_out_trans(x.squeeze(1))
            if not x.ndim == 3:
                x = x.unsqueeze(0)
            return x, None
        else:
            if not hidden:
                x, hidden = self.lstm(x)
            else:
                x, hidden = self.lstm(x, hidden)
            x = self.linear_out(x.squeeze(1))
            # if batch:
            #     x = x.squeeze(2)


            # 1 timestep samples are 2d
            if not x.ndim == 3:
                x = x.unsqueeze(0)


            return x, hidden
