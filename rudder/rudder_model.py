import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from babyai.rl.utils.supervised_losses import required_heads
from widis_lstm_tools.nn import LSTMCell


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def lambda_replace(x):
    return x


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
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


from abc import abstractmethod, abstractproperty
import torch.nn as nn
import torch.nn.functional as F


# copied these to avoid circular dependency
class MyACModel:
    recurrent = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs):
        pass


class MyRecurrentACModel(MyACModel):
    recurrent = True

    @abstractmethod
    def forward(self, obs, memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass


# copy from babyAI model, slightly modified
class ACModel(nn.Module, MyRecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, instr_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False, arch="cnn1",
                 aux_info=None, add_actions_to_lstm=True, add_actions_to_film=True, use_value=False, use_widi=False,
                 use_endpool=False, use_residual=False, use_visual_embedding=False):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = use_instr
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim
        self.action_space = action_space
        self.add_actions_to_lstm = add_actions_to_lstm
        self.add_actions_to_film = add_actions_to_film
        self.use_value = use_value
        self.use_widi = use_widi
        self.obs_space = obs_space
        self.end_pool = use_endpool
        self.res = use_residual
        self.use_visual_embedding = use_visual_embedding

        if arch == "cnn1":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=image_dim, kernel_size=(2, 2)),
                nn.ReLU()
            )
        elif arch.startswith("expert_filmcnn"):
            if not self.use_instr:
                raise ValueError("FiLM architecture can be used when instructions are enabled")

            if self.end_pool:
                self.image_conv = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                )
            else:
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
            self.film_pool = nn.MaxPool2d(kernel_size=(7, 7) if self.end_pool else (2, 2), stride=2)
        else:
            raise ValueError("Incorrect architecture name: {}".format(arch))

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model in ['gru', 'bigru', 'attgru']:
                self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
                if self.lang_model in ['gru', 'bigru', 'attgru']:
                    gru_dim = self.instr_dim
                    if self.lang_model in ['bigru', 'attgru']:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim, gru_dim, batch_first=True,
                        bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    self.final_instr_dim = self.instr_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList([
                        nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == 'attgru':
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

        # Define memory
        lstm_input_dim = self.image_dim

        if self.use_memory:
            if self.use_visual_embedding:
                # assuming that ppo model is standard
                lstm_input_dim *= 2
            if use_value:
                lstm_input_dim += 1
            if self.add_actions_to_lstm:
                lstm_input_dim += action_space.n

            lstm_cell = LSTMCell(
                n_fwd_features=lstm_input_dim, n_lstm=self.memory_dim,
                # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
                w_ci=(torch.nn.init.xavier_normal_, False),
                # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
                w_ig=(False, torch.nn.init.xavier_normal_),
                # output gate: disable all connection (=no forget gate) and disable bias
                w_og=False, b_og=False,
                # forget gate: disable all connection (=no forget gate) and disable bias
                w_fg=False, b_fg=False,
                # LSTM output activation is set to identity function
                a_out=lambda_replace
            )
            if use_widi:
                self.memory_rnn = lstm_cell
            else:
                self.memory_rnn = nn.LSTMCell(lstm_input_dim, self.memory_dim)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_instr and not "filmcnn" in arch:
            self.embedding_size += self.final_instr_dim

        if arch.startswith("expert_filmcnn"):
            if arch == "expert_filmcnn":
                num_module = 2
            else:
                num_module = int(arch[(arch.rfind('_') + 1):])
            self.controllers = []
            if self.add_actions_to_film:
                embedding_input_dim = self.final_instr_dim + self.action_space.n
            else:
                embedding_input_dim = self.final_instr_dim

            for ni in range(num_module):
                if ni < num_module - 1:
                    mod = ExpertControllerFiLM(
                        in_features=embedding_input_dim,
                        out_features=128, in_channels=128, imm_channels=128)
                else:
                    mod = ExpertControllerFiLM(
                        in_features=embedding_input_dim, out_features=self.image_dim,
                        in_channels=128, imm_channels=128)
                self.controllers.append(mod)
                self.add_module('FiLM_Controler_' + str(ni), mod)

        # Define actor's model
        self.embedding_with_action_size = self.embedding_size + self.action_space.n
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Define critic's model
        self.rudder_critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, instr_embedding=None, visual_embedding=None,actions=None, value=None):
        if self.use_instr and instr_embedding is None:
            instr_embedding = self._get_instr_embedding(obs.instr)
        if self.use_instr and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            mask = (obs.instr != 0).float()
            instr_embedding = instr_embedding[:, :mask.shape[1]]
            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * instr_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            instr_embedding = (instr_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        if self.add_actions_to_film or self.add_actions_to_lstm:
            one_hot_actions = F.one_hot(actions, num_classes=self.action_space.n).float()

        if self.add_actions_to_film:
            instr_embedding = torch.cat([instr_embedding, one_hot_actions], dim=1)

        if self.arch.startswith("expert_filmcnn"):
            x = self.image_conv(x)
            for controler in self.controllers:
                out = controler(x, instr_embedding)
                if self.res:
                    out += x
                x = out
            x = F.relu(self.film_pool(x))
        else:
            x = self.image_conv(x)

        x = x.reshape(x.shape[0], -1)

        if self.add_actions_to_lstm:
            x = torch.cat([x, one_hot_actions], dim=1)
        if self.use_value:
            x = torch.cat([x, value.unsqueeze(1)], dim=1)
        if self.use_visual_embedding:
            x = torch.cat([x, visual_embedding], dim=1)


        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            if self.use_widi:
                hidden = self.memory_rnn(x, hidden[0], hidden[1])
                hidden = (hidden[0].to("cuda"), hidden[1].to("cuda"))
                embedding = hidden[0]
            else:
                hidden = self.memory_rnn(x, hidden)
                hidden = (hidden[1], hidden[0])
                embedding = hidden[1]

            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_instr and not "filmcnn" in self.arch:
            embedding = torch.cat((embedding, instr_embedding), dim=1)

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        x = self.actor(embedding)
        logits = F.log_softmax(x, dim=1)
        dist = Categorical(logits=logits)

        x = self.critic(embedding)
        value = x.squeeze(1)

        # RUDDER
        x = self.rudder_critic(embedding)
        rudder_value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions,
                "embedding": embedding, "rudder_value": rudder_value, "logits": logits}

    def _get_instr_embedding(self, instr):
        if self.lang_model == 'gru':
            _, hidden = self.instr_rnn(self.word_embedding(instr))
            return hidden[-1]

        elif self.lang_model in ['bigru', 'attgru']:
            lengths = (instr != 0).sum(1).long()
            masks = (instr != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if instr.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(instr)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.instr_rnn(inputs)
            else:
                instr = instr[:, 0:lengths[0]]
                outputs, final_states = self.instr_rnn(self.word_embedding(instr))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            if outputs.shape[1] < masks.shape[1]:
                masks = masks[:, :(outputs.shape[1] - masks.shape[1])]
                # the packing truncated the original length
                # so we need to change mask to fit it

            return outputs if self.lang_model == 'attgru' else final_states

        else:
            ValueError("Undefined instruction architecture: {}".format(self.use_instr))
