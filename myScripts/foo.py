# import pickle
# import torch
# import os
# import pickle
# import blosc
# from torch import nn
# import torch.nn.functional as F
#
# # Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
# def initialize_parameters(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         m.weight.data.normal_(0, 1)
#         m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
#         if m.bias is not None:
#             m.bias.data.fill_(0)
#
#
# # Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
# class ExpertControllerFiLM(nn.Module):
#     def __init__(self, in_features, out_features, in_channels, imm_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
#         self.bn1 = nn.BatchNorm2d(imm_channels)
#         self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
#         self.bn2 = nn.BatchNorm2d(out_features)
#
#         self.weight = nn.Linear(in_features, out_features)
#         self.bias = nn.Linear(in_features, out_features)
#
#         self.apply(initialize_parameters)
#
#     def forward(self, x, y):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.conv2(x)
#         out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
#         out = self.bn2(out)
#         out = F.relu(out)
#         return out
#
# def create_subset():
#     file = "../scripts/demos/shuffled_train0.pkl"
#     with open(file, "rb") as f:
#         episodes = pickle.load(f)
#         episodes = episodes[10000:12000]
#
#     with open("../scripts/demos/10KvalidSet.pkl", "wb") as f2:
#         pickle.dump(episodes, f2)
#
#
# #
# # def lower_protocol():
# #     file = "../scripts/demos/only42Episodes.pkl"
# #     with open(file, "rb") as f:
# #         episodes = pickle.load(f)
# #
# #     with open("../scripts/demos/Prot4_only42Episodes.pkl","wb") as f2:
# #         pickle.dump(episodes,f2,protocol=4)
# #
# # create_subset()
# # # lower_protocol()
# # file = "../scripts/demos/train/Prot4only42Episodes.pkl"
# # with open(file, "rb") as f:
# #     episodes = pickle.load(f)
#
# #
# # print("d")
# # zer=torch.tensor(list(range(7)),dtype=torch.int64)
# # z2=torch.zeros_like(zer)
# # zer=torch.stack([zer,z2])
# # # zer[-1]=1
# # t=torch.nn.functional.one_hot(zer,num_classes=7)
# # print(t)
# #
# # def concat_to_bigger_pkl():
# #
# #     # get all files from folder
# #     path="../scripts/demos/1mDS/train/"
# #     files=os.listdir(path)
# #     files.sort()
# #     print(files)
# #     # get first 4
# #     big_list=[]
# #     for f in files[:4]:
# #         with open(path+f,"rb") as myFile:
# #             lis=pickle.load(myFile)
# #             # concat
# #             big_list.extend(lis)
# #
# #
# #     #save
# #     with open("myConcat320K.pkl","wb") as myFile:
# #         pickle.dump(big_list,myFile)
# from babyai.utils.demos import transform_demos
#
#
# def demos_to_pickle_protocol_4(path, file_name):
#     print("you have to change pack_array in blosc to prot 4 also!")
#     with open(path + file_name, "rb") as f:
#         demos = pickle.load(f)
#     lower_prot_demos = []
#     for demo in demos:
#         mission = demo[0]
#         all_images = demo[1]
#         directions = demo[2]
#         actions = demo[3]
#
#         all_images = blosc.unpack_array(all_images)
#         all_images = blosc.pack_array(all_images)
#         lower_prot_demos.append((mission, all_images, directions, actions))
#
#     with open(path + "prot4_" + file_name, "wb") as f:
#         pickle.dump(lower_prot_demos, f, protocol=4)
#
#
# # def demo_folder_to_lower_protocol(path):
# #     files= os.listdir(path)
# #     for f in files:
# #         demos_to_pickle_protocol_4(path,f)
# # create_subset()
# # demo_folder_to_lower_protocol("/home/nick/Downloads/trainset1M/")
#
# #
# # rew=torch.zeros(10)
# # rew[-1]=1
# # print(scale_rewards_minus_one_to_1(rew))
#
# def load_model_for_testing():
#     # Define model
#     class TheModelClass(nn.Module):
#         def __init__(self):
#             super(TheModelClass, self).__init__()
#             self.conv1 = nn.Conv2d(3, 6, 5)
#             self.pool = nn.MaxPool2d(2, 2)
#             self.conv2 = nn.Conv2d(6, 16, 5)
#             self.fc1 = nn.Linear(16 * 5 * 5, 120)
#             self.fc2 = nn.Linear(120, 84)
#             self.fc3 = nn.Linear(84, 10)
#             self.instr_dim = 128
#             self.image_dim = 128
#             self.device = "cpu"
#
#             num_module = 2
#             self.controllers = []
#             for ni in range(num_module):
#                 if ni < num_module - 1:
#                     mod = ExpertControllerFiLM(
#                         in_features=self.instr_dim,
#                         out_features=128, in_channels=128, imm_channels=128)
#                 else:
#                     # output controller
#                     mod = ExpertControllerFiLM(
#                         in_features=self.instr_dim, out_features=self.image_dim,
#                         in_channels=128, imm_channels=128)
#                 self.controllers.append(mod.to(self.device))
#                 self.add_module('FiLM_Controler_' + str(ni), mod)
#
#         def forward(self, x):
#             x = self.pool(F.relu(self.conv1(x)))
#             x = self.pool(F.relu(self.conv2(x)))
#             x = x.view(-1, 16 * 5 * 5)
#             x = F.relu(self.fc1(x))
#             x = F.relu(self.fc2(x))
#             x = self.fc3(x)
#             return x
#
#     # Initialize model
#     model = TheModelClass()
#
#     print("Model's state_dict:")
#     for param_tensor in model.state_dict():
#         print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#
#
#
#
# # load_model_for_testing()
#
# def load_my_model():
#     model = torch.load("stdLSTm_model.pt")
#     model.eval()
#
#     print("Model's state_dict:")
#     for param_tensor in model.state_dict():
#         print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#
#
# load_my_model()

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("/home/nick/PycharmProjects/babyRudder/babyai/rudder/logs/BabyAI-PutNextLocal-v0_IL_expert_filmcnn_gru_seed1_actionIn_True_20-09-18-23-46-58/log.csv")
plt.rcParams.update({'font.size': 18})
plt.plot(df["main_loss"],label="main loss",linewidth=3.0)
plt.plot(df["aux_loss"],label="aux loss",linewidth=3.0)
plt.plot(df["validation_main_loss"],label="validation main loss",linewidth=3.0)
plt.plot(df["validation_aux_loss"],label="validation aux loss",linewidth=3.0)
plt.legend(loc="lower left")
plt.yscale("log")
plt.title("Training on data set 2 - 8 epochs á 10 batches")
plt.ylabel("loss")
plt.xlabel("total batches")
plt.show()

print("")