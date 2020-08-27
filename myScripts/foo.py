import pickle
import torch
import os
import pickle
import blosc

def create_subset():
    file="../scripts/demos/shuffled_train0.pkl"
    with open(file,"rb") as f:
        episodes = pickle.load(f)
        episodes = episodes[10000:12000]

    with open("../scripts/demos/10KvalidSet.pkl","wb") as f2:
        pickle.dump(episodes,f2)
#
# def lower_protocol():
#     file = "../scripts/demos/only42Episodes.pkl"
#     with open(file, "rb") as f:
#         episodes = pickle.load(f)
#
#     with open("../scripts/demos/Prot4_only42Episodes.pkl","wb") as f2:
#         pickle.dump(episodes,f2,protocol=4)
#
# create_subset()
# # lower_protocol()
# file = "../scripts/demos/train/Prot4only42Episodes.pkl"
# with open(file, "rb") as f:
#     episodes = pickle.load(f)

#
# print("d")
# zer=torch.tensor(list(range(7)),dtype=torch.int64)
# z2=torch.zeros_like(zer)
# zer=torch.stack([zer,z2])
# # zer[-1]=1
# t=torch.nn.functional.one_hot(zer,num_classes=7)
# print(t)
#
# def concat_to_bigger_pkl():
#
#     # get all files from folder
#     path="../scripts/demos/1mDS/train/"
#     files=os.listdir(path)
#     files.sort()
#     print(files)
#     # get first 4
#     big_list=[]
#     for f in files[:4]:
#         with open(path+f,"rb") as myFile:
#             lis=pickle.load(myFile)
#             # concat
#             big_list.extend(lis)
#
#
#     #save
#     with open("myConcat320K.pkl","wb") as myFile:
#         pickle.dump(big_list,myFile)
from babyai.utils.demos import transform_demos


def demos_to_pickle_protocol_4(path,file_name):
    print("you have to change pack_array in blosc to prot 4 also!")
    with open(path+file_name,"rb") as f:
        demos=pickle.load(f)
    lower_prot_demos=[]
    for demo in demos:
        mission = demo[0]
        all_images = demo[1]
        directions = demo[2]
        actions = demo[3]

        all_images = blosc.unpack_array(all_images)
        all_images=blosc.pack_array(all_images)
        lower_prot_demos.append((mission,all_images,directions,actions))

    with open(path+"prot4_"+file_name, "wb") as f:
        pickle.dump(lower_prot_demos,f,protocol=4)



# def demo_folder_to_lower_protocol(path):
#     files= os.listdir(path)
#     for f in files:
#         demos_to_pickle_protocol_4(path,f)
# create_subset()
# demo_folder_to_lower_protocol("/home/nick/Downloads/trainset1M/")

def scale_rewards_minus_one_to_1(rewards):
    a=-1
    b=1
    return a+((rewards-rewards.min())*(b-a))/\
           (rewards.max()-rewards.min())
#
# rew=torch.zeros(10)
# rew[-1]=1
# print(scale_rewards_minus_one_to_1(rew))

r=torch.randn(100000)
print(r.min())
