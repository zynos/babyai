import pickle
import torch
import os
import pickle

# def create_subset():
#     file="../scripts/demos/shuffled_train0.pkl"
#     with open(file,"rb") as f:
#         episodes = pickle.load(f)
#         episodes = episodes[:1000]
#
#     with open("../scripts/demos/1000Episodes.pkl","wb") as f2:
#         pickle.dump(episodes,f2)
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

def concat_to_bigger_pkl():

    # get all files from folder
    path="../scripts/demos/1mDS/train/"
    files=os.listdir(path)
    files.sort()
    print(files)
    # get first 4
    big_list=[]
    for f in files[:4]:
        with open(path+f,"rb") as myFile:
            lis=pickle.load(myFile)
            # concat
            big_list.extend(lis)


    #save
    with open("myConcat320K.pkl","wb") as myFile:
        pickle.dump(big_list,myFile)
