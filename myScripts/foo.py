import pickle
import torch

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
zer=torch.tensor(list(range(7)),dtype=torch.int64)
z2=torch.zeros_like(zer)
zer=torch.stack([zer,z2])
# zer[-1]=1
t=torch.nn.functional.one_hot(zer,num_classes=7)
print(t)
