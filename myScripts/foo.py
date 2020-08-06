import os
import pickle
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