import os
import cv2
import numpy as np
import random


path2data="D:\\Datasets"
sub_folder="LOLdataset"
sub_folder_train="our485"
sub_folder_eval="eval15"
extension=".jpg"
path2aCatgs=os.path.join(path2data,sub_folder)
listOfCategories=os.listdir(path2aCatgs)
print(listOfCategories,len(listOfCategories))
for cat in listOfCategories:
    print("category:",cat)
    path2aCat=os.path.join(path2aCatgs,cat)
    listOfSubs=os.listdir(path2aCat)
    print("number of sub-folders:",len(listOfSubs))
    print("-"*50)

i = 0
for root,dirs,files in os.walk(path2aCatgs,topdown=False):
    if i < 15:
        i += 1
    else:
        break
    for name in files:
        if extension not in name:
            continue
        path2vid=os.path.join(root,name)
        frames,vlen=get_frames(path2vid,n_frames=n_frames)
        prefix = "test" if random.random() > train_split else "train"
        path2store=path2vid.replace(sub_folder,os.path.join(sub_folder_jpg, prefix))
        path2store=path2store.replace(extension,"")
        print(path2store)
        os.makedirs(path2store, exist_ok=True)
        store_frames(frames,path2store)		