import os
import cv2
import numpy as np
import random


def get_frames(filename,n_frames=1):
    frames=[]
    v_cap=cv2.VideoCapture(filename)
    v_len=int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list=np.linspace(0,v_len-1,n_frames+1,dtype=np.int16)
    for fn in range(v_len):
        success, frame=v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    v_cap.release()
    return frames,v_len

def store_frames(frames,path2store):
    for ii,frame in enumerate(frames):
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        path2img=os.path.join(path2store,"frame"+str(ii)+".jpg")
        cv2.imwrite(path2img, frame)


path2data="D:\\Datasets"
sub_folder="hmdb51_org"
sub_folder_jpg="hmdb51_jpg_lite"
extension=".avi"
n_frames=16
train_split = 0.8
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