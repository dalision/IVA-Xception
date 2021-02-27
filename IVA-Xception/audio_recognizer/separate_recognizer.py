from testfile import audio
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
import torchvision.utils as utils
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from model import InceptionV3 , xception
import time
import json
import subclass
import os
import numpy as np 
import operator
import sys
import argparse
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from scipy.io import wavfile

def final(model,path,device,s2nthreshold=0.058,threshold=0.3,num_results=3,):
    #generate spec according to audio
    while(1): 
        if(s2nthreshold<0):
            print("the file is noise")
            return -1,-1
        count = 0
        s_b = []
        # print("here global is s2nthreshold{}".format(s2nthreshold))
        for spec in audio.specsFromFile(path,
                              rate=44100,
                              seconds=1,
                              overlap=0,
                              minlen=1,
                              shape=(299, 299),
                              fmin=900,
                              fmax=15100,
                              spec_type='melspec'):
           
            s2n = audio.signal2noise(spec)
            if s2n>s2nthreshold:
                s_b.append(spec)
                count = count + 1 
        if count>0:
            break
        s2nthreshold = s2nthreshold-0.01
#SNR threshold 
    predictions =np.array([])   
    for spec in s_b:
        p = preprobility(spec,model,device)
        if len(predictions):
            predictions = np.vstack([predictions, p])  
        else:
            predictions = p 
    # predictions N(picnum)X50(classmum) 
#choose affirmative predictions
    while(1):
        d = []
        for i in range(predictions.shape[0]):
            if np.max(predictions[i])< threshold:
                # print(np.max(predictions[i]))
                d.append(i)
    predictions = np.delete(predictions,d,0)
    if predictions.shape[0] == 0:
        return None,-1
    p_pool = np.mean(predictions, axis=0)
    #get class labels for predictions
    p_labels = {}
    for i in range(p_pool.shape[0]):
        p_labels[list(CLASS.keys())[i]] = p_pool[i]
    p_sorted =  sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)[:num_results]
    print(p_sorted)
    return p_pool,1

def turn(spec):
    spec = 255*spec#299X299   0-1
    spec = Image.fromarray(spec) 
    tfms = transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    spec = tfms(spec)
    spec = torch.unsqueeze(spec, 0)#1x3x299x299 torch tensor
    return spec

#get the probility of a spec 
def preprobility(spec,model,device):
    spec = turn(spec)
    output = model(spec)
    m = nn.Softmax(dim=1)
    output = m(output)
    p = output.detach().numpy()
    p = p[0][:50]
    p = p[np.newaxis,:]
    return p


if __name__ == "__main__":
    restart = False
    rootpath=r""
    csvpath = r""#csv\seperation.csv
    datapath = r""#overlapping_separation_audio_generator\two_birds\audios
    random.seed(7)
    torch.manual_seed(7)           
    torch.cuda.manual_seed(7)      
    torch.cuda.manual_seed_all(7)
    device = torch.device("cuda:0" if  torch.cuda.is_available()  else "cpu")
    model = xception()
    print("model done!")
    load_model = torch.load(os.path.join(rootpath,"save/best_Xecption.pt"),map_location=torch.device('cpu'))
    model.load_state_dict(load_model["model_state_dict"])
    print("load done")
    model.to(device)
    model.eval()
    CLASS = subclass.CLASS 
    if restart == False:
        pre = []
        labels = []
        save = {"pre":[],"labels":[],"lastindex":-1}
        lastindex = -1
    else:
        with open(r"data/seperationtemp.json",'r') as f:
            save = json.load(f)
            pre = save["pre"]
            labels = save["labels"]
            lastindex = int(save["lastindex"])
    df = pd.read_csv(csvpath)
    for i in range(lastindex+1,int(df.shape[0]/2)):
        df1 = df.iloc[i*2,:]
        df2 = df.iloc[i*2+1,:]
        aname1 = os.path.join(datapath,df1["filepaths"])
        aname2 = os.path.join(datapath,df2["filepaths"])
        label = df1["labels"].split("+")
        print(label)
        label = [CLASS[i] for i in label]
      #start to predict
        tempkey =  np.zeros(50, dtype=float)
        p1,key1 = final(model = model,path = aname1,device = device)
        p2,key2 = final(model = model,path = aname2,device = device)
        tempkey = tempkey+p1
        tempkey = tempkey+p2
        top2 = list(tempkey.argsort()[-2:][::-1])
        print(label)
        print(top2)
        label = list(map(float,label))
        top2 = list(map(float,top2))
        classes = [4,5,9,19,49]
        tempr = top2.copy()
        templ = label.copy()
        match = []
        for p in top2:
            if p in label:
                match.append(p)
                templ.remove(p)
                tempr.remove(p)
        for m in match:
            pre.append(m)
            labels.append(m)
        for k in tempr:
            if k in classes:
                pre.append(k)
            else:pre.append(-1)
        for m in templ:
            labels.append(m)
        save["lastindex"]=str(i)
        save["pre"]=pre
        save["labels"]=labels
        print(save)
        with open('data/seperationtemp.json', 'w') as f:
            json.dump(save, f)

