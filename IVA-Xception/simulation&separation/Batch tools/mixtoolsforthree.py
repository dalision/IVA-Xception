from itertools import combinations
import  random
import os
import numpy as np
from scipy.io import wavfile 
import random 
random.seed(11)

def listfile(path):
    name = {}
    for dirs in os.listdir(path):
        fpath = os.path.join(path,dirs)
        name[dirs]=list(os.listdir(fpath))
    return name      

def read_3(path1,path2,path3):
    rate1, data1 = wavfile.read(path1)
    rate2, data2 = wavfile.read(path2)
    rate3, data3 = wavfile.read(path3)
    return data1,data2,data3

import os 
opath = r""
root = r""
dic = listfile(root)
classes = list(os.listdir(root))
for cs in combinations(classes,3):#c53
    dname = cs[0]+"+"+cs[1]+"+"+cs[2]
    dirname = os.path.join(opath,dname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for i in range(50):
        mix = []
        source = []
        for con in cs:
            path = os.path.join(root,con,"new_"+str(i)+".wav")
            mix.append(path)
            source.append(path)
        odpath = os.path.join(dirname,str(i),"mixed")
        odpath1 = os.path.join(dirname,str(i),"original")
        odpath2 = os.path.join(dirname,str(i),"separation")
        if not os.path.exists(odpath):
            os.makedirs(odpath)
        if not os.path.exists(odpath1):
            os.makedirs(odpath1)
        if not os.path.exists(odpath2):
            os.makedirs(odpath2)
        three = read_3(source[0],source[1],source[2])
        for n in range(3):
            ofpath1 = os.path.join(odpath1,str(n)+".wav")
            wavfile.write(ofpath1,44100,three[n])
