import re
import os
import time
import random
import shutil
from scipy.io import wavfile
import numpy as np
rootPath=r""
newPath=r""

    
bird_list=os.listdir(rootPath)
for count in range(50):
    for i in bird_list:
        sour_list=os.listdir(rootPath+"\\"+i)
        r=count
        sour=sour_list[r]
        path1=rootPath+"\\"+i+"\\"+sour 
        index=bird_list.index(i)
        while (index<4):
            index+=1
            sour_list2=os.listdir(rootPath+"\\"+bird_list[index])
            r=count
            sour2=sour_list2[r]
            path2=rootPath+"\\"+bird_list[index]+"\\"+sour2 
            mixdir1=newPath+"\\"+i+"+"+bird_list[index]+"\\"+str(count)+"\\mixed"
            mixdir2=newPath+"\\"+i+"+"+bird_list[index]+"\\"+str(count)+"\\original"
            mixdir3=newPath+"\\"+i+"+"+bird_list[index]+"\\"+str(count)+"\\separation"
            os.makedirs(mixdir1)
            os.makedirs(mixdir2)
            os.makedirs(mixdir3)
            shutil.copy(path1,mixdir2+"\\sourceX.wav")
            shutil.copy(path2,mixdir2+"\\sourceY.wav")

