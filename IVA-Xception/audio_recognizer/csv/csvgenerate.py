import os
from progressbar import *
import string
import random
import pandas as pd
from sklearn.model_selection import train_test_split
labels = []
filepaths = []
datapath =  r""
for label in os.listdir(datapath):
    for i in range(50):
        filepath1 = os.path.join(label,str(i),r"separation\0.wav")
        filepath2 = os.path.join(label,str(i),r"separation\1.wav")
        filepath3 = os.path.join(label,str(i),r"separation\2.wav")
        labels.append(label)
        labels.append(label)
        labels.append(label)
        filepaths.append(filepath1)
        filepaths.append(filepath2)
        filepaths.append(filepath3)
# print(labels,filepaths)
dic = {"labels":labels,"filepaths":filepaths}
df = pd.DataFrame(dic)
df.to_csv(r"",index=False)

##mix##
import os
from progressbar import *
import string
import random
import pandas as pd
from sklearn.model_selection import train_test_split
labels = []
filepaths = []
datapath =  r""
for label in os.listdir(datapath):
    for i in range(50):
        filepath1 = os.path.join(label,str(i),r"mixed\mixed.wav")
        labels.append(label)
        filepaths.append(filepath1)
# print(labels,filepaths)
dic = {"labels":labels,"filepaths":filepaths}
df = pd.DataFrame(dic)
df.to_csv(r"",index=False)