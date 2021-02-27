# Edited by Yusheng Dai 2020 
#School of Cyber Science and Engineering, Sichuan University,
#Chengdu, the People’s Republic of China
import argparse
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
import torchvision.utils as utils
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from PIL import Image
from model import inception_v3 , Inception3, xception
import time
import pandas as pd
from torch.utils.data.dataset import Dataset
from progressbar import *
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from torchsummary import summary

# progress = ProgressBar()
#about we define the root for the root path ,and the saveroot means whose rootpath is root ;
parser = argparse.ArgumentParser()
parser.add_argument('--root',default ="/home/ubuntu/user_space/model",help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--step', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.01')#learning
parser.add_argument('--momen', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', default = 7,type=int, help='manual seed')
parser.add_argument('--saveroot',default = 'save',help = 'save path')
parser.add_argument('--csvroot',default = 'data/csv',help = 'csv path')
opt = parser.parse_args()

try:
    os.makedirs(os.path.join(opt.root,opt.saveroot))
except OSError:
    pass

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Basic') != -1:
        pass
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
class mydataset(Dataset):
    def __init__(self,csv_path,transforms = None):
        self.data_info = pd.read_csv(csv_path,header = None)
        self.transform = transforms
        self.X_train = np.asarray(self.data_info.iloc[:, 1:])
        self.y_train = np.asarray(self.data_info.iloc[:, 0])

    def __getitem__(self, index):
        image_name = os.path.join(opt.root,self.X_train[index][0])
        img = Image.open(image_name)
        ##############efficient #################
        image_size =300
        
        tfms = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                    transforms.Resize(image_size),transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # tfms = transforms.Compose([transforms.ToTensor()])#for iception and xception
        img = tfms(img)
        label = self.y_train[index]
        return (img, label)
    def __len__(self):
        return len(self.data_info.index)



def test(args,model,device,critetion,test_loader):
    model.eval()
    print("----------------testing-----------------")
    # print("test_loader:",len(test_loader))
    total_loss = 0.0
    correct = 0
    pbar = ProgressBar().start()
    class_correct = list(0. for i in range(50))
    class_total = list(0. for i in range(50))
    preds = []
    tas = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            pbar.update(int((batch_idx / (len(test_loader) - 1)) * 100)) 
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = critetion(output, target)
            epoch_loss = loss
            total_loss += epoch_loss.item()
            pred = output.max(1, keepdim=True)[1]
            tas.extend(target.cpu().numpy().tolist())
            preds.extend(pred.cpu().numpy().reshape(1,-1).tolist()[0]) 
            c = (pred == target).squeeze()
            for i in range(opt.batchSize):
              label = target[i]
              class_correct[label] += c[i][i]
              class_total[label] += 1
              class_correct[label] += c[i][i]
              class_total[label] += 1
            correct += pred.eq(target.view_as(pred)).sum()
    pbar.finish()
  
    total_loss /= len(test_loader.dataset)/args.batchSize
    accuracy = correct / len(test_loader.dataset)
    su = 0
    co = 0
    subac = []
    for i in range(50):
      print('Accuracy of %5s : %.2f %%' % (
        "classes", 100 * class_correct[i] / class_total[i]))
      subac.append((class_correct[i]/class_total[i]).item())
      su = su+class_total[i]
      co = co+class_correct[i]
    ###subaccuracy####
    subdic ={"SubClassAccuracy":subac}
    asubd = pd.DataFrame(subdic)
    print(asubd)
    asubd.to_csv(os.path.join(opt.root,"save/test_subclass.csv"))
    #####
    tas = np.array(tas)
    preds = np.array(preds)
    accu = metrics.accuracy_score(tas,preds)
    pmicro = precision_score(tas, preds, average="micro")
    pmacro = precision_score(tas, preds, average="macro")
    f1_score1 = metrics.f1_score(tas, preds, average='micro')
    f1_score2 = metrics.f1_score(tas, preds, average='macro')
    f1_score3 = metrics.f1_score(tas, preds, average='weighted')
    rmicro = metrics.recall_score(tas, preds, average='micro')
    rmacro = metrics.recall_score(tas, preds, average='macro')
    print('''loss:%.6f\nprecision_score(miro):%.6f\nprecision_score(macro):%.6f
  recall_score(miro):%.6f\nrecall_score(macro):%.6f\nAccurracy:%.6f
  f1-score(micro):%.6f\nf1-score(macro):%.6f\nf1_weight:%.6f\n
        '''%(total_loss,pmicro,pmacro,rmicro,rmacro,accu,f1_score1,f1_score2,f1_score3))
    rec_names = ["id","Loss","Precision(macro)","Precision(micro)","Recall(micro)","Recall(macro)","Accuracy","F1_score(micro)","F1-score(macro)","F1-score(weight)"]
    rec_valuse = [0,total_loss,pmicro,pmacro,rmicro,rmacro,accu,f1_score1,f1_score2,f1_score3]
    find = pd.DataFrame(dict(zip(rec_names,rec_valuse)),index=[0])
    find.to_csv(os.path.join(opt.root,"save/test_values.csv"))
    print(find)
    return total_loss,accuracy
    





if __name__ == "__main__":
    
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)           
    torch.cuda.manual_seed(opt.manualSeed)       
    torch.cuda.manual_seed_all(opt.manualSeed) 
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")    
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    test_dataset = mydataset(os.path.join(opt.root,opt.csvroot,"fold2.csv")  )
    test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers),pin_memory=True)
    critetion = nn.CrossEntropyLoss()


    #############model#########################################
    model_name = 'efficientnet-b3'
    model = EfficientNet.from_name(model_name,num_classes=50).to(device)
    #######model 2#############################
    # model = xception().to(device)
    #######model 3#############################
    # model = inception_v3().to(device)
    #model.apply(weights_init)#belong to iception-v3
    print("model done!")
    load_model = torch.load(os.path.join(opt.root,"save/best.pt"))
    model.load_state_dict(load_model["model_state_dict"])
    model.to(device)
    # summary(model,(3,300,300))
    v_loss,accuracy_v = test(opt , model , device , critetion , test_loader) 
     
     
      
 

       


