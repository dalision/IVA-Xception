
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


#about we define the root for the root path ,and the saveroot means whose rootpath is root ;
parser = argparse.ArgumentParser()
#####################################here you should change####################################
parser.add_argument('--root',default ="/home/ubuntu/user_space/model",help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--step', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.01')#learning
parser.add_argument('--momen', type=float, default=0.9, help='momentum, default=0.9')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', default = 7,type=int, help='manual seed')
parser.add_argument('--saveroot',default = 'save',help = 'save path')
parser.add_argument('--csvroot',default = 'smalldata/csv',help = 'csv path')
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
    def __init__(self,csv_path,name,transforms = None):
        self.data_info = pd.read_csv(csv_path,header = None)
        self.transform = transforms
        self.X_train = np.asarray(self.data_info.iloc[:, 1:])
        self.y_train = np.asarray(self.data_info.iloc[:, 0])
        self.name = name 
#####################################here you should change####################################
    def __getitem__(self, index):
        image_name = os.path.join(opt.root,self.X_train[index][0])
        img = Image.open(image_name)
        ##############efficient #################
        image_size =300
        # for efficient-b3
        tfms = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                    transforms.Resize(image_size),transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # tfms = transforms.Compose([transforms.ToTensor()])#for Inception-v3 and xception
        img = tfms(img)
        label = self.y_train[index]
        return (img, label)
    def __len__(self):
        return len(self.data_info.index)


def test(args,model,device,critetion,test_loader,fv,optimizer):
    model.eval()
    print("----------------testing-----------------")
    # print("test_loader:",len(test_loader))
    total_loss = 0.0
    correct = 0
    pbar = ProgressBar().start()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            pbar.update(int((batch_idx / (len(test_loader) - 1)) * 100)) 
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = critetion(output, target)
            epoch_loss = loss
            total_loss += epoch_loss.item()
            pred = output.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()
    pbar.finish()
    total_loss /= len(test_loader.dataset)/args.batchSize
    accuracy = correct / len(test_loader.dataset)
    print('Epoch:{:2d} Loss:{:.4f} Accuracy:{:.6f}'.format(epoch,
        total_loss, correct / len(test_loader.dataset)))
    print('Epoch:{:2d} Loss:{:.4f} Accuracy:{:.6f}'.format(epoch,
        total_loss, correct / len(test_loader.dataset)),file = fv)
    return total_loss,accuracy
    

def train(args,model,device,train_loader,optimizer,critetion,epoch,ft):
    model.train()
    correct = 0
    print("----------------training-----------------")
    # print("train_loader:",len(train_loader))
    total_loss = 0.0
    pbar = ProgressBar().start()
    for batch_idx, (data, target) in enumerate(train_loader):
        pbar.update(int((batch_idx / (len(train_loader) - 1)) * 100))
        data, target = data.to(device), target.to(device)#img.to(device)//nptotensor img.cpu().numpy()//tensor to np //torch.from_numpy().float() np to tensor
        optimizer.zero_grad()#.cpu() .gpu()
        output = model(data)
        loss = critetion(output, target)
        epoch_loss = loss
        total_loss += epoch_loss.item()
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1] 
        correct += pred.eq(target.view_as(pred)).sum().item()
    pbar.finish()
    accuracy = correct / len(train_loader.dataset)
    total_loss /= len(train_loader.dataset)/args.batchSize
    print('Epoch:{:2d} Loss:{:.4f} Accuracy:{:.6f}'.format(epoch,
        total_loss, correct / len(train_loader.dataset)))
    print('Epoch:{:2d} Loss:{:.4f} Accuracy:{:.6f}'.format(epoch,
        total_loss, correct / len(train_loader.dataset)),file = ft)
    return total_loss,accuracy

def save_checkpoint(model,epoch,optimizer,scheduler):
    os.makedirs(os.path.join(opt.root,opt.saveroot), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.num_bad_epochs
    },
        os.path.join(os.path.join(opt.root,opt.saveroot),'{}.pt'.format('best')))




if __name__ == "__main__":
    
    restart =False
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)            # CPU seed
    torch.cuda.manual_seed(opt.manualSeed)       # GPU seed 
    torch.cuda.manual_seed_all(opt.manualSeed) 
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if not restart:
        ft = open(os.path.join(opt.root,opt.saveroot,"t_log.txt"),"w")
        fv = open(os.path.join(opt.root,opt.saveroot,"v_log.txt"),"w")
        
    else:
        ft = open(os.path.join(opt.root,opt.saveroot,"t_log.txt"),"a")
        print("restart",file = ft)
        fv = open(os.path.join(opt.root,opt.saveroot,"v_log.txt"),"a")
        print("restart",file = fv)
    
    train_dataset = mydataset(os.path.join(opt.root,opt.csvroot,"train.csv"),"train")
    val_dataset = mydataset(os.path.join(opt.root,opt.csvroot,"val.csv") ,"val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,num_workers=int(opt.workers),pin_memory=True)

    val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers),pin_memory=True)
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    critetion = nn.CrossEntropyLoss()
    
    #####edit by dilision 2020 test for efficent
    model_name = 'efficientnet-b3'
    model = EfficientNet.from_name(model_name,num_classes=50)
    
    print("model done!")
    if restart:
        checkpoint = torch.load(os.path.join(opt.root,"save/best.pt")) 
        model.load_state_dict(checkpoint['model_state_dict'])
        print("model loaded!")
    model.to(device)
    # summary(model,(3,300,300))
    #######model 2#############################
    # model = xception().to(device)
    #######model 3#############################
    # model = inception_v3().to(device)
    #model.apply(weights_init)#belong to iception-v3
    new_epoch = 0 
    f_h = open(os.path.join(opt.root,opt.saveroot ,"hyperparameter.txt"),"w")
    print(opt,file = f_h)
    f_h.close()
    optimizer = optim.Adam(model.parameters(),lr = opt.lr , betas=(opt.beta1, 0.999),weight_decay=0.0001)
    checkpoint['optim_state_dict']['param_groups'][0]['lr'] = opt.lr
    if restart:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        print("optimizer loader done!")
        new_epoch = checkpoint['epoch']+1
    scheduler = ReduceLROnPlateau(optimizer,'min',patience=3,min_lr= 1e-8,verbose=True)
    if restart:
        scheduler.last_epoch = checkpoint['epoch']+1
        print("scheduler loader done!")

    best_vloss = 1000
    no_improve = 0
    best_accuracy = 0
    if not restart:
      accuracies_t = []
      accuracies_v = []
      t_losses = []
      v_losses = []
      times = []
    else: 
      accuracies_t = list(np.load(os.path.join(opt.root,opt.saveroot,"accuracy_t.npy")))[:new_epoch]
      accuracies_v = list(np.load(os.path.join(opt.root,opt.saveroot,"accuracy_v.npy")))[:new_epoch]
      t_losses = list(np.load(os.path.join(opt.root,opt.saveroot,"t_loss.npy")))[:new_epoch]
      v_losses = list(np.load(os.path.join(opt.root,opt.saveroot,"v_loss.npy")))[:new_epoch]
      times = list(np.load(os.path.join(opt.root,opt.saveroot,"time.npy")))[:new_epoch]
      print("files loaded ")



    for epoch in range(new_epoch,opt.step):
        print(optimizer.param_groups[0]['lr'])
        print('current epoch:{}'.format(epoch))
        t1 = time.time()
        t_loss,accuracy_t = train( opt , model , device , train_loader , optimizer , critetion , epoch,ft)
        t2 = time.time()
        Time = t2-t1
        v_loss,accuracy_v = test(opt , model , device , critetion , val_loader,fv,optimizer) 
        scheduler.step(v_loss)
        if epoch % 2 == 0 and epoch != 0 :
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.num_bad_epochs
           },
            os.path.join(os.path.join(opt.root,opt.saveroot),'{}model.pt'.format(epoch)))
   
        if best_accuracy<accuracy_v:
            best_accuracy = accuracy_v
            save_checkpoint(model,epoch,optimizer,scheduler)
        ft.close()
        fv.close()
        ft = open(os.path.join(opt.root,opt.saveroot,"t_log.txt"),"a")
        fv = open(os.path.join(opt.root,opt.saveroot,"v_log.txt"),"a")

        ###save train\val data
        t_losses.append(t_loss)
        v_losses.append(v_loss)
        accuracies_t.append(accuracy_t)
        accuracies_v.append(accuracy_v)
        times.append(Time)
        np.save(os.path.join(opt.root,opt.saveroot,"time"),np.array(times))
        np.save(os.path.join(opt.root,opt.saveroot,"t_loss"),np.array(t_losses))
        np.save(os.path.join(opt.root,opt.saveroot,"v_loss"),np.array(v_losses))
        np.save(os.path.join(opt.root,opt.saveroot,"accuracy_t"),np.array(accuracies_t))
        np.save(os.path.join(opt.root,opt.saveroot,"accuracy_v"),np.array(accuracies_v))
       


