from . import audio
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
from .model import xception
import time
import json
from . import subclass
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


def final(model, path, device, s2nthreshold=0.058, threshold=0.3, num_results=3, ):
    # generate spec according to audio
    while (1):
        if (s2nthreshold < 0):
            print("the file is noise")
            return -1, -1
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
            if s2n > s2nthreshold:
                s_b.append(spec)
                count = count + 1
        if count > 0:
            break
        s2nthreshold = s2nthreshold - 0.01
    # SNR threshold
    predictions = np.array([])
    for spec in s_b:
        p = preprobility(spec, model, device)
        if len(predictions):
            predictions = np.vstack([predictions, p])
        else:
            predictions = p
            # predictions N(picnum)X50(classmum)
    # choose affirmative predictions
    while (1):
        d = []
        for i in range(predictions.shape[0]):
            if np.max(predictions[i]) < threshold:
                # print(np.max(predictions[i]))
                d.append(i)
        predictions = np.delete(predictions, d, 0)
        if predictions.shape[0] != 0:
            break
        if predictions.shape[0] == 0:
            threshold = threshold - 0.05
    p_pool = np.mean(predictions, axis=0)
    # get class labels for predictions

    return p_pool, 1


def turn(spec):
    spec = 255 * spec  # 299X299   0-1
    spec = Image.fromarray(spec)
    tfms = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    spec = tfms(spec)
    spec = torch.unsqueeze(spec, 0)  # 1x3x299x299 torch tensor
    return spec


# get the probility of a spec
def preprobility(spec, model, device):
    spec = turn(spec)
    output = model(spec)
    m = nn.Softmax(dim=1)
    output = m(output)
    p = output.detach().numpy()
    p = p[0][:50]
    p = p[np.newaxis, :]
    return p


def main(rootpath, datapath, num_results):
    ###model preparation####
    random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = xception()
    print("model done!")
    load_model = torch.load(os.path.join(rootpath, "save/best_Xecption.pt"), map_location=torch.device('cpu'))
    model.load_state_dict(load_model["model_state_dict"])
    print("load done")
    model.to(device)
    model.eval()
    global CLASS
    CLASS = subclass.CLASS
    #####recoginize######
    pre = []
    tempkey = np.zeros(50, dtype=float)
    for filepath in os.listdir(datapath):
        aname = os.path.join(datapath, filepath)
        p, key1 = final(model=model, path=aname, device=device, num_results=num_results)
        tempkey = tempkey + p
    tempkey = tempkey / len(os.listdir(datapath))
    p_labels = {}
    for i in range(tempkey.shape[0]):
        p_labels[list(CLASS.keys())[i]] = tempkey[i]
    p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)    # [:num_results]
    print(p_sorted)
    return p_sorted
    # [('Acrocephalus arundinaceus_Great Reed Warbler', 0.45788547), ('Anthus trivialis_Tree Pipit', 0.24906369)


if __name__ == "__main__":
    ####path setting########
    import sys

    rootpath = ("\\").join(sys.argv[0].split("\\")[:-1])  # the path where stores save dir
    print(rootpath)
    print("path：" + os.path.dirname(sys.argv[0]))
    # datapath = os.path.join(("\\").join(sys.argv[0].split("\\")[:-2]),
    datapath = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])),
                            r"simulation&separation\separation\two_result")  # 2bird
    # datapath = os.path.join(("\\").join(sys.argv[0].split("\\")[:-2]),
    #             r"simulation&separation\separation\three_result")#3bird
    num_results = 2

    main(rootpath, datapath, num_results)

    """
    main(rootpath,datapath,num_results)
    rootpath:audio_recognizer的绝对目录
    datapath: 对这个目录下的所有文件进行识别
    num_results: 规定一次预测返回的前几位
    """
