# refer to Stefan Kahl, 2018, Chemnitz University of Technology https://github.com/kahst/BirdCLEF-Baseline
import os
import time
import numpy as np
import cv2
from sklearn.utils import shuffle
import json
import config as cfg
from utils import audio
from utils import log


RANDOM = cfg.getRandomState()

def getSpecs(path):
    
    specs = []
    noise = []
    for spec in audio.specsFromFile(path,
                                    rate=cfg.SAMPLE_RATE,
                                    seconds=cfg.SPEC_LENGTH,
                                    overlap=cfg.SPEC_OVERLAP,
                                    minlen=cfg.SPEC_MINLEN,
                                    fmin=cfg.SPEC_FMIN,
                                    fmax=cfg.SPEC_FMAX,
                                    spec_type=cfg.SPEC_TYPE,
                                    shape=(cfg.IM_SIZE[1], cfg.IM_SIZE[0])):


        s2n = audio.signal2noise(spec)
        specs.append(spec)
        noise.append(s2n)
    specs, noise = shuffle(specs, noise, random_state=RANDOM)

    return specs, noise

def parseDataset():


    CLASSES = [c for c in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'train')))]
    
    for c in CLASSES:
        log.i(c) 
        index = []
        afiles = [f for f in sorted(os.listdir(os.path.join(cfg.TRAINSET_PATH, 'train', c)))]
        max_specs = cfg.MAX_SPECS_PER_CLASS // len(afiles) + 1
        for i in range(len(afiles)):

            spec_cnt = 0
            try:
                print (i + 1, '/', len(afiles), c, afiles[i])
                specs, noise = getSpecs(os.path.join(cfg.TRAINSET_PATH, 'train', c, afiles[i]))
                for s in range(len(specs)):
                    if np.isnan(noise[s]):
                        noise[s] = 0.0          
                    if noise[s] >= cfg.SPEC_SIGNAL_THRESHOLD:
                        filepath = os.path.join(cfg.DATASET_PATH, c)
                        if not os.path.exists(filepath):
                            os.makedirs(filepath)
                    elif noise[s] <= cfg.NOISE_THRESHOLD:
                        if RANDOM.choice([True, False], p=[0.60, 0.40]):
                            filepath = os.path.join(cfg.NOISE_PATH)
                            if not os.path.exists(filepath):
                                os.makedirs(filepath) 
                        else:
                            filepath = None                    
                    
                    else:
                        filepath = None
                    if filepath:                    
                        filename = str(int(noise[s] * 10000)).zfill(4) + '_' + afiles[i].split('.')[0] + '_' + str(s).zfill(3)
                        # Write to HDD
                        cv2.imwrite(os.path.join(filepath, filename + '.png'), specs[s] * 255.0)                                        
                    if spec_cnt >= max_specs:
                        break
            except:
                log.e((spec_cnt, 'specs', 'ERROR DURING SPEC EXTRACT'))
                continue

if __name__ == '__main__':
    parseDataset()
