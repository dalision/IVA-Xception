#encoding:UTF-8

import os
import time
import numpy as np
import cv2
from sklearn.utils import shuffle
import json
import config as cfg
from utils import audio
from utils import log

######################## CONFIG #########################
RANDOM = cfg.getRandomState()

######################### SPEC ##########################
def getSpecs(path):
    
    specs = []
    noise = []
    # Get mel-specs for file
    for spec in audio.specsFromFile(path,
                                    rate=cfg.SAMPLE_RATE,
                                    seconds=cfg.SPEC_LENGTH,
                                    overlap=cfg.SPEC_OVERLAP,
                                    minlen=cfg.SPEC_MINLEN,
                                    fmin=cfg.SPEC_FMIN,
                                    fmax=cfg.SPEC_FMAX,
                                    spec_type=cfg.SPEC_TYPE,
                                    shape=(cfg.IM_SIZE[1], cfg.IM_SIZE[0])):

    
        # Determine signal to noise ratio
        s2n = audio.signal2noise(spec)
        #rint(s2n) #control s2n
        specs.append(spec)
        noise.append(s2n)
    # Shuffle arrays (we want to select randomly later)
    specs, noise = shuffle(specs, noise, random_state=RANDOM)

    return specs, noise

def parseDataset():
   
    filedata = {}
    with open(r"flockfiles.json",'r') as f:
        filedata = json.load(f)

    # List of classes, subfolders as class names
    CLASSES = [c for c in sorted(os.listdir(r""))]
    CLASSES = CLASSES[28:]
    
    # Parse every class
    for c in CLASSES:
        log.i(c) 
        index = []
        # List all audio files
        print(c)
        afiles = [f for f in sorted(os.listdir(os.path.join(r"", c)))]
        print(len(afiles))
        # Calculate maximum specs per file
        for i in range(len(afiles)):
            if (afiles[i][:-4]+'.WAV' in filedata[c]):
                index.append(i)
                print("del one")
        index.reverse()
        for i in index:            
            del afiles[i]
        print(len(afiles))
                
        max_specs = cfg.MAX_SPECS_PER_CLASS // len(afiles) + 1

        # Get specs for every audio file
        for i in range(len(afiles)):

            spec_cnt = 0

            try:

                # Stats
                print (i + 1, '/', len(afiles), c, afiles[i])

                # Get specs and signal to noise ratios
                specs, noise = getSpecs(os.path.join(r"", c, afiles[i]))
                # Save specs if it contains signal
                for s in range(len(specs)):
                    # NaN?
                    if np.isnan(noise[s]):
                        noise[s] = 0.0

                    # Above SIGNAL_THRESHOLD?
                    if noise[s] >= cfg.SPEC_SIGNAL_THRESHOLD:
                        # Create target path for accepted specs
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
                        # Filename contains s2n-ratio
                        filename = str(int(noise[s] * 10000)).zfill(4) + '_' + afiles[i].split('.')[0] + '_' + str(s).zfill(3)
                        # Write to HDD
                        cv2.imwrite(os.path.join(filepath, filename + '.png'), specs[s] * 255.0)                        
                    # Do we have enough specs already?
                    if spec_cnt >= max_specs:
                        break

               

            except:
                log.e((spec_cnt, 'specs', 'ERROR DURING SPEC EXTRACT'))
                continue



if __name__ == '__main__':
    parseDataset()
