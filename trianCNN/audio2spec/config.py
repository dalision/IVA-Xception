# refer to Stefan Kahl, 2018, Chemnitz University of Technology https://github.com/kahst/BirdCLEF-Baseline

import os
import numpy as np

# Fixed random seed
def getRandomState():

    RANDOM_SEED = 1337
    RANDOM = np.random.RandomState(RANDOM_SEED)

    return RANDOM

########################  DATASET  ########################

# Path settings (train audio and xml files, specs, test audio files and json metadata)
# Use 'sort_data.py' to organize the BirdCLEF dataset accordingly
# Extract the BirdCLEF TrainingSet data into TRAINSET_PATH
TRAINSET_PATH = r''#audio path \trianCNNdata(mono)
DATASET_PATH = r""#trianCNNcode_data\trianCNNdata（spec）\data
NOISE_PATH = r""#trianCNNcode_data\trianCNNdata（spec）\data\noise
METADATA_PATH = os.path.join(TRAINSET_PATH, 'metadata')

# Set this path to 'val', 'BirdCLEF2018MonophoneTest' or 'BirdCLEF2018SoundscapesTest' depending on which dataset you want to analyze
TESTSET_PATH = os.path.join(TRAINSET_PATH, 'val')

# Define if you want to 'copy', 'move' or 'symlink' audio files
# If you use 'symlink' make sure your OS does support symbolic links and define TRAINSET_PATH absolute
SORT_MODE = 'copy'

# Maximum number of classes to use (None = no limit)
MAX_CLASSES = None

# Use this whitelist to pre-select species; leave the list empty if you want to include all species
CLASS_WHITELIST = []

# If not sorted, using only a subset of classes (MAX_CLASSES) will select classes randomly
SORT_CLASSES_ALPHABETICALLY = False  

# Specify minimum and maximum amount of samples (specs) per class
MIN_SAMPLES_PER_CLASS = -1   # -1 = no minimum                                      
MAX_SAMPLES_PER_CLASS = None # None = no limit

# Specify the signal-to-noise interval you want to pick samples from (filename contains value)
S2N_INTERVAL = [50, 2500]

# Size of validation split (0.05 = 5%)
VAL_SPLIT = 0.05

######################  SPECTROGRAMS  ######################

# Type of frequency scaling, mel-scale = 'melspec', linear scale = 'linear'
SPEC_TYPE = 'melspec'

# Sample rate for recordings, other sampling rates will force re-sampling
SAMPLE_RATE = 44100

# Specify min and max frequency for low and high pass
SPEC_FMIN = 900
SPEC_FMAX = 15100

# Define length of chunks for spec generation, overlap of chunks and chunk min length
SPEC_LENGTH = 1.0
SPEC_OVERLAP = 0.0
SPEC_MINLEN = 1.0

# Threshold for distinction between noise and signal
SPEC_SIGNAL_THRESHOLD = 0.058
NOISE_THRESHOLD=0.03
#SPEC_SIGNAL_THRESHOLD = 0.0001(origianl)

# Limit the amount of specs per class when extracting spectrograms (None = no limit)
MAX_SPECS_PER_CLASS = 1000

#########################  IMAGE  #########################

# Number of channels
IM_DIM = 1

# Image size (width, height), should be the same as spectrogram shape
IM_SIZE = (299, 299)


RESIZE_MODE = 'squeeze'

# Normalization mode (values between -1 and 1)
ZERO_CENTERED_NORMALIZATION = True




