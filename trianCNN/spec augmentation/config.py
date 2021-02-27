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
TRAINSET_PATH = r''
DATASET_PATH = r""
NOISE_PATH = r""
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

# Resize mode, options are:
# 'crop': Crops from center of the image
# 'cropRandom': Crops from random position
# 'squeeze': Ignores aspect ratio when resizing
# 'fill': Fills with random noise to keep aspect ratio
RESIZE_MODE = 'squeeze'

# Normalization mode (values between -1 and 1)
ZERO_CENTERED_NORMALIZATION = True

# List of rejected specs, which we want to use as noise samples during augmentation
if os.path.exists(NOISE_PATH):
    NOISE_SAMPLES = [os.path.join(NOISE_PATH, s) for s in os.listdir(NOISE_PATH)]
else:
    NOISE_SAMPLES = []



IM_AUGMENTATION = {'roll_h':0.5,                   # Horizontal roll
                   'noise':0.05,                   # Gaussian noise
                   'brightness':0.15,              # Adjust brightness
                   'dropout':0.25,                 # Dropout single pixels
                   'blur':3,                       # Image blur
                   'multiply':0.25,                # Multiply pixel values
                   'pitch_shift':1, 
                   'cut_vertical':1,
                   'cut_horizon':1,
                   'fun_color':0,
                   'fun_Contrast':0,
                   'fun_Sharpness':0,
                   'fun_bright':0,
                   'pitch_shift':1
                  }

# Maximum number of random augmentations per image
# Each try has 50% chance of success; we do not use duplicate augmentations
AUGMENTATION_COUNT = 2

# Probability for image augmentation
AUGMENTATION_PROBABILITY = 0.5
