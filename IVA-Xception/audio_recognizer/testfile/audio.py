#encoding:UTF-8
import os
import sys
import soundfile as sf
import numpy as np
import librosa
from scipy import interpolate
import python_speech_features as psf
import cv2
import scipy.ndimage as ndimage
def changeSampleRate(sig, rate):

    duration = sig.shape[0] / rate

    time_old  = np.linspace(0, duration, sig.shape[0])
    time_new  = np.linspace(0, duration, int(sig.shape[0] * 44100 / rate))

    interpolator = interpolate.interp1d(time_old, sig.T)
    new_audio = interpolator(time_new).T

    sig = np.round(new_audio).astype(sig.dtype)
    
    return sig, 44100

def openAudioFile(path, sample_rate=44100, as_mono=True, mean_substract=False):
    
    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(path, sr=sample_rate, mono=as_mono)
    # Noise reduction?
    if mean_substract:
        sig -= sig.mean()

    return sig, rate


def splitSignal(sig, rate, seconds, overlap, minlen):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]
    
        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, np.zeros((int(rate * seconds) - len(split),))))
        
        sig_splits.append(split)

    return sig_splits

def melspec(sig, rate, shape=(300, 300), fmin=500, fmax=15000, normalize=True, preemphasis=0.95):

    # shape = (height, width) in pixels
  
    # Mel-Spec parameters
    SAMPLE_RATE = rate
    N_FFT = shape[0] * 8 # = window length
    N_MELS = shape[0]
    HOP_LEN = len(sig) // (shape[1] - 1)    
    FMAX = fmax
    FMIN = fmin

    # Preemphasis as in python_speech_features by James Lyons
    if preemphasis:
        sig = np.append(sig[0], sig[1:] - preemphasis * sig[:-1])

    # Librosa mel-spectrum
    melspec = librosa.feature.melspectrogram(y=sig, sr=SAMPLE_RATE, hop_length=HOP_LEN, n_fft=N_FFT, n_mels=N_MELS, fmax=FMAX, fmin=FMIN, power=1.0)
    
    # Convert power spec to dB scale (compute dB relative to peak power)
    melspec = librosa.amplitude_to_db(melspec, ref=np.max, top_db=80)

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    melspec = melspec[::-1, ...]

    # Trim to desired shape if too large
    melspec = melspec[:shape[0], :shape[1]]

    # Normalize values between 0 and 1
    if normalize:
        melspec -= melspec.min()
        if not melspec.max() == 0:
            melspec /= melspec.max()
        else:
            melspec = np.clip(melspec, 0, 1)
    return melspec.astype('float32')



def filter_isolated_cells(array, struct):

    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    
    return filtered_array

def signal2noise(spec):
    spec = spec.copy()
    spec = cv2.medianBlur(spec,3)

    col_median = np.median(spec, axis=0, keepdims=True)
    row_median = np.median(spec, axis=1, keepdims=True)
    spec[spec < row_median*0.9 ] = 0
    spec[spec < col_median*1.3 ] = 0
    spec[spec > 0] = 1  
    spec = cv2.medianBlur(spec,5)
    
 
    spec = filter_isolated_cells(spec, struct=np.ones((3,3)))
    spec_sum = spec.sum()
    s2n = spec_sum / (spec.shape[0]*spec.shape[1])
    return s2n



def specsFromFile(path, rate, seconds, overlap, minlen, shape, start=-1, end=-1, **kwargs):

    # Open file
    sig, rate = openAudioFile(path, rate)

    # Trim signal?
    if start > -1 and end > -1:
        sig = sig[int(start * rate):int(end * rate)]
        minlen = 0

    # Yield all specs for file
    sig_splits = splitSignal(sig, rate, seconds, overlap, minlen)
    # Extract specs for every sig split
    for sig in sig_splits:
        # Get spec for signal chunk
        spec = melspec(sig, rate, shape)

        yield spec
    
if __name__ == '__main__':

    count = 0
    for spec in specsFromFile(r'',
                              rate=44100,
                              seconds=5,
                              overlap=3,
                              minlen=5,
                              shape=(299, 299),
                              fmin=900,
                              fmax=15100,
                              spec_type='melspec'):
        filepath = r''
        cv2.imwrite(os.path.join(filepath, str(count) + '.png'), spec * 255.0)
        count=count+1

        
