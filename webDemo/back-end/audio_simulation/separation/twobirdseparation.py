#Edited by Yusheng Dai and Haipeng Zhou 2021 
#School of Cyber Science and Engineering, Sichuan University,
#Chengdu, the People’s Republic of China
import sys
import time
import pyroomacoustics as pra
from .overiva import overiva
import os
from scipy.io import wavfile
import numpy as np
import matplotlib



  
def wav_read_center(wav_list):
    rows = []
    for fn in wav_list:
        fs_loc, data = wavfile.read(fn)# float32
        data = data.astype(np.float64)# float64
        if data.ndim > 1:
            import warnings
            warnings.warn('Discarding extra channels of non-monaural file')
            data = data[:,0]
        if fs_loc!=44100:
            import warnings
            warnings.warn('please check the audios and resample them to 44100')
            exit(1)
        rows.append(data)
    min_len = np.min([d.shape[0] for d in rows])
    output = np.zeros((len(rows), min_len), dtype=rows[0].dtype)
    for r,row in enumerate(rows):
        output[r,:row.shape[0]] = row[:min_len]
    
    return output #nsampleXlength



def seperation(mixfiles,n_sources_target,outputpath):
    np.random.seed(10)
    # STFT parameters
    framesize = 4096
    win_a = pra.hann(framesize)
    win_s = pra.transform.compute_synthesis_window(
        win_a, framesize // 2)
    n_iter = 60  
    dist = "gauss"  # guass or laplace
    #mixshape
    mix = wav_read_center(mixfiles)
    # START BSS
    X_all = pra.transform.analysis(
        mix.T, framesize, framesize // 2, win=win_a
    ).astype(np.complex128)
    X_mics = X_all[:, :, :len(mixfiles)]


    # Run BSS
    Y = overiva(
        X_mics,
        n_src=n_sources_target,
        n_iter=n_iter,
        proj_back=True,
        model=dist,
        init_eig="eig",
        callback=None,
    )

    # Run iSTFT
    if Y.shape[2] == 1:
        y = pra.transform.synthesis(Y[:, :, 0], framesize, framesize // 2, win=win_s)[
            :, None
        ]
        y = y.astype(np.float64)
    else:
        y = pra.transform.synthesis(Y, framesize, framesize // 2, win=win_s).astype(
            np.float64
        )

    # If some of the output are uniformly zero, just add a bit of noise to compare
    for k in range(y.shape[1]):
        if np.sum(np.abs(y[:, k])) < 1e-10:
            y[:, k] = np.random.randn(y.shape[0]) * 1e-10

    import shutil
    shutil.rmtree(outputpath)
    os.mkdir(outputpath)
    ####save mix and separation #######
    for i, sig in enumerate(y.T):
        wavfile.write(
            os.path.join(outputpath,"birdsep{}.wav".format(i + 1)),
            44100,
            pra.normalize(sig, bits=16).astype(np.int16).T)
    print("separation done!")


def two_seperate():
    rootpath = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])))
    mixfilepaths = os.path.join(rootpath, r"seperate_source")
    mixlist = [os.path.join(mixfilepaths, i) for i in os.listdir(mixfilepaths)]
    outputpath = os.path.join(rootpath, r"seperate_result")
    source_num = 2
    seperation(mixlist, source_num, outputpath)



if __name__ == "__main__":
    import sys
    rootpath1 = os.path.join(os.path.dirname(os.path.dirname(sys.argv[0])))
    mixfilepaths = os.path.join(rootpath1,r"twosimu_result")
    mixlist = [os.path.join(mixfilepaths,i) for i in os.listdir(mixfilepaths)] 
    rootpath2 = ("\\").join(sys.argv[0].split("\\")[:-1])
    outputpath = os.path.join(rootpath2, r"two_result")
    source_num = 2
    seperation(mixlist, source_num, outputpath)

    """
    seperation(mixlist,source_num,outputpath)
    mixlist：混合的通道文件被存放在simu_result文件中
    source_num：目标声源数量 three bird情况需要设置为3
    outputpath：默认问当前目录下的two_result目录
    """