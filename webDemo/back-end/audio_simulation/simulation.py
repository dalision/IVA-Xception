#Edited by Yusheng Dai and Haipeng Zhou 2021 
#School of Cyber Science and Engineering, Sichuan University,
#Chengdu, the People’s Republic of China
import sys
import time
import pyroomacoustics as pra
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
    output = output / 2**15
    return output # nsample X length


def createroom(sourcepaths,noises,mic_locs,target_locs,interferer_locs,room_dim,absorption,max_order,n_mics,outputpath,draw):
    #resampling
    fs = 44100
    snr=60
    sinr=10
    print("success run")
    n_sources = len(sourcepaths)+len(noises) 
    n_sources_target = len(sourcepaths)
    n_mics = n_mics 
    
    # micro position 
    mic_locs = np.transpose(mic_locs)
    # target position 
    target_locs = np.transpose(target_locs)
    #interfere position 
    if len(interferer_locs):
        interferer_locs = np.transpose(interferer_locs)
        source_locs = np.concatenate((target_locs, interferer_locs), axis=1)
    else:source_locs = target_locs
    
    wav_files = sourcepaths + noises
    signals = wav_read_center(wav_files)

    # create room
    print(room_dim)
    room = pra.ShoeBox(room_dim, fs=44100, absorption=absorption,
                        max_order=max_order, air_absorption=True, humidity=50)

    # add source
    for sig, loc in zip(signals, source_locs.T):
        room.add_source(loc, signal=sig)

    # add micro
    room.add_microphone_array(
        pra.MicrophoneArray(mic_locs, fs=room.fs))
  
    if draw:
        x = mic_locs[:2][0]
        y = mic_locs[:2][1]
        import matplotlib.pyplot as plt
        plt.scatter(x,y)
        plt.axis('equal')
        plt.xlim([0,20])
        plt.ylim([0,20])
        x1 = source_locs[:2][0]
        y1 = source_locs[:2][1]
        plt.scatter(x1,y1)
        plt.xlim([0,20])
        plt.ylim([0,20])
        plt.axis('equal')
        if len(interferer_locs):
            x1 = interferer_locs[:2][0]
            y1 = interferer_locs[:2][1]
            plt.scatter(x1,y1)
            plt.xlim([0,20])
            plt.ylim([0,20])
            plt.axis('equal')
        # plt.show()

    premix = room.simulate(return_premix=True)
    # power set

    ref_mic=0 
    p_mic_ref = np.std(premix[:, ref_mic, :], axis=1)
    premix /= p_mic_ref[:, None, None]  
    sources_var = np.ones(n_sources_target) # 
    # scale to pre-defined variance
    premix[:n_sources_target, :, :] *= np.sqrt(sources_var[:, None, None])

    # compute noise variance
    sigma_n = np.sqrt(10 ** (-snr / 10) * np.sum(sources_var))

    # now compute the power of interference signal needed to achieve desired SINR
    if n_sources-n_sources_target !=0:
        sigma_i = np.sqrt(
            np.maximum(0, 10 ** (-sinr / 10) * np.sum(np.ones(n_sources_target) ) - sigma_n ** 2)
            / (n_sources-n_sources_target)
        )
        # print(sigma_i)
        premix[n_sources_target:, :, :] *= sigma_i*0.1


    mix = np.sum(premix, axis=0)#nsource X length

    import shutil
    shutil.rmtree(outputpath)
    os.mkdir(outputpath)
    ###float 32 normalization store ####
    for i in range(mix.shape[0]):
        wavfile.write(
                os.path.join(outputpath, "birdmix{}.wav".format(i)),
                room.fs,
                pra.normalize(mix.T[:, i].astype(np.float32)),
            )
    print("Simulation done.")

     
     
if __name__ == "__main__":     
    ######预制的二鸟参数#####
    room_dim = np.array([20, 20, 10])  # room_size
    max_order = 17  #不用改
    absorption = 0.9 #不用改
    n_mics = 3  # mic_number

    noisemodle = True #是否添加干扰点
    if noisemodle:
        interferer_locs = [[18.60466515, 11.67235208, 4.11756127],
                      [18.29351178, 16.5251919, 5.07336436],
                      [18.18467719, 5.00183, 5.38060485]]
        rootpath2 = ("\\").join(sys.argv[0].split("\\")[:-1])
        noise1 = r"sampledatas\noises\0.wav"
        noise2 = r"sampledatas\noises\1.wav"
        noise3 = r"sampledatas\noises\2.wav"
        noises = [os.path.join(rootpath2, i) for i in [noise1, noise2, noise3]]
        print(noises)
    else: interferer_locs, noises =[], []

    mic_locs = [[12.9787868, 10.0212132, 3.5], [12.9787868, 9.9787868, 3.5], [11.9787868, 8.0212132, 3.5]]
    target_locs = [[7, 10, 6], [9, 16, 6]]
    import sys
    rootpath1 =("\\").join(sys.argv[0].split("\\")[:-1])
    outputpath = os.path.join(rootpath1, r"twosimu_result")
    bird1 = r'sampledatas\2birdsamplefiles\1.wav'
    bird2 = r'sampledatas\2birdsamplefiles\2.wav'
    sourcepaths = [os.path.join(rootpath1, i) for i in [bird1, bird2]]
   
    
  
    # # ######预制的三鸟参数#####
    # room_dim = np.array([20, 20, 10])  # room_size
    # max_order = 17  #不用改
    # absorption = 0.9 #不用改
    # n_mics = 4  # mic_number
    # interferer_locs =[]
    # mic_locs = [[13,9.9775 ,3.5 ],[13,9.9925,3.5 ],[13, 10.0075,3.5],[13,10.0225,3.5]]
    # target_locs = [[7,5,6],[7,10,6],[7,15,6]]
    # import sys
    # rootpath = ("\\").join(sys.argv[0].split("\\")[:-1])
    # outputpath = os.path.join(rootpath,r"threesimu_result")
    # bird1 = r'sampledatas\3birdsamplefiles\1.wav'
    # bird2 = r'sampledatas\3birdsamplefiles\2.wav'
    # bird3 = r'sampledatas\3birdsamplefiles\3.wav'
    # sourcepaths = [os.path.join(rootpath,i) for i in [bird1,bird2,bird3]]
    # noises = []



    createroom(sourcepaths,noises,mic_locs,target_locs,interferer_locs,room_dim,absorption,max_order,n_mics,outputpath,True)
    
    '''
    createroom(sourcepaths,noises,mic_locs,target_locs,interferer_locs,room_dim,absorption,max_order,n_mics,outputpath,draw)
    sourcepaths：source singal filepath list
    noises: inferences singal filepath list
    mic_locs: [[x-axis,y-axis,height],[x-axis,y-axis,height]] list of list NX3
    target_locs:同上
    interferer_locs:同上
    room_dim: size of the space
    absorption: 墙面吸收系数
    max_order:墙面最大反射次数
    n_mics: number of n_mics
    outputpath: 文件输出位置
    draw：是否画出俯视图
    '''