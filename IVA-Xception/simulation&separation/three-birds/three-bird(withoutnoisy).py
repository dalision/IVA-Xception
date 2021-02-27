#Edited by Yusheng Dai and Haipeng Zhou 2021 
#School of Cyber Science and Engineering, Sichuan University,
#Chengdu, the People’s Republic of China
import sys
import time
import pyroomacoustics as pra
from generate_samples import sampling, wav_read_center
from ive import ogive, ogive_matlab_wrapper
from auxiva_pca import auxiva_pca
from overiva import overiva
from routines import (
    PlaySoundGUI,
    grid_layout,
    semi_circle_layout,
    random_layout,
    gm_layout,)
from mir_eval.separation import bss_eval_sources
import os
from scipy.io import wavfile
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
if __name__ == "__main__":
    algo_choices = [
        "overiva",
        "ilrma",
        "auxiva",
        "auxiva_pca",
        "ogive",
        "ogive_matlab",
    ]
    model_choices = ['laplace', 'gauss']
    init_choices = ['eye', 'eig']
    init = init_choices[0]

    def semi_line_layout(mic_p,angle,mic_d,n_mics):  
        if n_mics%2==0:
            y1=[mic_p[1]+mic_d/2*np.sin(angle) +mic_d*i*np.sin(angle) for i in range(int(n_mics/2))]
            y2=list(reversed([mic_p[1]-mic_d/2*np.sin(angle)-mic_d*i*np.sin(angle) for i in range(int(n_mics/2))]))
            y=y2+y1
            x1=[mic_p[0]+mic_d/2*np.cos(angle) +mic_d*i*np.cos(angle) for i in range(int(n_mics/2))]
            x2=list(reversed([mic_p[0]-mic_d/2*np.cos(angle)-mic_d*i*np.cos(angle) for i in range(int(n_mics/2))]))
            x=x2+x1
            z=[mic_p[2]]*n_mics
            mic_locs=[x,y,z]
        if n_mics%2==1:
            x = np.linspace(mic_p[0]-np.cos(angle)*(mic_d*(n_mics-1)/2),mic_p[0]+np.cos(angle)*(mic_d*(n_mics-1)/2),n_mics)
            y = np.linspace(mic_p[1]-np.sin(angle)*(mic_d*(n_mics-1)/2),mic_p[1]+np.sin(angle)*(mic_d*(n_mics-1)/2),n_mics)
            z = [mic_p[2]]*n_mics
            mic_locs=[x,y,z]
        return np.array(mic_locs)

    def callback_mix(premix, snr=0, sir=0, ref_mic=0, n_src=None, n_tgt=None, src_std=None):

        # first normalize all separate recording to have unit power at microphone one
        p_mic_ref = np.std(premix[:, ref_mic, :], axis=1)
        premix /= p_mic_ref[:, None, None]
        premix[:n_tgt, :, :] *= src_std[:, None, None]

        # compute noise variance
        sigma_n = np.sqrt(10 ** (-snr / 10) * np.mean(src_std ** 2))

        # now compute the power of interference signal needed to achieve desired SIR
        num = 10 ** (-sir / 10) * np.sum(src_std ** 2)
        sigma_i = np.sqrt(num / (n_src - n_tgt))
        premix[n_tgt:n_src, :, :] *= sigma_i

        # Mix down the recorded signals
        mix = np.sum(premix[:n_src, :], axis=0) + sigma_n * np.random.randn(
            *premix.shape[1:]
        )

        return mix


    def line_createroom(Bird1,Bird2,Bird3,callback_mix):

        roomdim = np.array([20, 20, 10]) 
        max_order = 17  
        absorption = 0.9    

        mic_p = [13, 10, 3.5]   #mic_center_point
        mic_d = 0.015      
            #mic_dinstance
        sour_p = [7,10,6]    #source_postion
        sour_d = 5           #source_distance

        n_mics = 4          #mic_number
        n_sources = 3 

        mic_rot = np.pi/2 
        bird_rot = np.pi/2 
    
     
        ### params setting ###
        np.random.seed(10)
        # STFT parameters
        framesize = 4096
        win_a = pra.hann(framesize)
        win_s = pra.transform.compute_synthesis_window(
            win_a, framesize // 2)
        ogive_mu = 0.1
        ogive_update = "switching"
        ogive_iter = 2000
        SIR = 10  # dB
        SNR = (60)
        algo = algo_choices[0]
        no_cb = True
        save = True
        n_iter = 60  
        dist = "gauss"  # laplace
        fs = 44100
        n_sources_target = 3
        assert n_sources_target <= n_mics, "More sources than microphones is not supported"

   
        source_std = np.ones(n_sources_target)
        # room size
        room_dim = roomdim

        # micro position
        mic_locs = semi_line_layout(mic_p,mic_rot,mic_d,n_mics)
     

        # target position 
        source_locs = semi_line_layout(sour_p,bird_rot,sour_d,n_sources)
   
        source_locs[0][0],source_locs[0][2] = source_locs[0][0]+0,source_locs[0][2]+0##push value
   
        # target_locs = np.transpose([[7, 10, 6], [9, 16, 6]])

        # audio loaded
        wav_files = [Bird1,Bird2,Bird3]
     
        signals = wav_read_center(wav_files, seed=123)

        #create room
        room = pra.ShoeBox(room_dim, fs=44100, absorption=absorption,
                            max_order=max_order, air_absorption=True, humidity=50)

        # add source
        for sig, loc in zip(signals, source_locs.T):
            room.add_source(loc, signal=sig)

        # add micro
        room.add_microphone_array(
            pra.MicrophoneArray(mic_locs, fs=room.fs))

        callback_mix_kwargs = {
            "snr": SNR,
            "sir": SIR,
            "n_src": n_sources,
            "n_tgt": n_sources_target,
            "src_std": source_std,
            "ref_mic": 0,
        }
        # # draw
        # x = mic_locs[:2][0]
        # y = mic_locs[:2][1]
        # import matplotlib.pyplot as plt
        # plt.scatter(x,y)
        # plt.axis('equal')
        # plt.xlim([0,20])
        # plt.ylim([0,20])
        # x1 = source_locs[:2][0]
        # y1 = source_locs[:2][1]
        # plt.scatter(x1,y1)
        # plt.xlim([0,20])
        # plt.ylim([0,20])
        # plt.axis('equal')
        # plt.show()

        # Run the simulation
        separate_recordings = room.simulate(
            callback_mix=callback_mix,
            callback_mix_kwargs=callback_mix_kwargs,
            return_premix=True,
        )
        mics_signals = room.mic_array.signals
        print("line Simulation done.")

   

        # Monitor Convergence
        ref = np.moveaxis(separate_recordings, 1, 2)
        if ref.shape[0] < n_mics:
            ref = np.concatenate(
                (ref, np.random.randn(n_mics -
                                        ref.shape[0], ref.shape[1], ref.shape[2])),
                axis=0,
            )

        convergence_callback = None

        X_all = pra.transform.analysis(
            mics_signals.T, framesize, framesize // 2, win=win_a
        ).astype(np.complex128)
        X_mics = X_all[:, :, :n_mics]

        tic = time.perf_counter()

        if algo == "auxiva":
            # Run AuxIVA
            Y = overiva(
                X_mics,
                n_iter=n_iter,
                proj_back=True,
                model=dist,
                callback=convergence_callback,
            )
        elif algo == "auxiva_pca":
            # Run AuxIVA
            Y = auxiva_pca(
                X_mics,
                n_src=n_sources_target,
                n_iter=n_iter,
                proj_back=True,
                model=dist,
                callback=convergence_callback,
            )
        elif algo == "overiva":
            # Run AuxIVA
            Y = overiva(
                X_mics,
                n_src=n_sources_target,
                n_iter=n_iter,
                proj_back=True,
                model=dist,
                init_eig=(init == init_choices[1]),
                callback=convergence_callback,
            )
           
        elif algo == "ilrma":
            # Run AuxIVA
            Y = pra.bss.ilrma(
                X_mics,
                n_iter=n_iter,
                n_components=2,
                proj_back=True,
                callback=convergence_callback,
            )
        elif algo == "ogive":
            # Run OGIVE
            Y = ogive(
                X_mics,
                n_iter=ogive_iter,
                step_size=ogive_mu,
                update=ogive_update,
                proj_back=True,
                model=dist,
                init_eig=(init == init_choices[1]),
                callback=convergence_callback,
            )
        elif algo == "ogive_matlab":
            # Run OGIVE
            Y = ogive_matlab_wrapper(
                X_mics,
                n_iter=ogive_iter,
                step_size=ogive_mu,
                update=ogive_update,
                proj_back=True,
                init_eig=(init == init_choices[1]),
                callback=convergence_callback,
            )
        else:
            raise ValueError("No such algorithm {}".format(algo))

    

    

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

        # For conventional methods of BSS, reorder the signals by decreasing power
        if algo != "blinkiva":
            new_ord = np.argsort(np.std(y, axis=0))[::-1]
            y = y[:, new_ord]

        # Compare SIR
        m = np.minimum(y.shape[0] - framesize // 2, ref.shape[1])
        sdr, sir, sar, perm = bss_eval_sources(
            ref[:n_sources_target, :m, 0],
            y[framesize // 2: m + framesize // 2, :n_sources_target].T,
        )

        # reorder the vector of reconstructed signals
        y_hat = y[:, perm]

        #return 
        mixdata = pra.normalize(mics_signals, bits=16).astype(np.int16).T
        separationdata = []
        for sig in y_hat.T:
            separationdata.append(pra.normalize(sig, bits=16).astype(np.int16).T)
        print("sdr",sdr)        
        return sdr,sir,mixdata,separationdata  #wavefile(mixdata) wavefile(separationdata[0]) wavefile(separationdata[1])

   
    
    def fan_createroom(Bird1,Bird2,Bird3,callback_mix):
        #params settings
        np.random.seed(10)
        # STFT parameters
        framesize = 4096
        win_a = pra.hann(framesize)
        win_s = pra.transform.compute_synthesis_window(
            win_a, framesize // 2)
        # algorithm parameters
        # param ogive
        ogive_mu = 0.1
        ogive_update = "switching"
        ogive_iter = 2000
        SIR = 10  # dB
        SNR = (60)
        roomdim = np.array([20, 20, 10]) 
        max_order = 17  
        absorption = 0.9    
        mic_p = [13, 10, 3.5]  
        mic_d = 0.025 
        sour_p = [13,10,6]
        sour_d = 5 
        n_mics = 4  
        n_sources=3
        mic_rot = np.pi
        bird_rot = np.pi*2/3
        algo = algo_choices[0]
        no_cb = True
        save = True
        n_iter = 60  
        dist = "gauss"  # laplace
        fs = 44100
        n_sources_target = 3
        assert n_sources_target <= n_mics, "More sources than microphones is not supported"
    # set the source powers, the first one is half
        source_std = np.ones(n_sources_target)
 
        #room size
        room_dim = roomdim

        # micro phone position 
        mic_locs = semi_circle_layout(mic_p, mic_rot, mic_d, n_mics, np.pi-mic_rot/2)


        # target sources position 
        source_locs = semi_circle_layout(sour_p,
         bird_rot , sour_d, 3, np.pi-bird_rot/2)

        # audio loaded 
        wav_files = [Bird1,Bird2,Bird3]
     
        signals = wav_read_center(wav_files, seed=123)

        # createroom 
        room = pra.ShoeBox(room_dim, fs=44100, absorption=absorption,
                            max_order=max_order, air_absorption=True, humidity=50)

        # add source
        for sig, loc in zip(signals, source_locs.T):
            room.add_source(loc, signal=sig)

        # add micro
        room.add_microphone_array(
            pra.MicrophoneArray(mic_locs, fs=room.fs))

        callback_mix_kwargs = {
            "snr": SNR,
            "sir": SIR,
            "n_src": n_sources,
            "n_tgt": n_sources_target,
            "src_std": source_std,
            "ref_mic": 0,
        }
        # draw
        # x = mic_locs[:2][0]
        # y = mic_locs[:2][1]
        # import matplotlib.pyplot as plt
        # plt.scatter(x,y)
        # plt.axis('equal')
        # plt.xlim([0,20])
        # plt.ylim([0,20])
        # x1 = source_locs[:2][0]
        # y1 = source_locs[:2][1]
        # plt.scatter(x1,y1)
        # plt.xlim([0,20])
        # plt.ylim([0,20])
        # plt.axis('equal')
        # plt.show()

        # Run the simulation
        separate_recordings = room.simulate(
            callback_mix=callback_mix,
            callback_mix_kwargs=callback_mix_kwargs,
            return_premix=True,
        )
        mics_signals = room.mic_array.signals
        print("fan Simulation done.")



        # Monitor Convergence
        ref = np.moveaxis(separate_recordings, 1, 2)
        if ref.shape[0] < n_mics:
            ref = np.concatenate(
                (ref, np.random.randn(n_mics -
                                        ref.shape[0], ref.shape[1], ref.shape[2])),
                axis=0,
            )

        # SDR, SIR, cost_func = [], [], []
        convergence_callback = None

        # START BSS

        # shape: (n_frames, n_freq, n_mics)
        X_all = pra.transform.analysis(
            mics_signals.T, framesize, framesize // 2, win=win_a
        ).astype(np.complex128)
        X_mics = X_all[:, :, :n_mics]


        # Run BSS
        if algo == "auxiva":
            # Run AuxIVA
            Y = overiva(
                X_mics,
                n_iter=n_iter,
                proj_back=True,
                model=dist,
                callback=convergence_callback,
            )
        elif algo == "auxiva_pca":
            # Run AuxIVA
            Y = auxiva_pca(
                X_mics,
                n_src=n_sources_target,
                n_iter=n_iter,
                proj_back=True,
                model=dist,
                callback=convergence_callback,
            )
        elif algo == "overiva":
            # Run AuxIVA
            Y = overiva(
                X_mics,
                n_src=n_sources_target,
                n_iter=n_iter,
                proj_back=True,
                model=dist,
                init_eig=(init == init_choices[1]),
                callback=convergence_callback,
            )
           
        elif algo == "ilrma":
            # Run AuxIVA
            Y = pra.bss.ilrma(
                X_mics,
                n_iter=n_iter,
                n_components=2,
                proj_back=True,
                callback=convergence_callback,
            )
        elif algo == "ogive":
            # Run OGIVE
            Y = ogive(
                X_mics,
                n_iter=ogive_iter,
                step_size=ogive_mu,
                update=ogive_update,
                proj_back=True,
                model=dist,
                init_eig=(init == init_choices[1]),
                callback=convergence_callback,
            )
        elif algo == "ogive_matlab":
            # Run OGIVE
            Y = ogive_matlab_wrapper(
                X_mics,
                n_iter=ogive_iter,
                step_size=ogive_mu,
                update=ogive_update,
                proj_back=True,
                init_eig=(init == init_choices[1]),
                callback=convergence_callback,
            )
        else:
            raise ValueError("No such algorithm {}".format(algo))

  

    

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

        # For conventional methods of BSS, reorder the signals by decreasing power
        if algo != "blinkiva":
            new_ord = np.argsort(np.std(y, axis=0))[::-1]
            y = y[:, new_ord]

        # Compare SIR
        m = np.minimum(y.shape[0] - framesize // 2, ref.shape[1])
        sdr, sir, sar, perm = bss_eval_sources(
            ref[:n_sources_target, :m, 0],
            y[framesize // 2: m + framesize // 2, :n_sources_target].T,
        )

        y_hat = y[:, perm]
        #return 
        mixdata = pra.normalize(mics_signals, bits=16).astype(np.int16).T
        separationdata = []
        for sig in y_hat.T:
            separationdata.append(pra.normalize(sig, bits=16).astype(np.int16).T)
        print("sdr",sdr)        
        return sdr,sir,mixdata,separationdata  #wavefile(mixdata) wavefile(separationdata[0]) wavefile(separationdata[1])

    SDRList, SIRList, cost_func = [], [], []
    for n in range(1): 
       
        path_Bird1 = r''
        path_Bird2 = r''
        path_Bird3 = r''
        path_output = r''
        for item in range(0,30):
            Bird1 = path_Bird1 + r'\new_{}.wav'.format(item)
            Bird2 = path_Bird2 + r'\new_{}.wav'.format(item)
            Bird3 = path_Bird3 + r'\new_{}.wav'.format(item)
            sdr1,sir1,mixdata1,separationdata1 = line_createroom(Bird1,Bird2,Bird3,callback_mix)
            sdr2,sir2,mixdata2,separationdata2 = fan_createroom(Bird1,Bird2,Bird3,callback_mix)
 
            if np.mean(sdr1)>=np.mean(sdr2):
                finalmix = mixdata1
                finalsep = separationdata1
                SDRList.append(sdr1)
                SIRList.append(sir1)
            if np.mean(sdr1)< np.mean(sdr2):
                finalmix = mixdata2
                finalsep = separationdata2
                SDRList.append(sdr2)
                SIRList.append(sir2)
            for z in range(3):
                wavfile.write("birdsep{}.wav".format(z),44100,finalsep[z])
            wavfile.write("birdmix.wav",44100,finalmix[:,0])
        
    import matplotlib.pyplot as plt
    x1,x2,y1,y2=[],[],[],[]
    fig = plt.figure()
    ax1, ax2 = fig.subplots(2, 1, sharey=True)
    f=open(path_output+r'\2birds_ac_am.txt','w')
    for i,item in enumerate(SDRList):
        x1.append([i,i,i])
        x2.append([i,i,i])
        y1.append([item[0],item[1],item[2]])
        y2.append([SIRList[i][0],SIRList[i][1],SIRList[i][2]])
        f.write('============={}=============\n'.format(i))
        f.write('SDR'+str(item)+'\n')
        f.write('SIR'+str(SIRList[i])+'\n')
    f.close()
    for i,item in enumerate(x1):
        ax1.plot(x1[i], y1[i], label='sin')
        ax1.annotate(str(round(y1[i][0],1)),xy=(x1[i][0],y1[i][0]),xytext=(x1[i][0],y1[i][0]),weight='ultralight',fontsize=6)
        ax1.annotate(str(round(y1[i][1],1)),xy=(x1[i][0],y1[i][1]),xytext=(x1[i][0],y1[i][1]),weight='ultralight',fontsize=6)
        ax1.annotate(str(round(y1[i][2],1)),xy=(x1[i][0],y1[i][2]),xytext=(x1[i][0],y1[i][2]),weight='ultralight',fontsize=6)
        ax1.set_title('SDR')
        ax2.plot(x2[i], y2[i], label='sin')
        ax2.annotate(str(round(y2[i][0],1)),xy=(x2[i][0],y2[i][0]),xytext=(x2[i][0],y2[i][0]),weight='ultralight',fontsize=6)
        ax2.annotate(str(round(y2[i][1],1)),xy=(x2[i][0],y2[i][1]),xytext=(x2[i][0],y2[i][1]),weight='ultralight',fontsize=6)
        ax2.annotate(str(round(y2[i][2],1)),xy=(x2[i][0],y2[i][2]),xytext=(x2[i][0],y2[i][2]),weight='ultralight',fontsize=6)
        ax2.set_title('SIR')
    plt.savefig(path_output+r'\2birds_ac_am.png')
    plt.show()

 
