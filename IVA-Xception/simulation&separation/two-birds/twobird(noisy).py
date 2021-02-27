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

    def createroom(amBird, saBird, noises, mic_p, mic_d, sour_p, sour_d, callback_mix, roomdim, absorption, max_order, n_mics, angle):
        ###########IVA params set################
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
      
        # dB, this is the SNR with respect to a single target source and microphone self-noise
     
    ########separation params##################
        algo = algo_choices[0]
        no_cb = True
        save = True
        n_iter = 60  
        dist = "gauss"  # guass or laplace
    ########paramas set##################
        fs = 44100
        snr=60
        sinr=10
        n_sources = 2+3  
        n_mics = n_mics 
        n_sources_target = 2  
        assert n_sources_target <= n_mics, "More sources than microphones is not supported"

    # set the source powers, the first one is half
        source_std = np.ones(n_sources_target)
    # position
        # room size
        room_dim = roomdim
        # micro position 
        rot = angle
        offset = np.pi-rot/2
        mic_locs = semi_circle_layout(
            mic_p, rot, mic_d, n_mics, rot=offset)  # micro2

        # target position 
        target_locs = np.transpose([[7, 10, 6], [9, 16, 6]])
        
        #interfere position 
        interferer_locs = random_layout([16, 2, 6], 3, offset=[5, 18, 3], seed=1)
        source_locs = np.concatenate((target_locs, interferer_locs), axis=1)
        # source_locs = target_locs
        # audio input 
        wav_files = [amBird, saBird, noises[0],noises[1],noises[2]]
        signals = wav_read_center(wav_files, seed=123)

        # create room 
        room = pra.ShoeBox(room_dim, fs=44100, absorption=absorption,
                           max_order=max_order, air_absorption=True, humidity=50)

        # add source
        for sig, loc in zip(signals, source_locs.T):
            room.add_source(loc, signal=sig)

        # add micro
        room.add_microphone_array(
            pra.MicrophoneArray(mic_locs, fs=room.fs))

        #     #   draw
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
        # x1 = interferer_locs[:2][0]
        # y1 = interferer_locs[:2][1]
        # plt.scatter(x1,y1)
        # plt.xlim([0,20])
        # plt.ylim([0,20])
        # plt.axis('equal')
        # plt.show()
        # callback_mix_kwargs = {
        #     "snr": SNR,
        #     "sir": SIR,
        #     "n_src": n_sources,
        #     "n_tgt": n_sources_target,
        #     "src_std": source_std,
        #     "ref_mic": 0,
        # }


        # power set  
        premix = room.simulate(return_premix=True)
        n_samples = premix.shape[2]
        # Normalize the signals so that they all have unit variance at the reference microphone
        ref_mic=0 
        p_mic_ref = np.std(premix[:, ref_mic, :], axis=1)
        premix /= p_mic_ref[:, None, None]  
        sources_var = np.ones(n_sources_target) # 
        # scale to pre-defined variance
        premix[:n_sources_target, :, :] *= np.sqrt(sources_var[:, None, None])

        # compute noise variance
        sigma_n = np.sqrt(10 ** (-snr / 10) * np.sum(sources_var))

        # now compute the power of interference signal needed to achieve desired SINR
        sigma_i = np.sqrt(
            np.maximum(0, 10 ** (-sinr / 10) * np.sum(sources_var) - sigma_n ** 2)
            / (n_sources-n_sources_target)
        )
        premix[n_sources_target:, :, :] *= sigma_i
 
       
        background = (
            np.sum(premix[n_sources_target:, :, :], axis=0)
        )
         
        # Mix down the recorded signals
        mix = np.sum(premix, axis=0)
        mics_signals = room.mic_array.signals
       
        print("Simulation done.")
        
        # Monitor Convergence
        ref = np.zeros(
        (n_sources_target+1, premix.shape[2], premix.shape[1]), dtype=premix.dtype)
        ref[:n_sources_target, :, :] = premix[:n_sources_target, :, :].swapaxes(1, 2)
        ref[n_sources_target, :, :] = background.T
        convergence_callback = None

        # START BSS

        X_all = pra.transform.analysis(
            mix.T, framesize, framesize // 2, win=win_a
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

        # reorder the vector of reconstructed signals
        y_hat = y[:, perm]
        SDRList.append(sdr)
        SIRList.append(sir)
        print("SDR:", sdr)
        print("SIR:", sir)

####save mix and separation #######
        if save:
            from scipy.io import wavfile
            wavfile.write(
                "birdmix.wav",
                room.fs,
                (pra.normalize(mics_signals, bits=16).astype(
                    np.int16).T)[:, 0],
            )
            for i, sig in enumerate(y_hat.T):
                wavfile.write(
                    "birdsep{}.wav".format(i + 1),
                    room.fs,
                    pra.normalize(sig, bits=16).astype(np.int16).T,
                )

    ###### micro 2#####
    mic_p = [13, 10, 3.5]  # mic_center_point
    mic_d = 0.03  # mic_dinstance
    sour_p = [7, 10, 6]  # source_postion
    sour_d = 5  # source_distance
    roomdim = np.array([20, 20, 10])  # room_size
    n_mics = 2  # mic_number
    max_order = 17  # flection
    absorption = 0.9
    angle = np.pi/2

    SDRList, SIRList, cost_func = [], [], []
    path_amBirds = r''
    path_saBirds = r''
    noises = [r'',
              r'',
              r'']
    path_output = r''
    for item in range(1):
        amBird = path_amBirds + r'\new_{}.wav'.format(item)
        saBird = path_saBirds + r'\new_{}.wav'.format(item)
        # if not os.path.exists(path_output+r'\{}'.format(item)):
        #     os.makedirs(path_output+r'\{}'.format(item))
        createroom(amBird, saBird,noises, mic_p, mic_d, sour_p, sour_d, callback_mix,
                   roomdim, absorption, max_order, n_mics, angle)

    import matplotlib.pyplot as plt
    x1,x2,y1,y2=[],[],[],[]
    fig = plt.figure()
    ax1, ax2 = fig.subplots(2, 1, sharey=True)
    f=open(path_output+r'\2birds_noise.txt','w')
    for i,item in enumerate(SDRList):
        x1.append([i,i])
        x2.append([i,i])
        y1.append([item[0],item[1]])
        y2.append([SIRList[i][0],SIRList[i][1]])
        f.write('============={}=============\n'.format(i))
        f.write('SDR'+str(item)+'\n')
        f.write('SIR'+str(SIRList[i])+'\n')
    f.close()
    for i,item in enumerate(x1):
        ax1.plot(x1[i], y1[i], label='sin')
        ax1.annotate(str(round(y1[i][0],1)),xy=(x1[i][0],y1[i][0]),xytext=(x1[i][0],y1[i][0]),weight='ultralight',fontsize=6)
        ax1.annotate(str(round(y1[i][1],1)),xy=(x1[i][0],y1[i][1]),xytext=(x1[i][0],y1[i][1]),weight='ultralight',fontsize=6)
        ax1.set_title('SDR')
        ax2.plot(x2[i], y2[i], label='sin')
        ax2.annotate(str(round(y2[i][0],1)),xy=(x2[i][0],y2[i][0]),xytext=(x2[i][0],y2[i][0]),weight='ultralight',fontsize=6)
        ax2.annotate(str(round(y2[i][1],1)),xy=(x2[i][0],y2[i][1]),xytext=(x2[i][0],y2[i][1]),weight='ultralight',fontsize=6)
        ax2.set_title('SIR')
    plt.savefig(path_output+r'\2birds_noise.png')
    plt.show()
