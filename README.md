#  Blind source separation-based IVA-Xception model for bird sound recognition in complex acoustic environments

[paper](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ell2.12160)

## Introduction：

Identification of bird species from audio recordings has been a major
area of interest within the field of ecological surveillance and biodi-
versity conservation. Previous studies have successfully identified bird
species from given recordings. However, most of these studies are only
adaptive to low-noise acoustic environments and the cases where each
recording contains only one bird’s sound simultaneously. In reality, bird
audios recorded in the wild often contain overlapping signals, such as
bird dawn chorus, which makes audio feature extraction and accurate
classification extremely difficult. This study is the first to focus on ap-
plying a blind source separation method to identify all foreground bird
species contained in overlapping vocalization recordings. The proposed
IVA-Xception model is based on independent vector analysis and con-
volutional neural network. Experiments on 2020 Bird Sound Recogni-
tion in Complex Acoustic Environments competition (BirdCLEF2020)
dataset show that this model could achieve a higher macro F1-score and
average accuracy compared with state-of-the-art methods.



## Model

![dir_struct](https://github.com/dalision/IVA-Xception/blob/master/images/model.png )

In a complete bird recognition process, the model first uses IVA  in the frequency domain to separate source signals from the original multi-channel signal. Then we utilize the CNN that has been trained to extract features from the converted spectrograms. Finally, the Softmax classifier is adopted to obtain the identification result. For CNN architecture selection, we utilize Xception as the adaptive CNN architecture for our system after comparing neural networks’ performance on spectrogram feature extraction. We also apply data augmentation techniques to converted spectrograms to improve the robustness of the system and solve the data imbalance problem.



# Web Platform

**Recognition page:**

The gif  below showes a process of recognizing all the foreground species from a 3-channel signal , which contains the overlapping vocalization of 3 birds.  

![dir_structure](https://github.com/dalision/IVA-Xception/blob/master/images/recognize.gif )



**Simulation page:**

We apply auralization technology to build a simulation experiment environment for researchers to generate synthesized signals in reality include reverberation and reflection. Users could batch simulate the transmission process of sound source signals, and the system finally outputs multi-channel sound field signals recorded by microphone array set by users.

![dir_structure](https://github.com/dalision/IVA-Xception/blob/master/images/simulation_lab.gif )



## Directory structure：

![dir_structure](https://github.com/dalision/IVA-Xception/blob/master/images/dir_structure.png )
