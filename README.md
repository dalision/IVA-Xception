#  Blind source separation-based IVA-Xception model
for bird sound recognition in complex acoustic
environments

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



# Web Platform

**Recognition page:**

The gif  below showes a process of recognizing all the foreground species from a 3-channel signal , which contains the overlapping vocalization of 3 birds.  

![dir_structure](https://github.com/dalision/IVA-Xception/blob/master/images/recognize.gif )

**Simulation page:**

![dir_structure](https://github.com/dalision/IVA-Xception/blob/master/images/simulation_lab.gif )



## Directory structure：

![dir_structure](https://github.com/dalision/IVA-Xception/blob/master/images/dir_structure.png )