#  IVA-Xception：

## Introduction：

IVA-Xception model which can achieve high performance in identifying multiple birds from overlapping bird sounds recordings based on IVA and CNN.

## Directory structure：

│ README.md
├─IVA-Xception
│ │
│ ├─audio_recognizer
│ │ │ 2bird_mutiple_recognizer(withoutIVA).py
│ │ │ 3bird_mutiple_recognizer(withoutIVA).py
│ │ │ model.py
│ │ │ separate_recognizer.py
│ │ │ subclass.py
│ │ │
│ │ ├─csv
│ │ │ csvgenerate.py
│ │ │ sample.csv
│ │ │
│ │ ├─save
│ │ │ best_Xecption.pt
│ │ │
│ │ └─testfile
│ │ audio.py
│ │ pic.py
│ │
│ └─simulation&separation
│ ├─Batch tools
│ │ mixtoolsforthree.py
│ │ mixtoolsfortwo.py
│ │ separation_path.py
│ │ separation_three.py
│ │
│ ├─three-birds
│ │ auxiva_pca.py
│ │ generate_samples.py
│ │ overiva.py
│ │ routines.py
│ │ three-bird(noisy).py
│ │ three-bird(withoutnoisy).py
│ │
│ └─two-birds
│ auxiva_pca.py
│ generate_samples.py
│ overiva.py
│ routines.py
│ twobird(clean).py
│ twobird(noisy).py
│
└─trianCNN
│ model.py
│ test.py
│ train.py
│
├─audio2spec
│ audio.py
│ config.py
│ spec.py
│
├─csv
│ test.csv
│ train.csv
│ val.csv
│
├─save
│ Intructions.txt
│
├─spec augmentation
│ config.py
│ image.py
│
└─trianCNNcode_data
website.txt
