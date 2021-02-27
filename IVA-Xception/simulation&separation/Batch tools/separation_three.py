import os 
from scipy.io import wavefile
def pathss(root):
    for clas in list(os.listdir(root)):
        dirp = os.path.join(root,clas)
        for i in range(0,50,1):
            sourceX = os.path.join(dirp,str(i),"original","0.wav")
            sourceY = os.path.join(dirp,str(i),"original","1.wav")
            sourcez = os.path.join(dirp,str(i),"original","2.wav")
            mixpath =  os.path.join(dirp,str(i),"mixed","mixed.wav")
            separatep1 = os.path.join(dirp,str(i),"separation","0.wav")
            separatep2 = os.path.join(dirp,str(i),"separation","1.wav")
            separatep3 = os.path.join(dirp,str(i),"separation","2.wav")
            mix,sep1,spe2,spe3 = separate3(sourceX,sourceY,sourcez)
            wavefile.write(mixpath,44100,mix)
            wavefile.write(separatep1,44100,sep1)
            wavefile.write(separatep2,44100,spe2)
            wavefile.write(separatep3,44100,spe3)
            
            
path = r""
pathss(path)

def separate3():
    pass 