import os 
from scipy.io import wavefile
def pathss(root):
    for clas in list(os.listdir(root)):
        dirp = os.path.join(root,clas)
        for i in range(0,50,1):
            sourceX = os.path.join(dirp,str(i),"original","sourceX.wav")
            sourceY = os.path.join(dirp,str(i),"original","sourceY.wav")
            mixpath =  os.path.join(dirp,str(i),"mixed","mixed.wav")
            separatep1 = os.path.join(dirp,str(i),"separation","0.wav")
            separatep2 = os.path.join(dirp,str(i),"separation","1.wav")
            mix,sep1,spe2 = separate2(sourceX,sourceY)
            wavefile.write(mixpath,44100,mix)
            wavefile.write(separatep1,44100,sep1)
            wavefile.write(separatep2,44100,spe2)
            
            

path = r""
pathss(path)

def separate2():
    pass 