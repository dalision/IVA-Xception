# Edited by Yusheng Dai and Haipeng Zhou 2020 
#School of Cyber Science and Engineering, Sichuan University,
#Chengdu, the People’s Republic of China
import random
import sys
import  os
sys.path.append("..")
sys.path.append("./")
from PIL import Image
from PIL import ImageEnhance
import math
import copy
import numpy as np
import cv2
######################## CONFIG ##########################
import config as cfg
import shutil
random.seed(46)
#Fixed random seed
RANDOM = cfg.getRandomState()

def resetRandomState():
    global RANDOM
    RANDOM = cfg.getRandomState()

########################## I/O ###########################
def openImage(path, im_dim=1):
    
    # Open image
    if im_dim == 3:
        img = cv2.imread(path, 1)
    else:
        img = cv2.imread(path, 0)
    # Convert to floats between 0 and 1
    img = np.asarray(img / 255., dtype='float32')    

    return img

def showImage(img, name='IMAGE', timeout=-1):

    cv2.imshow(name, img)
    cv2.waitKey(timeout)

def saveImage(img, path):

    cv2.imwrite(path, img)

#################### PRE-PROCESSING ######################
def normalize(img, zero_center=False):

    # Normalize
    if not img.min() == 0 and not img.max() == 0:
        img -= img.min()
        img /= img.max()
    else:
        img = img.clip(0, 1)

    # Use values between -1 and 1
    if zero_center:
        img -= 0.5
        img *= 2

    return img

def substractMean(img, clip=True):

    # Only suitable for RGB images
    if len(img.shape) == 3:

        # Normalized image?
        if img.max() <= 1.0:

            img[:, :, 0] -= 0.4850 #B
            img[:, :, 1] -= 0.4579 #G
            img[:, :, 2] -= 0.4076 #R

        else:

            img[:, :, 0] -= 123.680 #B
            img[:, :, 1] -= 116.779 #G
            img[:, :, 2] -= 103.939 #R
        
    else:
        img -= np.mean(img)

    # Clip values
    if clip:
        img = img.clip(0, img.max())

    return img

def prepare(img):

    # ConvNet inputs in Theano are 4D-vectors: (batch size, channels, height, width)
    
    # Add axis if grayscale image
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]

    # Transpose axis, channels = axis 0
    img = np.transpose(img, (2, 0, 1))

    # Add new dimension
    img = np.expand_dims(img, 0)

    return img

######################## RESIZING ########################
def resize(img, width, height, mode='squeeze'):

    if img.shape[:2] == (height, width):
        return img

    if mode == 'crop' or mode == 'cropCenter':
        img = cropCenter(img, width, height)
    elif mode == 'cropRandom':
        img = cropRandom(img, width, height)
    elif mode == 'fill':
        img = fill(img, width, height)
    else:
        img = squeeze(img, width, height)

    return img

def squeeze(img, width, height):

    # Squeeze resize: Resize image and ignore aspect ratio
    
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

def cropRandom(img, width, height):

    # Random crop: Scale shortest side to minsize, crop with random offset

    # Original image shape
    h, w = img.shape[:2]
    aspect_ratio = float(max(h, w)) / float(min(h, w))

    # Scale original image
    minsize = int(max(width, height) * 1.1)
    if w <= h and w < minsize:
        img = squeeze(img, minsize, int(minsize * aspect_ratio))
    elif h < w and h < minsize:
        img = squeeze(img, int(minsize * aspect_ratio), minsize)

    #crop with random offset
    h, w = img.shape[:2]
    top = RANDOM.randint(0, h - height)
    left = RANDOM.randint(0, w - width)
    new_img = img[top:top + height, left:left + width]

    return new_img

def cropCenter(img, width, height):

    # Center crop: Scale shortest side, crop longer side

    # Original image shape
    h, w = img.shape[:2]
    aspect_ratio = float(max(h, w)) / float(min(h, w))

    # Scale original image
    if w == h:
        img = squeeze(img, max(width, height), max(width, height))
    elif width >= height:
        if h >= w:
            img = squeeze(img, width, int(width * aspect_ratio))
        else:            
            img = squeeze(img, int(height * aspect_ratio), height)
    else:
        if h >= w:
            img = squeeze(img, int(height / aspect_ratio), height)
        else:
            img = squeeze(img, int(height * aspect_ratio), height)

    #Crop from original image
    top = (img.shape[0] - height) // 2
    left = (img.shape[1] - width) // 2
    new_img = img[top:top + height, left:left + width]
    
    return new_img

def fill(img, width, height):

    # Fill mode: Scale longest side, pad shorter side with noise

    # Determine new shape
    try:
        new_shape = (height, width, img.shape[2])
    except:
        new_shape = (height, width)

    # Allocate array with noise
    new_img = RANDOM.normal(0.0, 1.0, new_shape)

    # Original image shape
    h, w = img.shape[:2]
    aspect_ratio = float(max(h, w)) / float(min(h, w))

    # Scale original image
    if w == h:
        img = squeeze(img, min(width, height), min(width, height))
    elif width >= height:
        if h >= w:
            img = squeeze(img, int(height / aspect_ratio), height)
        else:
            img = squeeze(img, width, int(width / aspect_ratio))
    else:
        if h >= w:
            img = squeeze(img, width, int(width * aspect_ratio))            
        else:
            img = squeeze(img, width, int(width / aspect_ratio))
            
    # Place original image at center of new image
    top = (height - img.shape[0]) // 2
    left = (width - img.shape[1]) // 2
    new_img[top:top + img.shape[0], left:left + img.shape[1]] = img
    
    return new_img

###################### AUGMENTATION ######################
def augment(img, augmentation={}, count=1, probability=0.5,img_name=''):
    # Make working copy
    augmentations = copy.deepcopy(augmentation)
    # Choose number of augmentations according to count
    # Count = 3 means either 0, 1, 2 or
    # 3 different augmentations

    # Roll the dice if we do augment or not
        # Choose one method
    while(count>0):
        if(count==2 and random.randint(0,4)>0):
            continue
        aug = RANDOM.choice(list(augmentations.keys()))
        # Call augementation methods
        if aug == 'flip':
            img = flip(img, augmentations[aug])
        elif aug == 'rotate':
            img = rotate(img, augmentations[aug])
        elif aug == 'zoom':
            img = zoom(img, augmentations[aug])
        elif aug == 'crop':
            if isinstance(augmentations[aug], float):
                img = crop(img, top=augmentations[aug], left=augmentations[aug], right=augmentations[aug], bottom=augmentations[aug])
            else:
                img = crop(img, top=augmentations[aug][0], left=augmentations[aug][1], bottom=augmentations[aug][2], right=augmentations[aug][3])
        elif aug == 'roll':
            img = roll(img, vertical=augmentations[aug], horizontal=augmentations[aug])
        elif aug == 'roll_v':
            img = roll(img, vertical=augmentations[aug], horizontal=0)
        elif aug == 'roll_h':
            img = roll(img, vertical=0, horizontal=augmentations[aug])
        elif aug == 'mean':
            img = mean(img, augmentations[aug])
        elif aug == 'noise':
            img = noise(img, augmentations[aug])
        elif aug == 'dropout':
            img = dropout(img, augmentations[aug])
        elif aug == 'blackout':
            img = blackout(img, augmentations[aug])
        elif aug == 'blur':
            img = blur(img, augmentations[aug])
        elif aug == 'brightness':
            img = brightness(img, augmentations[aug])
        elif aug == 'multiply':
            img = randomMultiply(img, augmentations[aug])
        elif aug == 'hue':
            img = hue(img, augmentations[aug])
        elif aug == 'lightness':
            img = lightness(img, augmentations[aug])
        elif aug == 'add':
            img = add(img, augmentations[aug])
        elif aug == 'pitch_shift':
            img = pitch_shift(img, augmentations[aug])#stand for probabolily
        elif aug == 'cut_vertical':
            img = cut_vertical(img, augmentations[aug])#stand for probabolily
        elif aug == 'cut_horizon':
            img = cut_horizon(img, augmentations[aug])#stand for probabolily
        elif aug == 'fun_color':                      #color change
            img = fun_color(img)
        elif aug == 'fun_Contrast':
            img = fun_Contrast(img)
        elif aug == 'fun_Sharpness':
            img = fun_Sharpness(img)
        elif aug == 'fun_bright':
            img = fun_bright(img)
        else:
            pass
        count-=1
        img_name=img_name+'_'+aug
    # Remove key so we avoid duplicate augmentations
    del augmentations[aug]
    # Count (even if we did not augment)
    print("using{} has been finished".format(aug))
    return (img,img_name)    
######################################we added ######################################
def pitch_shift(img,p):
    rand=np.random.rand(1)
    if p > rand:    
        height=img.shape[0]
        width=img.shape[1]
        img=img[6:height-10]
        img=cv2.resize(img,(width,height),interpolation=cv2.INTER_CUBIC) 
      
    return img 
def cut_horizon(pic,aug):
    width = pic.shape[1]
    height = pic.shape[0]
    pieces = []
    flag = True
    while flag:
        fraction = np.random.randint(10, 100)
        if pic.shape[0] > fraction:
            piece = pic[:fraction]
            pic = pic[fraction:]
        else:
            piece = pic
            flag = False
            fraction=piece.shape[0]
        ratio = np.random.uniform(0.95, 1.05)         
        if ratio>1:
        
            piece = cv2.resize(piece, (width, int(ratio * fraction)),interpolation=cv2.INTER_AREA)  
            
        else:
            
            piece = cv2.resize(piece, (width, int(ratio * fraction)),interpolation=cv2.INTER_CUBIC)
            
        pieces.append(piece)
    
    new_pic = np.concatenate(pieces, axis=0)
    if new_pic.shape[0]>height:
        pic = cv2.resize(new_pic, (width, height),interpolation=cv2.INTER_AREA)
    else:
        pic = cv2.resize(new_pic, (width, height),interpolation=cv2.INTER_CUBIC)
    
    return pic


def cut_vertical(pic,aug):

    width = pic.shape[1]
    height = pic.shape[0]
    pieces = []
    flag = True
    while flag:
        fraction = np.random.randint(10, 100)
        if pic.shape[1] > fraction:
            piece = pic[:, :fraction]
            pic = pic[:, fraction:]
        else:
            piece = pic
            flag = False
            fraction=piece.shape[1]
        ratio = np.random.uniform(0.9, 1.1)  
        piece = cv2.resize(piece, (int(ratio * fraction), height))  
        pieces.append(piece)

    #  pieces=np.array(pieces)
    new_pic = np.concatenate(pieces, axis=1)
    pic = cv2.resize(new_pic, (width, height))
    
    
    return pic


#/*********************************************color duration by PIL**************/


def fun_color(image):
    image = np.asarray(image *255., dtype='float32')
    image = np.asarray(image, dtype='uint8')
    coefficient = np.random.uniform(0.8,1.5)
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    enh_col = ImageEnhance.Color(image)
    image_colored1 = enh_col.enhance(coefficient)
    image = cv2.cvtColor(np.asarray(image_colored1), cv2.COLOR_RGB2GRAY)    
    image = np.asarray(image / 255., dtype='float32')
    return image

def fun_Contrast(image):
    image = np.asarray(image *255., dtype='float32')
    image = np.asarray(image, dtype='uint8')
    coefficient = np.random.uniform(0.8,1.5)
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    enh_con = ImageEnhance.Contrast(image)
    image_contrasted1 = enh_con.enhance(coefficient)
    image = cv2.cvtColor(np.asarray(image_contrasted1), cv2.COLOR_RGB2GRAY)
    image = np.asarray(image / 255., dtype='float32')
    return image
 
def fun_Sharpness(image):
    image = np.asarray(image *255., dtype='float32')
    image = np.asarray(image, dtype='uint8')
    coefficient = np.random.uniform(0.8,3)
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    enh_sha = ImageEnhance.Sharpness(image)
    image_sharped1 = enh_sha.enhance(coefficient)
    image = cv2.cvtColor(np.asarray(image_sharped1), cv2.COLOR_RGB2GRAY)
    image = np.asarray(image / 255., dtype='float32')
    return image
 
def fun_bright(image):
    image = np.asarray(image *255., dtype='float32')
    image = np.asarray(image, dtype='uint8')
    coefficient=np.random.uniform(0.8,1.5)
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    enh_bri = ImageEnhance.Brightness(image)
    image_brightened1 = enh_bri.enhance(coefficient)
    image = cv2.cvtColor(np.asarray(image_brightened1), cv2.COLOR_RGB2GRAY)
    image = np.asarray(image / 255., dtype='float32')
    return image
#########################end##############################################
def flip(img, flip_axis=1):
    
    return cv2.flip(img, flip_axis)

def rotate(img, angle, zoom=1.0):

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), RANDOM.uniform(-angle, angle), zoom)
    
    return cv2.warpAffine(img, M,(w, h))

def zoom(img, amount=0.33):

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, 1 + RANDOM.uniform(0, amount))
    
    return cv2.warpAffine(img, M,(w, h))

def crop(img, top=0.1, left=0.1, bottom=0.1, right=0.1):

    h, w = img.shape[:2]

    t_crop = max(1, int(h * RANDOM.uniform(0, top)))
    l_crop = max(1, int(w * RANDOM.uniform(0, left)))
    b_crop = max(1, int(h * RANDOM.uniform(0, bottom)))
    r_crop = max(1, int(w * RANDOM.uniform(0, right)))

    img = img[t_crop:-b_crop, l_crop:-r_crop]    
    img = squeeze(img, w, h)

    return img

def roll(img, vertical=0.1, horizontal=0.1):

    # Vertical Roll
    img = np.roll(img, int(img.shape[0] * RANDOM.uniform(-vertical, vertical)), axis=0)

    # Horizontal Roll
    img = np.roll(img, int(img.shape[1] * RANDOM.uniform(-horizontal, horizontal)), axis=1)

    return img

def mean(img, normalize=True):

    img = substractMean(img, True)

    if normalize and not img.max() == 0:
        img /= img.max()

    return img

def noise(img, amount=0.05):
    print(img.shape)
    img += RANDOM.normal(0.0, RANDOM.uniform(0, amount**0.5), img.shape)
    img = np.clip(img, 0.0, 1.0)
    

    return img

def dropout(img, amount=0.25):

    d = RANDOM.uniform(0, 1, img.shape)
    d[d <= amount] = 0
    d[d > 0] = 1
    
    return img * d

def blackout(img, amount=0.25):

    b_width = int(img.shape[1] * amount)
    b_start = RANDOM.randint(0, img.shape[1] - b_width)

    img[:, b_start:b_start + b_width] = 0

    return img

def blur(img, kernel_size=3):

     return cv2.blur(img, (kernel_size, kernel_size))

def brightness(img, amount=0.25):

    img *= RANDOM.uniform(1 - amount, 1 + amount)
    img = np.clip(img, 0.0, 1.0)

    return img

def randomMultiply(img, amount=0.25):

    img *= RANDOM.uniform(1 - amount, 1 + amount, size=img.shape)
    img = np.clip(img, 0.0, 1.0)

    return img

def hue(img, amount=0.1):

    try:
        # Only works with BGR images
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] *= RANDOM.uniform(1 - amount, 1 + amount)
        hsv[:, :, 0].clip(0, 360)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)        
    except:
        pass

    return img

def lightness(img, amount=0.25):

    try:
        # Only works with BGR images
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] *= RANDOM.uniform(1 - amount, 1 + amount)
        lab[:, :, 0].clip(0, 255)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    except:
        pass

    return img

def add(img, items):

    # Choose one item from List
    index = RANDOM.randint(len(items))

    # Open and resize image
    img2 = openImage(items[index], cfg.IM_DIM)
    img2 = resize(img2, img.shape[1], img.shape[0])

    # Generate random weights
    w1 = RANDOM.uniform(1, 2)
    w2 = RANDOM.uniform(1, 2)

    # Add images and calculate average
    img = (img * w1 + img2 * w2) / (w1 + w2)

    return img


if __name__ == '__main__':
    """
    cmd para
    such as:
    python image.py H://data_1/test H://noise_filtered H://output 900

    """
    path=sys.argv[1]
    noise_path=sys.argv[2]
    newPath=sys.argv[3]
    num=int(sys.argv[4])
    
    folders=os.listdir(path)
    noises=os.listdir(noise_path)
    for folder in folders:
        if not os.path.exists(newPath+folder):
            os.makedirs(newPath+folder)
        files=os.listdir(path+folder)
        new_file_num=len(os.listdir(newPath+folder))
        while new_file_num<num:
            new_file_num=len(os.listdir(newPath+folder))
            index=random.randint(0,len(files)-1)
            im_path=path+folder+'/'+files[index]
            print(im_path)
            img = cv2.imread(im_path,0)
            img = np.asarray(img/255.,dtype="float32")
            img_name=os.path.splitext(files[index])[0]
            if len(files)*15>num:
                try:
                    (img_out,img_name)=augment(img, cfg.IM_AUGMENTATION, 1, cfg.AUGMENTATION_PROBABILITY,img_name)
                except:
                    print("wrong")
            else:
                try:
                    (img_out,img_name)=augment(img, cfg.IM_AUGMENTATION, 2, cfg.AUGMENTATION_PROBABILITY,img_name)
                except:
                    print("wrong")
            img_noise=cv2.imread(noise_path+noises[random.randint(0,3300)],0)
            img_noise = np.asarray(img_noise/255.,dtype=img_out.dtype)
            try:
                img_out=cv2.addWeighted(img_out,0.9,img_noise,0.1,0.)
                cv2.imwrite(newPath+folder+'/'+img_name+'.png', img_out*255.)
            except:
                print("wrong")
        
        


    
    
    
