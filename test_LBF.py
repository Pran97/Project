import torch.nn as nn
import torch
torch.cuda.empty_cache()
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import cv2
import os
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt
import math
import glob

def psnr(i1, i2):
    mse = torch.mean( (i1 - i2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''
     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    
    '''    
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val =np.array( [1, 2, 4, 8, 16, 32, 64, 128])
    val = 0
    val_ar=np.array(val_ar)
    val=np.sum(val_ar*power_val)
    return val   

def random(value,dist,comp,rang):
    gen=np.random.randint
    if dist==1:
        if rang==0:
            low=gen(0,value+1)
            high=gen(value,256)
        else:
            low=gen(max(value-1-rang,0),value+1)
            high=gen(value,min(value+rang+1,256))
            #low=max(value-1-rang,0)
            #high=min(value+rang+1,256)
    if comp==1:        
        return high
    else:
        return low


    
def reconstruct(img_lbp,rang,dist):
    hei,wid=np.shape(img_lbp)
   
    recon_hei=hei*3
    recon_wid=math.floor(wid*3/2)
    recon_image=np.zeros((recon_hei,recon_wid), np.uint8)
    for i in np.arange(0,hei,dtype='int'):
           for j in np.arange(0, int(wid/2),dtype='int'):
                recon_image[3*i+1,3*j+1]=img_lbp[i,2*j+1]
                temp=np.unpackbits(np.array([img_lbp[i,2*j]]))
                value=img_lbp[i,2*j+1]
                recon_image[3*i,3*j+2]=temp[7]*random(value,dist,1,rang) + (1-temp[7])*random(value,dist,0,rang)
                recon_image[3*i+1,3*j+2]=temp[6]*random(value,dist,1,rang) + (1-temp[6])*random(value,dist,0,rang)
                recon_image[3*i+2,3*j+2]=temp[5]*random(value,dist,1,rang) + (1-temp[5])*random(value,dist,0,rang)
                recon_image[3*i+2,3*j+1]=temp[4]*random(value,dist,1,rang) + (1-temp[4])*random(value,dist,0,rang)
                recon_image[3*i+2,3*j]=temp[3]*random(value,dist,1,rang) + (1-temp[3])*random(value,dist,0,rang)
                recon_image[3*i+1,3*j]=temp[2]*random(value,dist,1,rang) + (1-temp[2])*random(value,dist,0,rang)
                recon_image[3*i,3*j]=temp[1]*random(value,dist,1,rang) + (1-temp[1])*random(value,dist,0,rang)
                recon_image[3*i,3*j+1]=temp[0]*random(value,dist,1,rang) + (1-temp[0])*random(value,dist,0,rang)
    
    return recon_image

def converttolbp(img):
    width,height = img.size

    data = list(img.getdata()) 
    data = [data[offset:offset+width] for offset in range(0, width* height, width)]
    image=np.array(data)
    Width=3*math.floor(width/3)
    Height=3*math.floor(height/3)
    image=image[0:Width,0:Height]
    img_lbp = np.zeros(( math.floor(Height/3), math.floor(2*Width/3)), np.uint8)
    for i in np.arange(0,Height/3,dtype='int'):
        for j in np.arange(0, Width/3,dtype='int'):
            img_lbp[i,2*j+1]=image[3*i+1,3*j+1]
            img_lbp[i, 2*j] = lbp_calculated_pixel(image, 3*i+1, 3*j+1)
            
   # print("LBP Program is finished")
    return image,img_lbp

#def main():
image_list = []
original_image=[]
recon_im=[]
for filename in glob.glob("C:\\Users\\prana\\Desktop\\Machine Learning A-Z Template Folder\\Assignment2-master\\Set12\\Set12\\*.png"):
    im=Image.open(filename)
    image_list.append(im)
i=0
for img in image_list:

    ################# Change this part
    img =img.convert('L')  # convert image to 8-bit grayscale

    [image,img_lbp]=converttolbp(img)
    rang=5  # vary this to experiment
    dist=1
    recon_image=reconstruct(img_lbp,rang,dist)
    original_image.append(image)
    recon_im.append(recon_image)
    #temp1=np.zeros(np.shape(image))
    #temp2=np.zeros(np.shape(image))
    #[m,n]=np.shape(image)
    #Temp 3 is RGB of imgg
   # temp3=np.zeros([3,m,n])
    #temp3[0]=recon_image
    #temp3[1]=recon_image
    #temp3[2]=recon_image
    #print(image[100,100])
    #print(recon_image[100,100])
    #imagg = Image.fromarray(image)

    #imagg.save("C:\\Users\\RGP\\Desktop\\Image Processing\\Project\\original_n\\"+str(i)+".png")
    #imagg.show()
    #recon = Image.fromarray(recon_image)

    #recon.save("C:\\Users\\RGP\\Desktop\\Image Processing\\Project\\recon_n\\"+str(i)+".png")
    # recon.show()
    #temp3[:,:,0]=image

    i=i+1
original_image=np.array(original_image)
recon_im=np.array(recon_im)


def calc_mask(x):
    mask=torch.ones(x.shape).cuda()
    l=np.linspace(1,178,60,dtype=int)
    for batch in range(len(x)):
        mask[batch,l,l]=0
    comp_mask=1-mask
    return mask,comp_mask





class DeoisingCNN(nn.Module):
    def __init__(self, num_channels, num_of_layers=17):
        super(DeoisingCNN, self).__init__()#Inheriting properties of nn.Module class
        l=[]
        #padding 1 as kernel is of size 3 and we need the o/p of CNN block to give same size img and no maxpooling or BN in this layer
        first_layer=nn.Sequential(nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=9, padding=4),nn.ReLU(inplace=True))
        l.append(first_layer)
        #All blocks in b/w the first and last are the same having same i.e having depth and no maxpooling layer
        for _ in range(num_of_layers-2):
            second_layer = nn.Sequential(
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=9, padding=4),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True))#0.2
            l.append(second_layer)
        #Final layer is similar to the first CNN block
        l.append(nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=9, padding=4))
        self.mdl = nn.ModuleList(l)
    def forward(self, x):
        mask,comp_mask=calc_mask(x.detach().squeeze(1))
        out = self.mdl[0](x)
        for i in range(len(self.mdl) - 2):
            out = self.mdl[i + 1](out)
        out = self.mdl[-1](out)
        mask=mask.unsqueeze(1)
        comp_mask=comp_mask.unsqueeze(1)
        #out=mask*out+x*comp_mask
        out=out*mask
        return out

model1=torch.load('works.pkl').cuda()
from skimage.measure import compare_ssim as ssim

import torchvision.datasets as dset
train_test_image_folder=dset.ImageFolder(root='Set12', transform=transforms.Compose([transforms.Resize((180,180))]))
d=0
for a,b in train_test_image_folder:
    x=np.asarray(a)#There is some problem in PIL to tensor conversion so first convert to np array
    x=np.mean(x,2)
    imagg = Image.fromarray(x)
    _,lbf=converttolbp(imagg)
    recon=reconstruct(lbf,10,1)
    
    x=x.reshape(1,1,180,180)#Greyscale conversion and need approriate dim of (x,1,180,180)
    recon=recon.reshape(1,1,180,180)
    #for model to work
    tareget_image=torch.from_numpy(x/255.0).float().cuda()
    image=torch.from_numpy(recon/255.0).float().cuda()
    model1.eval()
    out=model1(image)
    est_image=image+out
    print(ssim(out[0,0,:,:].cpu().data.numpy()+image[0,0,:,:].cpu().data.numpy(),tareget_image[0,0,:,:].cpu().data.numpy()))
    cv2.imshow(' Input Reccon Image',(image.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    cv2.imshow('Restored Denoised Image',(est_image.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    cv2.imshow('Target Image',(tareget_image.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    filename = "Works2/test_input%d.png"%d
    cv2.imwrite(filename,(image.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    filename = "Works2/test_restored%d.png"%d
    cv2.imwrite(filename,(est_image.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    filename = "Works2/test_target%d.png"%d
    cv2.imwrite(filename,(tareget_image.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    d=d+1
    cv2.waitKey(5000)
    cv2.destroyAllWindows()