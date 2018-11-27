
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
import pytorch_ssim

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

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()#Inheriting properties of nn.Module class
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 6, stride=2, padding=0),  # (Win - k )/S +1 =Wout
            nn.ReLU(True), 
            nn.BatchNorm2d(64), 
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 64, 6, stride=2, padding=0), 
            nn.ReLU(True),
            nn.BatchNorm2d(64),  
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1)
            #out 40x40 = Wout
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, stride=1),  
            nn.ReLU(True), 
            nn.BatchNorm2d(64),  
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(64, 64, 6, stride=2, padding=0),  
            nn.ReLU(True), 
            nn.BatchNorm2d(64),  
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(64, 1, 6, stride=2, padding=0),
            nn.Tanh()
        )
    def forward(self, x):
        
        encoded=self.encoder(x)
        
        decoded=self.decoder(encoded)
        return encoded,decoded
import torchvision.transforms as transforms
train_image_folder = dset.ImageFolder(root='train',transform=transforms.ToTensor())
train_test_image_folder=dset.ImageFolder(root='test', transform=transforms.Compose([transforms.Resize((180,180))]))

from torch.utils.data import DataLoader
loader_train=DataLoader(dataset=train_image_folder,batch_size=5, shuffle=True)

from torch.autograd import Variable
model=AutoEncoder().cuda()
criterion = nn.MSELoss()
from torch import optim
optimizer = optim.Adam(model.parameters(),  lr=0.0001)

best_loss=np.inf
from skimage.measure import compare_ssim as ssim
epochs=20
iter=0
l=[]
itr=[]
import matplotlib.pyplot as plt
#Standard procedure for training.
for epoch in range(epochs):
    for i, (im1,l1) in enumerate(loader_train):
        model.train()
        images = Variable(im1[:,:1,:,:].cuda())
        #print(type(images.cpu().data))
        criterion = nn.MSELoss()
        optimizer.zero_grad()
        _,outputs = model(images)
        criterion = nn.MSELoss()
        
        loss = criterion(outputs, images)
        criterion = pytorch_ssim.SSIM(window_size = 9).cuda()
        loss+=0.05*(1-criterion(outputs, images))
        loss.backward()
        optimizer.step()
        if iter % 5 == 0:
            print('Epoch: {}.Iteration: {}. Loss: {}. SSID: {}'.format(epoch,iter, loss.data[0],ssim(outputs[0,0,:,:].cpu().data.numpy(),images[0,0,:,:].cpu().data.numpy())))
            l.append(psnr(outputs,images).cpu().item())
            itr.append(iter)
            plt.plot(itr,l)
            plt.ylabel('PSNR')
            plt.xlabel('iterations *10')
        iter=iter+1
        
        if(loss.data[0]<best_loss):
                print('saving')
                best_loss=loss.data[0]
                torch.save(model, 'noise5.pkl')

        #Evalute model performance on test set
        if (epoch%5==0):
            print(epoch)
            t=0
            for a,b in train_test_image_folder:
                model.eval()
                x=np.asarray(a)#There is some problem in PIL to tensor conversion so first convert to np array
                x=x[:,:,:1].reshape(1,1,180,180)#Greyscale conversion and need approriate dim of (x,1,180,180)
                #for model to work
                
                test_img=torch.from_numpy(x/255.0).float().cuda()
                _,out=model(test_img)
                
                print("PSNR of test image"+str(t)+" is "+str(psnr(out,test_img).cpu().item()))
                t=t+1
