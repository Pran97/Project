
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
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import math
import glob

params = {
    'nbpatterns': 3,        # number of images per episode
    'nbprescycles': 3,      # number of presentations for each image
    'prestime': 20,         # number of time steps for each image presentation
    'prestimetest': 3,      # number of time steps for the test (degraded) image
    'interpresdelay': 2,    # number of time steps (with zero input) between two presentations
    'patternsize': 1600,    # size of the images (32 x 32 = 1024)
    'nbiter': 1000,       # number of episodes
    'probadegrade': .5,     # when contiguousperturbation is False (which it shouldn't be), probability of zeroing each pixel in the test image
    'lr': 1e-4,   # Adam learning rate
    'print_every': 10,      # how often to print statistics and save files
    'homogenous': 0,        # whether alpha should be shared across connections 
    'rngseed':0             # random seed
}


#ttype = torch.FloatTensor;         # For CPU
ttype = torch.cuda.FloatTensor;     # For GPU




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
        self.w = Variable(.01 * torch.randn(40,40).type(ttype), requires_grad=True)
        self.alpha = Variable(.01 * torch.randn(40,40).type(ttype),requires_grad=True)
        self.eta = Variable(.01 * torch.ones(1).type(ttype), requires_grad=True) 
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
    def forward(self, x,yin,hebb):
        
        encoded=self.encoder(x)
        
        input = encoded.view(encoded.size(0), -1)
        yout = F.tanh( yin.mm(self.w + torch.mul(self.alpha, hebb)) + input )
        hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0]
        yout=yout.view(yout.size(0),1,40,40)
        decoded=self.decoder(yout)
        return encoded,decoded,yout,hebb
    def initialZeroState(self):
        return Variable(torch.zeros(1, 40).type(ttype))

    def initialZeroHebb(self):
        return Variable(torch.zeros(40,40).type(ttype))

import torchvision.transforms as transforms
train_image_folder = dset.ImageFolder(root='train',transform=transforms.ToTensor())
train_test_image_folder=dset.ImageFolder(root='test', transform=transforms.Compose([transforms.Resize((180,180))]))

from torch.utils.data import DataLoader
loader_train=DataLoader(dataset=train_image_folder,batch_size=3, shuffle=True)

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
