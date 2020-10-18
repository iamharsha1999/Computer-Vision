import torch
import torch.nn as nn

def DoubleConv(in_channels, out_channels):
    
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding= 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
        nn.ReLU(inplace=True)
    )

def ConvTranspose(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4,stride=2, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):

    def __init__(self, n_classes):

        """
        n_classes -> Should exclude the background
        """

        super().__init__()
        
        self.layer1 = DoubleConv(3, 64)
        self.mp1 = nn.MaxPool2d(2)       

        self.layer2 = DoubleConv(64, 128)
        self.mp2 = nn.MaxPool2d(2)

        self.layer3 = DoubleConv(128, 256)
        self.mp3 = nn.MaxPool2d(2)

        self.layer4 = DoubleConv(256, 512)
        self.mp4 = nn.MaxPool2d(2)

        self.layer5 = DoubleConv(512, 1024)

        self.layer6 = ConvTranspose(1024, 256)
        self.layer7 = DoubleConv(256 + 512,512)

        self.layer8 = ConvTranspose(512 , 128)
        self.layer9 = DoubleConv(128 + 256,256)

        self.layer10 = ConvTranspose(256,64)
        self.layer11 = DoubleConv(64 + 128,128)

        self.layer12 = ConvTranspose(128, 32)
        self.layer13 = DoubleConv(32 + 64,64)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

        self.fl = nn.Conv2d(64, n_classes + 1, kernel_size = 1, padding=0)
    
    def forward(self, x):

        c1 = self.layer1(x)
        x = self.mp1(c1)

        c2 = self.layer2(x)
        x = self.mp2(c2)

        c3 = self.layer3(x)
        x = self.mp3(c3)

        c4 = self.layer4(x)
        
        x = self.mp4(c4)

        x = self.layer5(x)       

        x = self.layer6(x)
      
        x = torch.cat([x, c4], dim = 1)
        x = self.layer7(x)

        x = self.layer8(x)
        x = torch.cat([x, c3], dim = 1)
        x = self.layer9(x)

        x = self.layer10(x)
        x = torch.cat([x, c2], dim = 1)
        x = self.layer11(x)

        x = self.layer12(x)
        x = torch.cat([x, c1], dim = 1)
        x = self.layer13(x)

        x = self.fl(x)

        return x








        
        






