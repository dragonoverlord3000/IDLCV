import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        
        self.enc_conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, padding_mode='reflect')  
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='reflect')  
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, padding_mode='reflect')  
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, padding_mode='reflect')  
        
        self.bottleneck_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1, padding_mode='reflect')  
        
        self.dec_conv0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True)
        )  
        self.dec_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True)
        )  
        self.dec_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),    
            nn.ReLU(inplace=True)
        )  
        self.dec_conv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64 + 3, 64, kernel_size=3, padding=1),      
            nn.ReLU(inplace=True)
        )  

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        # Encoder time
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(e0))
        e2 = F.relu(self.enc_conv2(e1))

        # Bottleneck time
        b = F.relu(self.enc_conv3(e2))
        b = F.relu(self.bottleneck_conv(b))  
        b = self.up(b)

        # Decoder time
        d0 = self.dec_conv0(torch.cat([b, e2], dim=1))
        d1 = self.dec_conv1(torch.cat([d0, e1], dim=1))
        d2 = self.dec_conv2(torch.cat([d1, e0], dim=1))
        d3 = self.dec_conv3(torch.cat([d2, x], dim=1))

        output = self.final_conv(d3)  
        return output
