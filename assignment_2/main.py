class UNet(nn.Module):
    def __init__(self, num_layers=6):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1, stride=2, padding_mode="reflect")
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1, stride=2, padding_mode="reflect")
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1, stride=2, padding_mode="reflect")
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1, stride=2, padding_mode="reflect")

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect")

        # decoder (upsampling)
        self.dec_conv0 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 64, 3, padding=1))
        self.dec_conv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 64, 3, padding=1))
        self.dec_conv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 64, 3, padding=1))
        self.dec_conv3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(128, 1, 3, padding=1))

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(e0))
        e2 = F.relu(self.enc_conv2(e1))
        e3 = F.relu(self.enc_conv3(e2))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(torch.concat([b, e3], 1)))
        d1 = F.relu(self.dec_conv1(torch.concat([d0, e2], 1)))
        d2 = F.relu(self.dec_conv2(torch.concat([d1, e1], 1)))
        d3 = self.dec_conv3(torch.concat([d2, e0], 1))  
        return d3
