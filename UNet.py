# Import PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a class for UNet model
class UNet(nn.Module):
    # Initialize the model with parameters
    def __init__(self, in_channels, out_channels, n_filters, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.bilinear = bilinear

        # Define the encoder blocks
        self.enc1 = self.double_conv(in_channels, n_filters)
        self.enc2 = self.double_conv(n_filters, n_filters * 2)
        self.enc3 = self.double_conv(n_filters * 2, n_filters * 4)
        self.enc4 = self.double_conv(n_filters * 4, n_filters * 8)

        # Define the bottleneck block
        self.bottleneck = self.double_conv(n_filters * 8, n_filters * 16)

        # Define the decoder blocks
        self.dec1 = self.up_conv(n_filters * 16, n_filters * 8)
        self.dec2 = self.up_conv(n_filters * 8, n_filters * 4)
        self.dec3 = self.up_conv(n_filters * 4, n_filters * 2)
        self.dec4 = self.up_conv(n_filters * 2, n_filters)

        # Define the output block
        self.out = nn.Conv2d(n_filters, out_channels, kernel_size=1)

    # Define a function for double convolution
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # Define a function for up convolution
    def up_conv(self, in_channels, out_channels):
        if self.bilinear:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=1),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels , out_channels // 2,
                                   kernel_size=2,
                                   stride=2),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True)
            )

    # Define the forward pass
    def forward(self, x):
        # Encode the input
        x1 = self.enc1(x)
        x2 = F.max_pool2d(x1, kernel_size=2)
        x3 = self.enc2(x2)
        x4 = F.max_pool2d(x3, kernel_size=2)
        x5 = self.enc3(x4)
        x6 = F.max_pool2d(x5, kernel_size=2)
        x7 = self.enc4(x6)
        x8 = F.max_pool2d(x7, kernel_size=2)

        # Pass through the bottleneck
        x9 = self.bottleneck(x8)

        # Decode the output
        x10 = torch.cat([x7 ,self.dec1(x9)], dim=1)
        x11 = torch.cat([x5 ,self.dec2(x10)], dim=1)
        x12 = torch.cat([x3 ,self.dec3(x11)], dim=1)
        x13 = torch.cat([x1 ,self.dec4(x12)], dim=1)

        # Generate the output
        x14 = self.out(x13)

        return x14

# Create an instance of UNet model with parameters
model = UNet(in_channels=3,
             out_channels=1,
             n_filters=32)
