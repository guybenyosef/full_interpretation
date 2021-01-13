import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.unet_model import *

class uInterp(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(uInterp, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1_1 = down(64, 128)
        self.down1_2 = down(128, 256)
        self.down1_3 = down(256, 512)
        self.down1_4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        # self.down2_2 = down(128, 256)
        # self.down2_3 = down(256, 512)
        # self.down2_4 = down(512, 512)
        self.outc = outconv(64, n_classes)
        self.down1_5 = down(512, 64)
        #self.down1_6 = down(256, 256)
        #self.down1_7 = down(512, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1_1(x1)
        x3 = self.down1_2(x2)
        x4 = self.down1_3(x3)
        x5 = self.down1_4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        x6 = self.down1_5(x5)
        #x7 = self.down1_5(x6)
        # x8 = self.down1_5(x7)
        features = torch.flatten(x6, 1)

        return out, features


class uInterp_topdown(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(uInterp_topdown, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1_1 = down(64, 128)
        self.down1_2 = down(128, 256)
        self.down1_3 = down(256, 512)
        self.down1_4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.down2_1 = down(n_classes, 128)
        self.down2_2 = down(128, 256)
        self.down2_3 = down(256, 512)
        self.down2_4 = down(512, 512)
        self.outc = outconv(64, n_classes)
        self.down1_5 = down(512, 64)
        self.down2_5 = down(512, 64)
        self.final_linear = nn.Linear(1152, 576) #v2

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1_1(x1)
        x3 = self.down1_2(x2)
        x4 = self.down1_3(x3)
        x5 = self.down1_4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        interp_out = self.outc(x)
        x = self.down2_1(interp_out)
        x = self.down2_2(x)
        x = self.down2_3(x)
        x = self.down2_4(x)
        x6 = self.down1_5(x5)
        x7 = self.down2_5(x)
        #features = torch.flatten(x6+x7, 1) # v1
        featuresBU = torch.flatten(x6, 1) # v2
        featuresTD = torch.flatten(x7, 1) # v2
        features = self.final_linear(torch.cat((featuresBU, featuresTD), dim=1)) #v2

        return interp_out, features


# ------------------------
class BUTD_stream(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(BUTD_stream, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1_1 = down(64, 128)
        self.down1_2 = down(128, 256)
        self.down1_3 = down(256, 512)
        self.down1_4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.down1_5 = down(512, 64)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1_1(x1)
        x3 = self.down1_2(x2)
        x4 = self.down1_3(x3)
        x5 = self.down1_4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        interp_out = self.outc(x)
        x6 = self.down1_5(x5)
        featuresBU = torch.flatten(x6, 1)

        return interp_out, featuresBU


class uInterp_multistream(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(uInterp_multistream, self).__init__()
        self.s1 = BUTD_stream(n_channels, n_classes)
        self.s2 = BUTD_stream(n_classes, n_classes)
        self.down_1 = down(n_classes, 128)
        self.down_2 = down(128, 256)
        self.down_3 = down(256, 512)
        self.down_4 = down(512, 512)
        self.down_5 = down(512, 64)
        self.final_linear = nn.Linear(576*3, 576) #v2


    def forward(self, x):
        interp1, feat1BU = self.s1(x)
        interp2, feat2BU = self.s2(interp1)
        x = self.down_1(interp2)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.down_4(x)
        x = self.down_5(x)
        feat3BU = torch.flatten(x, 1) # v2
        # features = torch.flatten(x6+x7, 1) # v1
        features = self.final_linear(torch.cat((torch.cat((feat1BU, feat2BU), dim=1), feat3BU), dim=1))

        return interp2, features
