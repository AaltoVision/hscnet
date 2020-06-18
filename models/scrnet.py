import torch.nn as nn

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=stride, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
        )

def conv_(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0)
        )


class SCRNet(nn.Module):
    """
    implementation of the baseline scene coordinate regression network
    """
    def __init__(self):
        super(SCRNet, self).__init__()

        conv_planes = [64, 128, 128, 256, 256, 512, 512, 512, 4096, 4096]
        self.conv1a = conv(3,              conv_planes[0])
        self.conv1b = conv(conv_planes[0], conv_planes[1], stride=2)        
        self.conv2a = conv(conv_planes[1], conv_planes[2], stride=2)     
        self.conv3a = conv(conv_planes[2], conv_planes[3])
        self.conv3b = conv(conv_planes[3], conv_planes[4], stride=2)      
        self.conv4a = conv(conv_planes[4], conv_planes[5])
        self.conv4b = conv(conv_planes[5], conv_planes[6])
        self.conv4c = conv(conv_planes[6], conv_planes[7])       
        self.conv5a = conv(conv_planes[7], conv_planes[8], kernel_size=1)
        self.conv5b = conv(conv_planes[8], conv_planes[9], kernel_size=1)

        self.convout = conv_(conv_planes[9], 3)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1a(x)
        out = self.conv1b(out)       
        out = self.conv2a(out)      
        out = self.conv3a(out)
        out = self.conv3b(out)     
        out = self.conv4a(out)
        out = self.conv4b(out)
        out = self.conv4c(out)     
        out = self.conv5a(out)
        out = self.conv5b(out)
        out = self.convout(out)

        return out
