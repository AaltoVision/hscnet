import torch
import torch.nn as nn

def one_hot(x, N=25):   
    one_hot = torch.FloatTensor(x.size(0), N, x.size(1), 
                                x.size(2)).zero_().to(x.device)
    one_hot = one_hot.scatter_(1, x.unsqueeze(1), 1)           
    return one_hot

def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, 
                    padding=(kernel_size-1), dilation=2),
        nn.ELU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, 
                    padding=(kernel_size-1), dilation=2),
        nn.ELU(inplace=True)
    )

def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2,
                    padding=0, output_padding=0),
        nn.ELU(inplace=True)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                    stride=stride, padding=(kernel_size-1)//2),
        nn.ELU(inplace=True)
    )

def conv_(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0)
    )


class CondLayer(nn.Module):
    """
    implementation of the element-wise linear modulation layer
    """
    def __init__(self):
        super(CondLayer, self).__init__()
        self.elu = nn.ELU(inplace=True)
    def forward(self, x, gammas, betas):
        return self.elu((gammas * x) + betas)


class HSCNet(nn.Module):
    """
    implementation of the hierarchical scene coordinate network
    """
    def __init__(self, training=True, dataset='7S'):
        super(HSCNet, self).__init__()
        self.training = training
        self.dataset = dataset
        self.cond = CondLayer()

        # regression
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
        
        # classification 
        if self.dataset in ['i7S', 'i12S', 'i19S']:
            conv_planes_c2 = [256, 256, 256, 256, 128, 25]
        else:
            conv_planes_c2 = [128, 128, 128, 128, 64, 25]
        self.convc1 = conv(conv_planes[7], conv_planes_c2[0], kernel_size=3, 
                    stride=1)       
        self.dsconv1 = downsample_conv(conv_planes_c2[0], conv_planes_c2[1], 
                    kernel_size=3)
        self.upconv1 = upconv(conv_planes_c2[1], conv_planes_c2[2])
        self.convc2 = conv(conv_planes_c2[0] + conv_planes_c2[2], 
                    conv_planes_c2[3], kernel_size=1, stride=1)
        self.convc3 = conv(conv_planes_c2[3], conv_planes_c2[4], 
                    kernel_size=1, stride=1)
        self.convoutc2 = conv_(conv_planes_c2[4], conv_planes_c2[5])
        
        # classification 
        if self.dataset == 'i7S':
            conv_planes_c1 = [256, 256, 256, 256, 256, 128, 175]
        elif self.dataset == 'i12S':
            conv_planes_c1 = [256, 256, 256, 256, 256, 256, 300]
        elif self.dataset == 'i19S':
            conv_planes_c1 = [256, 256, 256, 256, 256, 256, 475]
        else:
            conv_planes_c1 = [256, 128, 128, 128, 128, 64, 25]
        self.dsconv2 = downsample_conv(conv_planes_c2[1], 
                    conv_planes_c1[0], kernel_size=3)      
        self.upconv2 = upconv(conv_planes_c1[0], conv_planes_c1[1])
        self.convc4 = conv(conv_planes_c1[1] + conv_planes_c2[1], 
                    conv_planes_c1[2], kernel_size=3, stride=1)
        self.upconv3 = upconv(conv_planes_c1[2], conv_planes_c1[3])
        self.convc5 = conv(conv_planes_c1[3] + conv_planes_c2[0], 
                    conv_planes_c1[4], kernel_size=1, stride=1)
        self.convc6 = conv(conv_planes_c1[4], conv_planes_c1[5], 
                    kernel_size=1, stride=1)
        self.convoutc1 = conv_(conv_planes_c1[5], conv_planes_c1[6])
        
        # generator 
        if self.dataset == 'i7S':
            conv_planes_g1 = [128, 128, 128, 256, 256]
        elif self.dataset == 'i12S':
            conv_planes_g1 = [256, 128, 128, 256, 256]
        elif self.dataset == 'i19S':
            conv_planes_g1 = [256, 128, 128, 256, 256]
        else:
            conv_planes_g1 = [32, 64, 128]
        self.gconv1_1 = conv(conv_planes_c1[6], conv_planes_g1[0], 
                    kernel_size=1, stride=1)
        self.gconv1_2 = conv(conv_planes_g1[0], conv_planes_g1[1], 
                    kernel_size=1, stride=1)
        self.gconv1_3 = conv(conv_planes_g1[1], conv_planes_g1[2], 
                    kernel_size=1, stride=1)   
        if self.dataset in ['i7S', 'i12S', 'i19S']:
            self.gconv1_4 = conv(conv_planes_g1[2], conv_planes_g1[3], 
                        kernel_size=1, stride=1)  
            self.gconv1_5 = conv(conv_planes_g1[3], conv_planes_g1[4], 
                        kernel_size=1, stride=1) 
            self.gconv1_gamma_1 = conv(conv_planes_g1[4], conv_planes_c2[3],
                        kernel_size=1, stride=1)
            self.gconv1_beta_1 = conv(conv_planes_g1[4], conv_planes_c2[3], 
                        kernel_size=1, stride=1)   
            self.gconv1_gamma_2 = conv(conv_planes_g1[4], conv_planes_c2[4], 
                        kernel_size=1, stride=1)
            self.gconv1_beta_2 = conv(conv_planes_g1[4], conv_planes_c2[4], 
                        kernel_size=1, stride=1)
        else:             
            self.gconv1_gamma_1 = conv(conv_planes_g1[2], conv_planes_c2[3], 
                        kernel_size=1, stride=1)
            self.gconv1_beta_1 = conv(conv_planes_g1[2], conv_planes_c2[3],  
                        kernel_size=1, stride=1)   
            self.gconv1_gamma_2 = conv(conv_planes_g1[2], conv_planes_c2[4],
                        kernel_size=1, stride=1)
            self.gconv1_beta_2 = conv(conv_planes_g1[2], conv_planes_c2[4],
                        kernel_size=1, stride=1)

        # generator
        if self.dataset == 'i7S':
            conv_planes_g2 = [128, 128, 128, 256, 512]
        elif self.dataset == 'i12S':
            conv_planes_g2 = [256, 128, 128, 256, 512]
        elif self.dataset == 'i19S':
            conv_planes_g2 = [256, 128, 128, 256, 512]
        else: 
            conv_planes_g2 = [50, 64, 128, 256, 512]
        self.gconv2_1 = conv(conv_planes_c1[6] + conv_planes_c2[5], 
                    conv_planes_g2[0], kernel_size=1, stride=1)
        self.gconv2_2 = conv(conv_planes_g2[0], conv_planes_g2[1], 
                    kernel_size=1, stride=1)
        self.gconv2_3 = conv(conv_planes_g2[1], conv_planes_g2[2], 
                    kernel_size=1, stride=1)
        self.gconv2_4 = conv(conv_planes_g2[2], conv_planes_g2[3], 
                    kernel_size=1, stride=1)
        self.gconv2_5 = conv(conv_planes_g2[3], conv_planes_g2[4], 
                    kernel_size=1, stride=1)       
        self.gconv2_gamma_1 = conv(conv_planes_g2[4], conv_planes[8], 
                    kernel_size=1, stride=1)
        self.gconv2_beta_1 = conv(conv_planes_g2[4], conv_planes[8], 
                    kernel_size=1, stride=1)   
        self.gconv2_gamma_2 = conv(conv_planes_g2[4], conv_planes[9], 
                    kernel_size=1, stride=1)
        self.gconv2_beta_2 = conv(conv_planes_g2[4], conv_planes[9], 
                    kernel_size=1, stride=1)
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, lbl_1=None, lbl_2=None):
        out = self.conv1a(x)
        out = self.conv1b(out)
        out = self.conv2a(out) 
        out = self.conv3a(out)
        out = self.conv3b(out)
        out = self.conv4a(out)
        out = self.conv4b(out)
        out_conv4c = self.conv4c(out)
        
        out_convc1 = self.convc1(out_conv4c)
        out_dsconv1 = self.dsconv1(out_convc1)
        
        out = self.dsconv2(out_dsconv1)
        out = self.upconv2(out)
        out = self.convc4(torch.cat((out, out_dsconv1), 1))
        out = self.upconv3(out)
        out = self.convc5(torch.cat((out, out_convc1), 1))
        out = self.convc6(out)
        out_lbl_1 = self.convoutc1(out)

        if self.training is not True:
            lbl_1 = torch.argmax(out_lbl_1, dim=1)
            lbl_1 = one_hot(lbl_1, out_lbl_1.size()[1])
        
        out = self.gconv1_1(lbl_1)
        out = self.gconv1_2(out)
        out = self.gconv1_3(out)
        if self.dataset in ['i7S', 'i12S', 'i19S']:
            out = self.gconv1_4(out)
            out = self.gconv1_5(out)        
        out_gconv1_gamma_1 = self.gconv1_gamma_1(out)
        out_gconv1_beta_1 = self.gconv1_beta_1(out)       
        out_gconv1_gamma_2 = self.gconv1_gamma_2(out)
        out_gconv1_beta_2 = self.gconv1_beta_2(out)
        
        out = self.upconv1(out_dsconv1)
        out = self.cond(self.convc2(torch.cat((out, out_convc1), 1)),
                    out_gconv1_gamma_1,out_gconv1_beta_1)
        out = self.cond(self.convc3(out), out_gconv1_gamma_2,
                    out_gconv1_beta_2)
        out_lbl_2 = self.convoutc2(out)
        
        if self.training is not True:
            lbl_2 = torch.argmax(out_lbl_2, dim=1)
            lbl_2 = one_hot(lbl_2, out_lbl_2.size()[1])
        
        out = self.gconv2_1(torch.cat((lbl_1, lbl_2), 1))
        out = self.gconv2_2(out)
        out = self.gconv2_3(out)
        out = self.gconv2_4(out)
        out = self.gconv2_5(out)       
        out_gconv2_gamma_1 = self.gconv2_gamma_1(out)
        out_gconv2_beta_1 = self.gconv2_beta_1(out)
        out_gconv2_gamma_2 = self.gconv2_gamma_2(out)
        out_gconv2_beta_2 = self.gconv2_beta_2(out)
           
        out = self.cond(self.conv5a(out_conv4c),out_gconv2_gamma_1,
                    out_gconv2_beta_1)
        out = self.cond(self.conv5b(out),out_gconv2_gamma_2,out_gconv2_beta_2)
        out_coord = self.convout(out)
        
        return out_coord, out_lbl_2, out_lbl_1
