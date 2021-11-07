import torch.nn as nn
import torch

class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2, layers_list=[64, 128, 256, 512, 1024]):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.layers_list = layers_list
        self.rcnn_layers = []

        for i, l in enumerate(self.layers_list):
            if(i==0):
                rcnn = RRCNN_block(ch_in=img_ch,ch_out=l,t=t)
                self.rcnn_layers.append(rcnn.cuda())
            else:
                rcnn = RRCNN_block(ch_in=self.layers_list[i-1],ch_out=l,t=t)
                self.rcnn_layers.append(rcnn.cuda())


        self.layers_list.reverse()
        self.up_layers = []
        for j, u in enumerate((self.layers_list)):
            if(j!=len(self.layers_list)-1):
                up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(u,self.layers_list[j+1],kernel_size=3,stride=1,padding=1,bias=True),nn.BatchNorm2d(self.layers_list[j+1]),
                nn.ReLU(inplace=True))
                self.up_layers.append(up.cuda())

        self.up_rcnn_layers = []
        for j, u in enumerate((self.layers_list)):
            if(j!=len(self.layers_list)-1):
                up_rcnn = RRCNN_block(ch_in=u, ch_out=self.layers_list[j+1],t=t)
                self.up_rcnn_layers.append(up_rcnn.cuda())
        
        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        ## Encoding
        x_li = []
        for rcnn in self.rcnn_layers:
            x1 = rcnn(x)
            x_li.append(x1)
            x1 = self.Maxpool(x1)
            x = x1
            
        # Decoding
        d = x_li[-1]
        for l, m, n in (zip(list(range(len(self.up_layers))), self.up_layers, self.up_rcnn_layers)):
                d = m(d)
                d = torch.cat((x_li[len(x_li)-(l+2)], d), dim=1)
                d = n(d)

        d = self.Conv_1x1(d)
        return d

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):
            if i==0:
                x1 = self.conv(x)
            x1 = self.conv(x+x1)
        return x1

class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


