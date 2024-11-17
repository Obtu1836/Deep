import torch as th
from torch import nn 
from torch.nn import functional as f

class Normal(nn.Module):
    def __init__(self,ins,ous,kernal,stride,pad):
        super().__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(ins,ous,kernal,stride,pad,bias=False),
            nn.BatchNorm2d(ous))
        
    def forward(self,x):
        return self.conv(x)

class Block(nn.Module):
    def __init__(self,ins,ous,stride=1,shortcut=None):
        super().__init__()
        mid=ous//4
        
        self.left=nn.Sequential(
            Normal(ins,mid,1,stride,0), 
            Normal(mid,ous,3,1,1) ,
            Normal(ous,ous,1,1,0),
            nn.ReLU(inplace=True))
        
        self.right=shortcut
    
    def forward(self,x):
        out=self.left(x)
        resdiual=x if self.right is None else self.right(x)
        out+=resdiual
        return f.relu(out,inplace=True)
    
class Net(nn.Module):
    def __init__(self,num_calss):
        super().__init__()

        self.layer=nn.Sequential(
            Normal(3,64,7,2,3),  # b,64,112,112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)) # b,64,56,56
        
        self.layer1=self.make_layer(64,256,1,3) # b,256,56,26
        self.layer2=self.make_layer(256,512,2,4) # b,512,28,28
        self.layer3=self.make_layer(512,1024,2,6) # b 1024,14,14
        self.layer4=self.make_layer(1024,2048,2,3) # b 2048,7,7
 
        self.fc=nn.Linear(2048,num_calss) # b,num_class

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    
    def make_layer(self,ins,ous,stride,num):

        shortcut=nn.Sequential(
            Normal(ins,ous,1,stride,0))
    
        layer=[]
        layer.append(Block(ins,ous,stride,shortcut))
        for i in range(1,num):
            layer.append(Block(ous,ous))
        
        return nn.Sequential(*layer)
    
    def forward(self,x):
        x=self.layer(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=f.avg_pool2d(x,7) # b,2048,1,1
        x=x.view(x.size(0),-1) # b,2048
        return self.fc(x)   # b,num_class
    

    def all_layer(self,net):
        outs=[]
        for layer in net.children():
            if list(layer.children())!=[]:
                outs.extend(self.all_layer(layer))
            else:
                outs.append(layer)
        return outs
    
    def get_all_layer(self):

        return self.all_layer(self)

    @staticmethod
    def hook(m,ins,ous):
        print(f"{ins[0].shape}-->{ous.shape}")

    @staticmethod
    def rm_hook(handles):
        map(lambda x:x.remove(),handles)

if __name__ == '__main__':
    
    data=th.rand(1,3,224,224)
    net=Net(10)

    layers=net.get_all_layer()
    handles=[]
    for ly in layers:
        hd=ly.register_forward_hook(net.hook)
        handles.append(hd)
    
    with th.no_grad():
        net(data)
    
    net.rm_hook(handles)
        
