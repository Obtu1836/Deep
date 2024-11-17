import torch as th 
from torch import nn 
from torch.nn import functional as f

class Block(nn.Module):
    def __init__(self,ins,ous,stride=1,shortcut=None):
        super().__init__()

        self.left=nn.Sequential(
            nn.Conv2d(ins,ous,3,stride,1,bias=False),
            nn.BatchNorm2d(ous),
            nn.ReLU(inplace=True),
            nn.Conv2d(ous,ous,3,1,1,bias=False),
            nn.BatchNorm2d(ous))
        
        self.right=shortcut
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return self.relu(out)

class Net(nn.Module):
    def __init__(self,num_class):
        super().__init__()

        self.layer=nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1))
        
        self.layer1=self.make_layer(64,64,1,2,False)
        self.layer2=self.make_layer(64,128,2,2)
        self.layer3=self.make_layer(128,256,2,2)
        self.layer4=self.make_layer(256,512,2,2)

        self.fc=nn.Linear(512,num_class)

        for model in self.modules():
            if isinstance(model,nn.Conv2d):
                nn.init.kaiming_normal_(model.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(model,nn.BatchNorm2d):
                nn.init.constant_(model.weight,1)
                nn.init.constant_(model.bias,0)

    
    def make_layer(self,ins,ous,stride,num,flag=True):

        if flag:
            shortcut=nn.Sequential(
                nn.Conv2d(ins,ous,1,stride,0,bias=False),
                nn.BatchNorm2d(ous))
        else:
            shortcut=None

        layers=[]
        layers.append(Block(ins,ous,stride,shortcut))
        for i in range(1,num):
            layers.append(Block(ous,ous))
        return nn.Sequential(*layers)
    
    def forward(self,x):

        x=self.layer(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=f.avg_pool2d(x,7)
        x=x.view(x.size(0),-1)

        return self.fc(x)

    def all_child(self,net):
        model=[]
        for var in net.children():
            if list(var.children())!=[]:
                model.extend(self.all_child(var))
            else:
                model.append(var)
        return model
    
    def get_all_child_modle(self):
        return self.all_child(self)

    @staticmethod
    def hook(modle,ins,ous):
        print(f"{ins[0].shape}-->{ous.shape}")

    @staticmethod
    def rm_hook(modules):
        map(lambda x:x.remove(),modules)
    
if __name__ == '__main__':
    
    net=Net(10)
    data=th.rand(10,3,224,224)

    layers=net.get_all_child_modle()
    for ly in layers:
        ly.register_forward_hook(net.hook)

    with th.no_grad():
        net(data)

    net.rm_hook(layers)
   
    


