import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        ###INPUT MODULES###
        #mobile net takes images of size (224x224x3) as input
        
        super(Model,self).__init__()
        self.mobile_net = models.mobilenet_v2(num_classes=512)
        self.image_module = nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU(True),
            nn.Linear(1024,512),
            nn.ReLU(True),

            )
        self.imu_module = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            )
        
        self.depth_module = nn.Sequential(#input of size: 200x88)
            nn.Conv2d(1,32,4,2,1,bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32,32,4,2,1,bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32,64,4,2,1,bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64,64,4,2,1,bias=True),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
          #  nn.Conv2d(64,128,4,2,1,bias=False),
          #  nn.Dropout(p=0.2),
          #  nn.BatchNorm2d(128),
          #  nn.ReLU(True),
          #  nn.Conv2d(128,128,4,2,1,bias=False),
          #  nn.Dropout(p=0.2),
          #  nn.BatchNorm2d(128),
          #  nn.ReLU(True),

          #  nn.Conv2d(128,256,4,2,1,bias=False),
          #  nn.Dropout(p=0.2),
          #  nn.BatchNorm2d(256),
          #  nn.ReLU(True),
          #  nn.Conv2d(256,256,4,2,1,bias=False),
          #  nn.Dropout(p=0.2),
          #  nn.BatchNorm2d(256),
          #  nn.ReLU(True),
            
        
            nn.Flatten(),

           #input to linear layer probably incorrect
           # nn.Linear(384 , 512),
           #  nn.Linear(3840,512),
            nn.Linear(3072,512),
            nn.Dropout(p=0.5),
            nn.ReLU(True),

            nn.Linear(512,512),
            nn.Dropout(p=0.5),
            nn.ReLU(True)
            )

        print("SUMMARY DEPTH MODULE")
        # print(summary(self.depth_module,(1,88,200)))
        self.speed_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True)
            )

        self.dense_layers = nn.Sequential(
            #512depth, 512image, 128speed, 256imu
            nn.Linear(1408,512),
            nn.ReLU(True)
            )

        ###COMMAND BRANCHEs###
        self.straight_net = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,3),  #outputlayer is supposed to have no activation function
            nn.Sigmoid()
            )

        self.right_net = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,3),  #outputlayer is supposed to have no activation function
            nn.Sigmoid()
            )
        self.left_net = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,3),  #outputlayer is supposed to have no activation function
            nn.Sigmoid()
            )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            print("Block:", block)
            if block=='mobile_net':
                pass
            else:
                for m in self._modules[block]:
                    print("M:", m)
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal(m.weight,mean=0, std=0.01)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal(m.weight,mean=0,std=0.01)
                    else:
                        pass
               # normal_init(m)

    def forward(self, input_data):
        image = input_data[0]
        image = self._image_module(image)
        imu   = input_data[2]
        imu   = self._imu_module(imu)
        depth = input_data[1]
        depth = self._depth_module(depth)
        speed = input_data[3]
        speed = self._speed_module(speed)
        command=input_data[4]
        
        concat = torch.cat((image,imu,depth,speed),1)
        concat = self._dense_layers(concat)
        
        ##########################################################
        ##1 here needs to be changed if batch size is changed!!##
        ##########################################################
        output = torch.Tensor()
        for i in range(1):
            if command[i]==1:
                if output.shape==torch.Size([0]):
                    output = self._straight_net(concat)
                else:
                    torch.stack(tensors=(output, self._straight_net(concat)),dim=1)

            elif command[i]==2:
                if output.shape==torch.Size([0]):
                    output = self._right_net(concat)
                else:
                    torch.stack(tensors=(output, self._right_net(concat)),dim=1)

            elif command[i]==0:
                if output.shape==torch.Size([0]):
                    output = self._left_net(concat)
                else:
                    torch.stack(tensors=(output, self._left_net(concat)),dim=1)

        return output

    
    def _image_module(self,x):
        x = self.mobile_net(x)
        return self.image_module(x)
        
    def _imu_module(self,x):
        return self.imu_module(x)

    def _depth_module(self,x):
        return self.depth_module(x)

    def _speed_module(self,x):
        return self.speed_module(x)

    def _dense_layers(self,x):
        return self.dense_layers(x)

    def _straight_net(self,x):
        return self.straight_net(x)

    def _right_net(self,x):
        return self.right_net(x)

    def _left_net(self,x):
        return self.left_net(x)
    
import torchvision
print('loading file')
model = torch.load('./new_model.pt')
print('finished loading')