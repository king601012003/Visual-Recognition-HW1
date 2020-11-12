import torch
import torch.nn as nn
import torch.nn.functional as F

class RessidualBlock_2(nn.Module):
    def __init__(self, input_channel, output_channel=5, stride=1):
        super(RessidualBlock_2, self).__init__()
        
        self.main_2 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channel)
            )
        
        self.skip_2 = nn.Sequential()
        
        #down sample
        if stride != 1:
            self.skip_2 = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channel)
                )
            
    def forward(self, x):
        
        x_skip = self.skip_2(x)
        x = self.main_2(x)
        x += x_skip
        x = F.relu(x)
        
        return x
    
class RessidualBlock_3(nn.Module):
    def __init__(self, input_channel, output_channel=5, stride=1):
        super(RessidualBlock_3, self).__init__()
        
        self.main_3 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel*4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channel*4),
            )
        
        self.skip_3 = nn.Sequential()
        
        #down sample
        if stride != 1 or input_channel == 64:
            self.skip_3 = nn.Sequential(
                nn.Conv2d(input_channel, output_channel*4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channel*4)
                )
            
    def forward(self, x):
        
        x_skip = self.skip_3(x)
        x = self.main_3(x)
        x += x_skip
        x = F.relu(x)
        
        return x
    
class ResNet18(nn.Module):
    def __init__(self, output_class=15, layer=[64, 64, 128, 256, 512], stride=[2, 1, 2, 2, 2]):
        super(ResNet18, self).__init__()
        
        model_layer = nn.ModuleList()
        
        self.start_layer = nn.Sequential(
            nn.Conv2d(3, layer[0], kernel_size=7, stride=stride[0], padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        
        for i in range(1, 5):
            model_layer.append(RessidualBlock_2(layer[i-1], layer[i], stride[i]))
            model_layer.append(RessidualBlock_2(layer[i], layer[i], 1))
            
        self.first_layer = nn.Sequential(
            model_layer[0],
            model_layer[1]
            )
        
        self.second_layer = nn.Sequential(
            model_layer[2],
            model_layer[3]
            )
        
        self.third_layer = nn.Sequential(
            model_layer[4],
            model_layer[5]
            )
        
        self.fourth_layer = nn.Sequential(
            model_layer[6],
            model_layer[7]
            )
        
        self.classify=nn.Sequential(
            nn.Linear(in_features=512, out_features=output_class, bias=True)
            )
        
    def forward(self, x):
        x = self.start_layer(x)
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.third_layer(x)
        x = self.fourth_layer(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=1, padding=0)
        x = x.view(-1, self.classify[0].in_features)
        x = self.classify(x)
        
        return x
        
class ResNet50(nn.Module):
    def __init__(self, output_class=196, layer=[64, 64, 128, 256, 512], stride=[2, 1, 2, 2, 2]):
        super(ResNet50, self).__init__()
        
        self.start_layer = nn.Sequential(
            nn.Conv2d(3, layer[0], kernel_size=7, stride=stride[0], padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        
        self.first_layer = nn.Sequential(
            RessidualBlock_3(layer[0], layer[1], stride[1]),
            RessidualBlock_3(layer[1]*4, layer[1], 1),
            RessidualBlock_3(layer[1]*4, layer[1], 1)
            )
        
        self.second_layer = nn.Sequential(
            RessidualBlock_3(layer[1]*4, layer[2], stride[2]),
            RessidualBlock_3(layer[2]*4, layer[2], 1),
            RessidualBlock_3(layer[2]*4, layer[2], 1),
            RessidualBlock_3(layer[2]*4, layer[2], 1)
            )
        
        self.third_layer = nn.Sequential(
            RessidualBlock_3(layer[2]*4, layer[3], stride[3]),
            RessidualBlock_3(layer[3]*4, layer[3], 1),
            RessidualBlock_3(layer[3]*4, layer[3], 1),
            RessidualBlock_3(layer[3]*4, layer[3], 1),
            RessidualBlock_3(layer[3]*4, layer[3], 1),
            RessidualBlock_3(layer[3]*4, layer[3], 1)
            )
        
        self.fourth_layer = nn.Sequential(
            RessidualBlock_3(layer[3]*4, layer[4], stride[4]),
            RessidualBlock_3(layer[4]*4, layer[4], 1),
            RessidualBlock_3(layer[4]*4, layer[4], 1)
            )
        
        self.classify=nn.Sequential(
            nn.Linear(in_features=2048, out_features=output_class, bias=True)
            )
        
    def forward(self, x):
        x = self.start_layer(x)
        x = self.first_layer(x)
        x = self.second_layer(x)
        x = self.third_layer(x)
        x = self.fourth_layer(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=1, padding=0)
        x = x.view(-1, self.classify[0].in_features)
        x = self.classify(x)
        
        return x

if __name__ == "__main__":
    print(ResNet50)     