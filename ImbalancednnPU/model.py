
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Net(nn.Module):
    def __init__(self, seed, parameter_momentum):
        super(Net, self).__init__()
        L0 = 784
        L1 = 300
        L2 = 301
        L3 = 302
        L4 = 303
        L5 = 304
        L6 = 305

        #torch.manual_seed(seed)

        self.L1 = nn.Linear(L0, L1, bias=False)
        torch.nn.init.xavier_uniform_(self.L1.weight)
        self.bn1 = nn.BatchNorm1d(L1, momentum=parameter_momentum)
        torch.nn.init.ones_(self.bn1.weight)

        self.L2 = nn.Linear(L1, L2, bias=False)
        torch.nn.init.xavier_uniform_(self.L2.weight)
        self.bn2 = nn.BatchNorm1d(L2, momentum=parameter_momentum)
        torch.nn.init.ones_(self.bn2.weight)

        self.L3 = nn.Linear(L2, L3, bias=False)
        torch.nn.init.xavier_uniform_(self.L3.weight)
        self.bn3 = nn.BatchNorm1d(L3, momentum=parameter_momentum)
        torch.nn.init.ones_(self.bn3.weight)

        self.L4 = nn.Linear(L3, L4, bias=False)
        torch.nn.init.xavier_uniform_(self.L4.weight)
        self.bn4 = nn.BatchNorm1d(L4, momentum=parameter_momentum)
        torch.nn.init.ones_(self.bn4.weight)

        # self.L5 = nn.Linear(L4, L5, bias=False)
        # torch.nn.init.xavier_uniform_(self.L4.weight)
        # self.bn5 = nn.BatchNorm1d(L5, momentum=parameter_momentum)
        # torch.nn.init.ones_(self.bn5.weight)
        #
        # self.L6 = nn.Linear(L5, L6, bias=False)
        # torch.nn.init.xavier_uniform_(self.L4.weight)
        # self.bn6 = nn.BatchNorm1d(L6, momentum=parameter_momentum)
        # torch.nn.init.ones_(self.bn6.weight)

        self.L5 = nn.Linear(L4, 1, bias=True)
        torch.nn.init.xavier_uniform_(self.L5.weight)
        torch.nn.init.zeros_(self.L5.bias)


    def forward(self, x):
        x = self.L1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.L2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.L3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.L4(x)
        x = self.bn4(x)
        x = F.relu(x)

        # x = self.L5(x)
        # x = self.bn5(x)
        # x = F.relu(x)
        #
        # x = self.L6(x)
        # x = self.bn6(x)
        # x = F.relu(x)

        x = self.L5(x)
        return x



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=100):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(192)
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, 192, kernel_size=1)
        self.bn8 = nn.BatchNorm2d(192)
        self.conv9 = nn.Conv2d(192, 10, kernel_size=1)
        self.bn9 = nn.BatchNorm2d(10)
        self.l1 = nn.Linear(640, 1000)
        self.l2 = nn.Linear(1000, 1000)
        self.l3 = nn.Linear(1000, 1)

        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = x.view(-1, 640)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        return x
if __name__ == '__main__':
    x = torch.zeros((1, 3, 32, 32))
    model = CNN()
    print(model(x))
