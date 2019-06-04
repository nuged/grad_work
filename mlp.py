import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch


class Net(nn.Module):
    def __init__(self, inputWidth, maxCategoryCount):
        super(Net, self).__init__()
        torch.manual_seed(52)
        self.fc1 = nn.Linear(inputWidth, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, maxCategoryCount)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, inputWidth, hiddenSize):
        super(AutoEncoder, self).__init__()
        torch.manual_seed(52)
        self.fc1 = nn.Linear(inputWidth, 1024)
        self.fc2 = nn.Linear(1024, hiddenSize)
        self.fc3 = nn.Linear(hiddenSize, 1024)
        self.fc4 = nn.Linear(1024, inputWidth)

        self.hidden = torch.zeros(hiddenSize, device='cuda')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        self.hidden = x
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

if __name__ == '__main__':
    ae = AutoEncoder(256, 32)
    ae.cuda()

    data = torch.rand((1, 256), device='cuda')
    criterion = nn.MSELoss()
    opt = optim.Adam(ae.parameters(), lr=1e-3)

    for i in range(1000):
        opt.zero_grad()

        out = ae(data)
        loss = criterion(out, data)
        print loss.item()
        loss.backward()
        opt.step()

    print data
    print ae(data)