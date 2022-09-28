#import torch
import torch
import torch.nn as nn
import torch.nn.functional as F



class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        #FC layer blocks
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        #dropout
        self.drop = nn.Dropout(0.3)
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, out_features=2),# binary classification
        )

    def forward(self, x):
        #x_ = x.view(-1) # input size:
        out = self.fc1(x)
        out = self.drop(F.relu(out))
        out = self.fc2(out)
        out = self.drop(F.relu(out))
        out = self.fc3(out)
        out = self.drop(F.relu(out))
        out = self.fc4(out)
        out = self.drop(F.relu(out))
        out = self.fc5(out)
        out = self.drop(F.relu(out))
        out = self.fc6(out)
        out = self.drop(F.relu(out))
        out = self.fc7(out)
        out = self.drop(F.relu(out))

        out = self.classifier(out)

        return out
