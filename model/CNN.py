import torch.nn as nn
import torch.nn.functional as F

class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        # 输入：1 x 28 x 28（灰度图）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输出：32 x 28 x 28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 输出：64 x 28 x 28
        self.pool = nn.MaxPool2d(2, 2)  # 池化后：64 x 14 x 14
        self.dropout1 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)  # 输出类别数为10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)

        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)  # log_softmax适用于nll_loss
