# CNN模型定义

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 64)  # 经过3次池化，28/2/2/2=3.5->3，但实际是7*7
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # 卷积 -> ReLU -> 池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        # 展平
        x = x.view(-1, 64 * 7 * 7)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def create_model():
    """创建并返回模型"""
    model = DigitCNN()
    return model


def save_model(model, path='digit_cnn.pth'):
    """保存模型"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")


def load_model(path='digit_cnn.pth'):
    """加载模型"""
    model = DigitCNN()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model