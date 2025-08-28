# CNN模型定义

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        # 卷积层 [batch, 1, 28, 28]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 输入通道1，输出32，3x3卷积，填充1，32个filter提取32个特征图
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 输入32，输出64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 输入64，输出64

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 64)         # 把 64 * 7 * 7 的tensor拉成64维度的向量
        self.dropout = nn.Dropout(0.5)                          # 随机丢弃 50% 的神经元，防止过拟合
        self.fc2 = nn.Linear(64, 10)      # 把 64 维度进成 10 维

        # 输入: [batch, 1, 28, 28]
        # conv1 + pool1: [batch, 32, 14, 14] (第一次池化)
        # conv2 + pool2: [batch, 64, 7, 7] (第二次池化)
        # conv3: [batch, 64, 7, 7] (只有卷积，没有池化)

    def forward(self, x):
        # 卷积 -> ReLU -> 池化
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        # 展平
        # 改变张量形状：从4D卷积输出 [batch_size, 64, 7, 7] 变为2D矩阵 [batch_size, 3136]
        # 为全连接层准备输入：全连接层需要2D输入 (batch_size, features)
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