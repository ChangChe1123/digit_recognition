# 自定义数据集类

import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch


class DigitDataset(Dataset):
    def __init__(self, data_dir='digit_data', transform=None, train=True, test_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        self._load_data()

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.images, self.labels, test_size=test_size,
            random_state=random_state, stratify=self.labels
        )

        if train:
            self.images = X_train
            self.labels = y_train
        else:
            self.images = X_test
            self.labels = y_test

    def _load_data(self):
        """加载数据"""
        for label in range(10):
            label_dir = os.path.join(self.data_dir, str(label))
            if not os.path.exists(label_dir):
                print(f"警告: 目录 {label_dir} 不存在，跳过数字 {label}")
                continue

            for image_file in os.listdir(label_dir):
                if image_file.endswith('.png'):
                    image_path = os.path.join(label_dir, image_file)
                    #灰度处理，减少计算量
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if image is None:
                        continue

                    # 确保图像是28x28，便于神经网络处理
                    if image.shape != (28, 28):
                        image = cv2.resize(image, (28, 28))

                    # 归一化像素值并转换为float32，加快收敛
                    image = image.astype(np.float32) / 255.0

                    self.images.append(image)
                    self.labels.append(label)

        if len(self.images) == 0:
            raise ValueError("没有找到任何图像数据！请先运行数据收集程序。")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 转换为PyTorch tensor
        image = torch.from_numpy(image).unsqueeze(0)  # 添加通道维度
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loaders(batch_size=32, test_size=0.2):
    """获取数据加载器"""
    train_dataset = DigitDataset(train=True, test_size=test_size)
    test_dataset = DigitDataset(train=False, test_size=test_size)

    #避免模型记忆顺序
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader