# 训练程序

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import DigitDataset, get_data_loaders
from model import DigitCNN, save_model


def train_model(epochs=30, batch_size=32, learning_rate=0.001):
    """训练模型"""
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # 创建模型
    model = DigitCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练过程
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()  # set the model in training mode
        running_loss = 0.0  # 每一个epoch对应一个running_loss
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad() # 如果不清零，每个batch的梯度会不断累加
            loss.backward()   # 反向传播
            optimizer.step()  # 参数更新

            running_loss += loss.item()  # 把当前batch的loss累加到所有batch的损失总和
            _, predicted = torch.max(outputs.data, 1)  # 最大值的位置索引（这就是预测的类别标签)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 统计当前batch中正确预测的样本数量

        train_loss = running_loss / len(train_loader)  # 当前epoch中平均每个batch的损失值
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)  # 不同epoch之间进行对比，可以验证是否在收敛
        train_accuracies.append(train_accuracy)

        # 测试阶段
        model.eval()  # set the model in testing mode
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # 不干预模型训练
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

    # 保存模型
    save_model(model)

    # 绘制训练曲线
    plot_training_curve(train_losses, test_losses, train_accuracies, test_accuracies)

    return model, train_losses, test_losses, train_accuracies, test_accuracies


def plot_training_curve(train_losses, test_losses, train_accuracies, test_accuracies):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.title('损失曲线')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='训练准确率')
    plt.plot(test_accuracies, label='测试准确率')
    plt.title('准确率曲线')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()


if __name__ == "__main__":
    train_model()