# 预测程序

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import load_model
from dataset import DigitDataset, get_data_loaders


def predict_random_samples(model, test_loader, num_samples=5):
    """预测随机样本"""
    model.eval()
    device = next(model.parameters()).device

    # 获取随机样本
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    with torch.no_grad():
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)  # softmax转变为概率分数

    # 显示结果
    plt.figure(figsize=(15, 3))
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        confidence = probabilities[i][predicted[i]].item() * 100
        plt.title(f'真实: {labels[i].item()}\n预测: {predicted[i].item()}\n置信度: {confidence:.1f}%')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()


def real_time_prediction(model):
    """实时预测"""
    import cv2

    device = next(model.parameters()).device
    model.eval()

    cap = cv2.VideoCapture(0)
    print("启动实时数字识别...按 'q' 键退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        normalized = resized.astype(np.float32) / 255.0

        # 转换为PyTorch tensor
        image_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)

        # 预测
        with torch.no_grad():
            output = model(image_tensor)
            probability = torch.softmax(output, dim=1)
            predicted = torch.argmax(output, dim=1).item()
            confidence = probability[0][predicted].item() * 100

        # 显示结果
        cv2.putText(frame, f'数字: {predicted}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'置信度: {confidence:.1f}%',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('实时数字识别', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = load_model()
    _, test_loader = get_data_loaders()
    predict_random_samples(model, test_loader)