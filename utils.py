# 工具函数

import os


def check_data_balance(data_dir='digit_data'):
    """检查数据平衡性"""
    print("\n数据分布统计:")
    total = 0
    for i in range(10):
        digit_dir = os.path.join(data_dir, str(i))
        if os.path.exists(digit_dir):
            count = len([f for f in os.listdir(digit_dir) if f.endswith('.png')])
            print(f"数字 {i}: {count} 个样本")
            total += count
        else:
            print(f"数字 {i}: 0 个样本")

    print(f"\n总样本数: {total}")
    return total


def get_class_distribution(data_dir='digit_data'):
    """获取类别分布"""
    distribution = {}
    for i in range(10):
        digit_dir = os.path.join(data_dir, str(i))
        if os.path.exists(digit_dir):
            count = len([f for f in os.listdir(digit_dir) if f.endswith('.png')])
            distribution[i] = count
        else:
            distribution[i] = 0
    return distribution