# 主程序入口

from data_collection import collect_data
from train import train_model
from predict import predict_random_samples, real_time_prediction
from dataset import get_data_loaders
from model import load_model
from utils import check_data_balance


def main():
    """主程序"""
    while True:
        print("\n" + "=" * 50)
        print("           PyTorch数字识别项目")
        print("=" * 50)
        print("1. 收集数据（绘制数字样本）")
        print("2. 训练模型")
        print("3. 测试模型")
        print("4. 实时识别")
        print("5. 检查数据分布")
        print("6. 退出")
        print("=" * 50)

        choice = input("请选择操作 (1-6): ").strip()

        if choice == '1':
            collect_data()

        elif choice == '2':
            try:
                epochs = int(input("请输入训练轮次 (默认30): ") or "30")
                batch_size = int(input("请输入批大小 (默认32): ") or "32")
                train_model(epochs=epochs, batch_size=batch_size)
            except Exception as e:
                print(f"训练错误: {e}")

        elif choice == '3':
            try:
                model = load_model()
                _, test_loader = get_data_loaders()
                predict_random_samples(model, test_loader)
            except Exception as e:
                print(f"测试错误: {e}")

        elif choice == '4':
            try:
                model = load_model()
                real_time_prediction(model)
            except Exception as e:
                print(f"实时识别错误: {e}")

        elif choice == '5':
            check_data_balance()

        elif choice == '6':
            print("感谢使用！")
            break

        else:
            print("无效选择！")


if __name__ == "__main__":
    main()