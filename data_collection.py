# 数据收集程序

import cv2
import numpy as np
import os


def create_data_directories():
    """创建数据目录"""
    if not os.path.exists('digit_data'):
        os.makedirs('digit_data')
        for i in range(10):
            os.makedirs(f'digit_data/{i}')
    print("数据目录创建完成")


def collect_data():
    """数据收集主函数"""
    print("正在启动数据收集程序...")
    print("使用说明:")
    print("1. 按0-9键选择要绘制的数字")
    print("2. 用鼠标在窗口中绘制数字")
    print("3. 按空格键保存当前绘制的数字")
    print("4. 按c键清空画布")
    print("5. 按ESC键退出程序")

    # 创建数据目录
    create_data_directories()

    # 初始化绘图窗口
    drawing = False
    ix, iy = -1, -1
    img = np.zeros((280, 280, 3), np.uint8)

    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, ix, iy, img
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(img, (x, y), 10, (255, 255, 255), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(img, (x, y), 10, (255, 255, 255), -1)

    cv2.namedWindow('数字绘制 - 按0-9选择数字，空格保存，ESC退出')
    cv2.setMouseCallback('数字绘制 - 按0-9选择数字，空格保存，ESC退出', draw_rectangle)

    current_digit = 0
    count = 0

    while True:
        cv2.imshow('数字绘制 - 按0-9选择数字，空格保存，ESC退出', img)
        k = cv2.waitKey(1) & 0xFF

        # 按0-9选择当前要绘制的数字
        if 48 <= k <= 57:  # 0-9的ASCII码
            current_digit = k - 48
            print(f"当前数字: {current_digit}")

        # 按空格键保存图像
        elif k == 32:
            # 调整大小为28x28并保存
            resized = cv2.resize(img, (28, 28))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # 统计当前数字的样本数量
            digit_dir = f"digit_data/{current_digit}"
            existing_files = len([f for f in os.listdir(digit_dir) if f.endswith('.png')])

            filename = f"{digit_dir}/{existing_files + 1}.png"
            cv2.imwrite(filename, gray)
            print(f"保存: {filename} (数字{current_digit}的第{existing_files + 1}个样本)")

            # 清空画布
            img = np.zeros((280, 280, 3), np.uint8)

        # 按c键清空画布
        elif k == ord('c'):
            img = np.zeros((280, 280, 3), np.uint8)
            print("画布已清空")

        # 按ESC退出
        elif k == 27:
            break

    cv2.destroyAllWindows()
    print("数据收集完成！")


if __name__ == "__main__":
    collect_data()