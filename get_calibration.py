import cv2
import math
"""获取标定的数据"""

def _dis(P1, P2):
    x1, y1 = P1
    x2, y2 = P2
    return float(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

def get_calibration(imgpath: str) -> []:
    # 创建一个空列表，用于存储坐标信息
    coordinates = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击事件
            print(f"Mouse position: ({x}, {y})")
            # 在图像上绘制一个填充的红色圆点
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(image, f'({x},{y})', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            coordinates.append((x, y))
            if len(coordinates) == 2:
                cv2.line(image, coordinates[0], coordinates[1], color=(0, 0, 255), thickness=2)


    image = cv2.imread(imgpath)
    w, h, _ = image.shape
    # 创建一个窗口
    cv2.namedWindow('Point Coordinates')
    # 设置鼠标回调函数
    cv2.setMouseCallback('Point Coordinates', mouse_callback)
    while True:
        cv2.imshow('Point Coordinates', image)
        k = cv2.waitKey(1) & 0xFF
        if k == 13 and len(coordinates) == 2:  # 如果按下回车Enter键，退出显示
            P1 = coordinates[0]; P2 = coordinates[1]
            cv2.destroyAllWindows() # 关闭窗口
            break
        if len(coordinates) > 2:
            coordinates = []
            print("请重新绘制线段")
            cv2.imshow('Point Coordinates', image)
    while True:
        d = input("请输入线段的真实长度:")
        if d.isdigit():
            d = float(d)
            break
    line = _dis(P1, P2)
    l_c = d / line
    return l_c

if __name__ == "__main__":
    path = r"data/test.jpg"
    l_c = get_calibration(path)
    print(l_c)