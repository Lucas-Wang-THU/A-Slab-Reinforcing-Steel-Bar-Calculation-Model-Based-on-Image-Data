import cv2
import os
import shutil
import time
import pandas as pd
import numpy as np
from dataset import Count, MyObb
from get_calibration import get_calibration

#weights的路径
weights_path = "weights/Yolov11_best_seg.pt"
#检测图片路径
img_path = r"data/DJI_0823.jpg"

def cut_lines(img: str, num: int, rotate = False) -> []:
    """将图像分割成num个小条进行处理"""
    img = cv2.imread(img)
    imgs = []
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        w, h, _ = img.shape
        dim_h = h // num
        h_now = 0
        for i in range(num):
            img_e = img[:, h_now: h_now + dim_h].copy()
            imgs.append(cv2.rotate(img_e, cv2.ROTATE_90_CLOCKWISE))
            h_now = h_now + dim_h
    else:
        w, h, _ = img.shape
        dim_h = h // num
        h_now = 0
        for i in range(num):
            img_e = img[:, h_now: h_now + dim_h].copy()
            imgs.append(cv2.rotate(img_e, cv2.ROTATE_90_CLOCKWISE))
            h_now = h_now + dim_h
    return imgs

def get_dataset(path:str, numh: int, numv: int) -> []:
    """把图片切分后成为一个数据库"""
    imgs_h = cut_lines(path, numh)
    imgs_v = cut_lines(path, numv, rotate=True)
    return imgs_h, imgs_v

def create_imgs(img:[]) -> [[]]:
    """把这一串数据改写成一个从右向左的数据处理"""
    imgs = []
    height, width, colors = img.shape
    video_width = width // 10
    video_height = height
    step_size = width // 100
    for x in range(0, (width - video_width), step_size):
        column_image = img[0:video_height, x:x + video_width].copy()
        column_image = cv2.copyMakeBorder(column_image, 0, 0, 20, 20, cv2.BORDER_CONSTANT, (0, 0, 0))
        imgs.append(column_image)
    return imgs, step_size

def plotting_line(img_l:[], obbs:[], color, output_path:str) -> []:
    # 将obbs的可视化表示在img_l中
    #img = cv2.rotate(img_l, cv2.ROTATE_90_CLOCKWISE)
    background = img_l.copy()  # 创建背景图像副本
    for obb in obbs:
        if isinstance(obb, MyObb):
            x1, y1, x2, y2 = obb.output_points()
            P1 = (x1, y1); P2 = (x2, y2)
            cv2.rectangle(img_l, P1, P2, color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            cv2.putText(img_l, 'rebar', P1, font, 2, color, 2)
            for mask in obb.masks:
                if len(mask) > 4:
                    contour = np.array(mask, dtype=np.int32)  # 确保轮廓是整数类型
                    cv2.drawContours(background, [contour], -1, color, thickness=cv2.FILLED)
    # 设置透明度，例如50%透明度
    alpha = 0.4
    beta = 1 - alpha  # 因为1-alpha是原始图像的权重
    gamma = 0  # 在这里我们不需要额外的gamma值因为它是加到输出中的常数项，这里设置为0

    # 混合图像
    blended = cv2.addWeighted(background, alpha, img_l, beta, gamma)
    cv2.imwrite(output_path, blended)
    print(f'Saved: {output_path}')
    return blended

def link_img_line(imgs:[]):
    # 将图片进行一个连接
    for i in range(len(imgs)):
        if i == 0:
            img_or = imgs[0]
        else:
            img_or = cv2.vconcat([img_or, imgs[i]])
    return img_or

def get_result_pic(img_h, img_v, num_h, num_v, output_path):
    img_h = cv2.rotate(img_h, cv2.ROTATE_90_CLOCKWISE)
    # 设置透明度，例如50%透明度
    alpha = 0.5
    beta = 1 - alpha  # 因为1-alpha是原始图像的权重
    gamma = 0  # 在这里我们不需要额外的gamma值因为它是加到输出中的常数项，这里设置为0

    # 混合图像
    blended = cv2.addWeighted(img_h, alpha, img_v, beta, gamma)
    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
    cv2.putText(blended, f"Rebars_V:{num_h} Rebars_H:{num_v}",
                (20, 200), font, 5, (0, 255, 255), 5)
    cv2.imwrite(output_path, blended)
    print(f'Saved: {output_path}')

def reset_Y(Obbs: [MyObb], I:int, H:int) -> [MyObb]:
    obbs = []
    for Obb in Obbs:
        Obb.set_Y(I, H)
        obbs.append(Obb)
    return obbs

def output_excel(Obbs_h: [MyObb], Obbs_v: [MyObb], R:int, rate_h: float, rate_v: float, filename: str):
    """将整合的钢筋存储为excel"""
    data = {'x1':[],
            'y1':[],
            'z1':[],
            'x2': [],
            'y2': [],
            'z2': [],
            "D":[],
            }
    for Obb in Obbs_h:
        if isinstance(Obb, MyObb):
            P1, P2 = Obb.points
            X1, Y1 = P1.get_num(); X2, Y2 = P2.get_num()
            Y1 += Obb.h // 2; Y2 -= Obb.h // 2
            if X1 != X2 and Y1 != Y2:
                data["x1"].append(Y1*rate_h); data["y1"].append(X1*rate_v)
                data["x2"].append(Y2*rate_h); data["y2"].append(X2*rate_v)
                data["z1"].append(0); data["z2"].append(0)
                data["D"].append(R)
    for Obb in Obbs_v:
        if isinstance(Obb, MyObb):
            P1, P2 = Obb.points
            X1, Y1 = P1.get_num(); X2, Y2 = P2.get_num()
            Y1 += Obb.h // 2; Y2 -= Obb.h // 2
            if X1 != X2 and Y1 != Y2:
                data["x1"].append(X1*rate_h); data["y1"].append(Y1*rate_v)
                data["x2"].append(X2*rate_h); data["y2"].append(Y2*rate_v)
                data["z1"].append(0); data["z2"].append(0)
                data["D"].append(R)
    df = pd.DataFrame(data)
    # 将DataFrame导出到Excel文件
    df.to_excel(filename, index=False)

def _Main(img_path: str):
    folder_path = 'runs/temp'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder {folder_path} deleted successfully.")
    else:
        print(f"Folder {folder_path} does not exist.")
    print("Set horizon rate")
    rate_h = get_calibration(img_path)
    print("Set vertial rate")
    rate_v = get_calibration(img_path)
    start_time = time.time()
    file_name = img_path.split('.')[0].split('/')[-1]
    imgs_h, imgs_v = get_dataset(img_path, 5, 5)
    rebar_num_h = 0;
    rebar_num_v = 0
    H_dl = 0;
    W_dl = 0
    result_h = [];
    result_v = []
    Obbs_h = [];
    Obbs_v = []
    for i in range(len(imgs_h)):
        imgs, step = create_imgs(imgs_h[i])
        name = file_name + f"_h_{i}.mp4"
        print(name)
        if i != len(imgs_h) - 1:
            counter = Count(imgs, weights_path, name, step, False)
        else:
            counter = Count(imgs, weights_path, name, step, True)
        counter.process()
        Obbs = counter.Link_rebar()
        outpath = os.path.join('result', file_name + f"_h_{i}" + '.jpg')
        result_h.append(plotting_line(imgs_h[i], Obbs, (0, 225, 0), outpath))
        rebar_num_h += counter.rebar_num_now
        H_dl += counter.rebar_dl
        obbs = reset_Y(Obbs, i, counter.w)
        for obb in obbs:
            Obbs_h.append(obb)
        print(counter)
    for i in range(len(imgs_v)):
        imgs, step = create_imgs(imgs_v[i])
        name = file_name + f"_v_{i}.mp4"
        print(name)
        if i != len(imgs_v) - 1:
            counter = Count(imgs, weights_path, name, step, False)
        else:
            counter = Count(imgs, weights_path, name, step, True)
        counter.process()
        Obbs = counter.Link_rebar()
        outpath = os.path.join('result', file_name + f"_v_{i}" + '.jpg')
        result_v.append(plotting_line(imgs_v[i], Obbs, (0, 0, 225), outpath))
        rebar_num_v += counter.rebar_num_now
        W_dl += counter.rebar_dl
        obbs = reset_Y(Obbs, i, counter.w)
        for obb in obbs:
            Obbs_v.append(obb)
        print(counter)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print(f"本区域钢筋共有:\n初始纵向钢筋:{rebar_num_h}\n初始横向钢筋:{rebar_num_v}"
          f"\n重复纵向钢筋:{H_dl}\n重复横向钢筋:{W_dl}"
          f"\n处理后纵向钢筋:{rebar_num_h - H_dl}\n处理后横向钢筋:{rebar_num_v - W_dl}")
    txt_path = os.path.join("result", file_name + "_result.txt")
    with open(txt_path, "w") as f:
        f.write(f"Elapsed time: {elapsed_time} seconds")
        f.write(f"\n本区域钢筋共有:\n初始纵向钢筋:{rebar_num_h}\n初始横向钢筋:{rebar_num_v}"
                f"\n重复纵向钢筋:{H_dl}\n重复横向钢筋:{W_dl}"
                f"\n处理后纵向钢筋:{rebar_num_h - H_dl}\n处理后横向钢筋:{rebar_num_v - W_dl}")
    f.close()
    R_h = link_img_line(result_h)
    R_v = link_img_line(result_v)
    s_path = os.path.join("result", file_name + "_result.png")
    get_result_pic(R_h, R_v, rebar_num_h - H_dl, rebar_num_v - W_dl, s_path)
    while True:
        R = input("请输入钢筋直径（mm）:")
        if R.isdigit():
            R = float(R)
            break
    s_path = os.path.join("result", file_name + "_result.xlsx")
    output_excel(Obbs_h, Obbs_v, R, rate_h, rate_v, s_path)

if __name__ == "__main__":
    _Main(img_path)