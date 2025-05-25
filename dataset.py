"""将基本的图像处理工具包放在这里"""
import os.path
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

class Mypoint:
    x: int #该点的x像素坐标
    y: int #该点的y像素坐标

    def get_num(self):
        return int(self.x), int(self.y)

    def __init__(self, x, y):
        self.x = x
        self.y = y

class MyObb:
    """定义一个矩形最基本类"""
    Id: int #每个框的信息
    x_mid: int #该矩形的中心x（pixel）坐标下同
    y_mid: int #该矩形的中心y坐标
    points: [Mypoint] #该矩形的角点(左上和右下)
    h: int #该矩形的长
    w: int #该矩形的宽
    masks: [] #把一些mask可以放到这里来

    def get_points(self):
        """获取该矩形的左上和右下两个点"""
        x1 = self.x_mid - 0.5 * self.w
        x2 = self.x_mid + 0.5 * self.w
        y1 = self.y_mid - 0.5 * self.h
        y2 = self.y_mid + 0.5 * self.h
        self.points.append(Mypoint(x1, y1))
        self.points.append(Mypoint(x2, y2))

    def output_points(self):
        """输出该矩形的左上和右下两个点"""
        x1 = self.x_mid - 0.5 * self.w
        x2 = self.x_mid + 0.5 * self.w
        y1 = self.y_mid - 0.5 * self.h
        y2 = self.y_mid + 0.5 * self.h
        return int(x1), int(y1), int(x2), int(y2)

    def __init__(self, id, x1, y1, w, l, mask, width=1, height=1):
        """从yolo的格式来将其初始化为像素坐标"""
        self.Id = id
        self.x_mid = int(x1 * width)
        self.y_mid = int(y1 * height)
        self.h = int(l * height)
        self.w = int(w * width)
        self.points = []
        self.masks = mask
        self.get_points()

    def set_Y(self, I: int, H: int):
        self.y_mid += I * H
        self.points = []
        self.get_points()

    def IOU(self, other) -> float:
        """计算两个矩形的IOU(交并比)"""
        if isinstance(other, MyObb):
            area_self = self.w * self.h
            area_other = other.w * other.h
            w = self.w / 2 + other.w / 2 - abs(self.x_mid - other.x_mid)
            h = self.h / 2 + other.h / 2 - abs(self.y_mid - other.y_mid)
            if w <= 0 or h <= 0:
                return 0
            else:
                area = w*h
                return float(area/(area_other + area_self - area))

    def renew_Obb(self, i: int, step: int):
        # 重构rebar
        self.x_mid += i * step
        self.points = []
        self.get_points()
        mask = []
        for m in self.masks[0]:
            x, y = m
            x += i * step
            mask.append([x, y])
        self.masks = [np.array(mask)]

    def catch_rebar(self, other, i: int, step: int):
        # 将同一个id的rebar的Myobb将其合并成一个
        if isinstance(other, MyObb):
            x1 = self.x_mid - 0.5 * self.w
            x2 = other.x_mid + 0.5 * other.w + i * step
            y1 = self.y_mid - 0.5 * self.h
            y2 = other.y_mid + 0.5 * other.h
            w = abs(x1 - x2); h = abs(y2 - y1)
            x_mid = (x1 + x2) // 2; y_mid = (y1 + y2) // 2
            mask = self.masks
            masks = []
            for m in other.masks[0]:
                x, y = m
                x += i * step
                masks.append([x, y])
            masks = np.array(masks)
            mask.append(masks)
            return MyObb(self.Id, x_mid, y_mid, w, h, mask)

class Count():
    w: int
    h: int
    step: int # 对于视频的步长
    name: str # 保存文件的name
    edge: bool # 是否是最后一行图片
    spath = r"E:\reaserching_com\final_reserch\counter_rebar\data\temp.mp4"
    weights_path: str # 权重地址
    imgs: [[]] # 最初的图像信息
    result_imgs: [[]] # 生成处理的图像信息
    masks: [] # 图像的掩码信息
    obbs: [] # 图像的obb框信息
    rebars_num: [int] # 每帧图像的rebar数量
    rebar_num_now: int # 当前帧的rebar数量
    rebar_dl: int #可能回到下一段的rebars数量（重复计数）

    def create_vedio(self, srouce, spath):
        frame_rate = 20
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(spath, fourcc, frame_rate, (self.h, self.w))
        for img in srouce:
            video_writer.write(img)
        video_writer.release()

    def __init__(self, imgs, path, name, step, edge):
        self.step = step
        self.edge = edge
        self.name = name
        self.imgs = imgs
        self.rebar_dl = 0
        self.rebar_num_now = 0
        self.w, self.h, _ = imgs[0].shape
        self.create_vedio(self.imgs, self.spath)
        self.weights_path = path
        self.rebars_num = []
        self.result_imgs = []
        self.obbs = []
        model = YOLO(self.weights_path)
        results = model.track(source=self.spath, conf=0.55, iou=0.5, save=True,
                              project=r"E:\reaserching_com\final_reserch\counter_rebar\runs", name="temp", show=False, stream=True) #可以试一下搞成track
        for result in results:
            Boxes = []
            if result != None and result.masks != None:
                boxes = result.boxes
                masks = result.masks.cpu().xy
                for box, mask in zip(boxes, masks):
                    if box!= None and box.id != None:
                        x, y, w, h = box.xywh.cpu()[0]
                        id = box.id.int().cpu().tolist()[0]
                    else:
                        continue
                    Boxes.append(MyObb(id, x, y, w, h, [mask]))
            else:
                Boxes.append([])
            self.obbs.append(Boxes)

    def Count_rebar(self):
        # 用来对目标进行计数
        Ids = []; Ids_dl = []
        if len(self.imgs) == len(self.obbs):
            for i in range(0, len(self.imgs), 1):
                if i == 0:
                    Obbs = self.obbs[i]
                    for obb in Obbs:
                        if isinstance(obb, MyObb):
                            sign = 1
                            for obb1 in Obbs:
                                if obb1.Id != obb.Id and obb1.IOU(obb) > 0.2:
                                    sign = 0
                            if sign == 1:
                                self.rebar_num_now += 1
                            Ids.append(obb.Id)
                            if not self.edge:
                                if obb.Id in Ids_dl:
                                    if (obb.y_mid + obb.h / 2) < self.w - 50:
                                        self.rebar_dl -= 1
                                        Ids_dl.remove(obb.Id)
                                if (obb.y_mid + obb.h / 2) >= self.w - 50:
                                    if obb.Id not in Ids_dl:
                                        self.rebar_dl += 1
                                        Ids_dl.append(obb.Id)
                    self.rebars_num.append(self.rebar_num_now)
                else:
                    for obb in self.obbs[i]:
                        if isinstance(obb, MyObb):
                            if obb.Id not in Ids:
                                for obb1 in self.obbs[i]:
                                    if obb1.Id != obb.Id and obb1.IOU(obb) > 0.2:
                                        sign = 0
                                if sign == 1:
                                    self.rebar_num_now += 1
                                Ids.append(obb.Id)
                            if not self.edge:
                                if obb.Id in Ids_dl:
                                    if (obb.y_mid + obb.h / 2) < self.w - 50:
                                        self.rebar_dl -= 1
                                        Ids_dl.remove(obb.Id)
                                if (obb.y_mid + obb.h / 2) >= self.w - 50:
                                    if obb.Id not in Ids_dl:
                                        self.rebar_dl += 1
                                        Ids_dl.append(obb.Id)
                    self.rebars_num.append(self.rebar_num_now)

    def Link_rebar(self):
        # 将这些相同的rebar来进行合并
        Ids = []
        for i in range(len(self.obbs)):
            Obbs = self.obbs[i]
            if i == 0:
                Obbs_or = []
                for obb in Obbs:
                    Obbs_or.append(obb)
                    Ids.append(obb.Id)
            else:
                Obbs_new = []
                for o in Obbs_or:
                    sign = 0
                    for obb in Obbs:
                        if o.Id == obb.Id:
                            Obbs_new.append(o.catch_rebar(obb, i, self.step))
                            sign = 1
                    if sign == 0:
                        Obbs_new.append(o)
                for obb in Obbs:
                    if obb.Id not in Ids:
                        obb.renew_Obb(i, self.step)
                        Obbs_new.append(obb)
                        Ids.append(obb.Id)
                Obbs_or = Obbs_new
        return Obbs_or

    def create_mp4_imgs(self):
        cap = cv2.VideoCapture(r"runs/temp/temp.avi")
        i = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"rebar num:{self.rebars_num[i]}"
                font_scale = 1.2
                font_thickness = 2
                text_position = (50, 50)
                text_color = (0, 255, 255)  # BGR格式颜色

                # 添加文字到图片
                cv2.putText(frame, text, text_position, font, font_scale, text_color, font_thickness)
                i += 1
                self.result_imgs.append(frame)
            else:
                break
        cap.release()

    def delete_temp(self):
        # 删除temp的所有文件
        folder_path = 'runs/temp'
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder {folder_path} deleted successfully.")
        else:
            print(f"Folder {folder_path} does not exist.")
        os.remove(self.spath)

    def process(self):
        self.Count_rebar()
        self.create_mp4_imgs()
        path = os.path.join("result", self.name)
        self.create_vedio(self.result_imgs, path)
        self.delete_temp()

    def __str__(self):
        return f"The rebar num:{self.rebar_num_now}"
