import os
import tifffile
import cv2
import numpy as np

class OpenCVReader1:
    def __init__(self, image_dir, color_mode):
        self.image_dir = image_dir
        self.color_mode = color_mode
        assert color_mode in ["RGB", "BGR", "GRAY"], f"{color_mode} not supported"  #图像的格式
        if color_mode != "BGR":
            self.cvt_color = getattr(cv2, f"COLOR_BGR2{color_mode}")
        else:
            self.cvt_color = None

    def __call__(self, filename, is_mask=False): #filename: bagel/train/good/rgb/221.png
        filename = os.path.join(self.image_dir, filename)
        assert os.path.exists(filename), filename
        if is_mask: 
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            return img
        # img = cv2.imread(filename, cv2.IMREAD_COLOR) #原来 is_mask=False 则使用 cv2.imread 以彩色模式 (cv2.IMREAD_COLOR) 读取图像。
        img = tifffile.imread(filename) #新加
        if img.dtype == np.float32 or img.dtype == np.int32: #新加
            img = (img * 255).astype(np.uint8)  #新加    假设是 0-1 的浮点型图像
        # if self.color_mode != "BGR":
        #     img = cv2.cvtColor(img, self.cvt_color)
        return img


def build_image_reader1(cfg_reader):
    if cfg_reader["type"] == "opencv":
        return OpenCVReader1(**cfg_reader["kwargs"])
    else:
        raise TypeError("no supported image reader type: {}".format(cfg_reader["type"]))
    

class OpenCVReader:
    def __init__(self, image_dir, color_mode):
        self.image_dir = image_dir
        self.color_mode = color_mode
        assert color_mode in ["RGB", "BGR", "GRAY"], f"{color_mode} not supported"  #图像的格式
        if color_mode != "BGR":
            self.cvt_color = getattr(cv2, f"COLOR_BGR2{color_mode}")
        else:
            self.cvt_color = None

    def __call__(self, filename, is_mask=False): #filename: bagel/train/good/rgb/221.png
        filename = os.path.join(self.image_dir, filename)
        assert os.path.exists(filename), filename
        if is_mask: 
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            return img
        img = cv2.imread(filename, cv2.IMREAD_COLOR) # is_mask=False 则使用 cv2.imread 以彩色模式 (cv2.IMREAD_COLOR) 读取图像。
        if self.color_mode != "BGR":
            img = cv2.cvtColor(img, self.cvt_color)
        return img

def build_image_reader(cfg_reader):
    if cfg_reader["type"] == "opencv":
        return OpenCVReader(**cfg_reader["kwargs"])
    else:
        raise TypeError("no supported image reader type: {}".format(cfg_reader["type"]))
