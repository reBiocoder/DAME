import cv2
from os.path import join, abspath, dirname, split
import os
from glob import glob
import numpy as np
import time


class ImageWorkSpace:
    grayTime = 0
    norTime = 0
    grayNumber = 0
    norNumber = 0
    WorkPath = dirname(dirname(abspath(__file__)))
    OriginPath = join(join(WorkPath, 'image'), 'origin1')
    FinishPath = join(join(WorkPath, 'image'), 'finish_common')

    def file_name(self, file_dir):
        for root, dirs, files in os.walk(file_dir):
            for match in glob(join(root, "*.jpg")):  # 图片的绝对路径
                dir_path, filename = split(match)
                finish_path = dir_path.split(ImageWorkSpace.OriginPath)[1][
                              1:]  # .replace(" (", "_").replace(")", "")[1:]
                #print("####", finish_path)
                filename = filename.split('.')[0]
                #print("基础路径", filename)
                #print(match)
                #print("存在文件：", join(ImageWorkSpace.FinishPath, finish_path))
                self.separator(match, join(ImageWorkSpace.FinishPath, finish_path), filename)

    def separator(self, img_path, write_path, origin_filename):
        """
        :param origin_filename:
        :param write_path:
        :param img_path: 图像路径
        :return:
        """
        # final_path = join(write_path, origin_filename)  # black/mark/pm5-mark
        # os.makedirs(final_path, exist_ok=True)
        os.makedirs(write_path, exist_ok=True)
        tmp_digit = int(''.join(filter(str.isdigit, origin_filename))) - 1
        ######################
        src = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if "mark" in origin_filename:
            src = src[0:-20, 0:-20]
        cnt = 1
        num = 1
        sub_images = []
        sub_image_num = 3  # 3*3
        src_height, src_width = src.shape[0], src.shape[1]

        sub_height = src_height // sub_image_num
        sub_width = src_width // sub_image_num
        for j in range(sub_image_num):
            for i in range(sub_image_num):
                if j < sub_image_num - 1 and i < sub_image_num - 1:
                    image_roi = src[j * sub_height: (j + 1) * sub_height, i * sub_width: (i + 1) * sub_width, :]
                elif j < sub_image_num - 1:
                    image_roi = src[j * sub_height: (j + 1) * sub_height, i * sub_width:, :]
                elif i < sub_image_num - 1:
                    image_roi = src[j * sub_height:, i * sub_width: (i + 1) * sub_width, :]
                else:
                    image_roi = src[j * sub_height:, i * sub_width:, :]
                sub_images.append(image_roi)
        if 'mark' in origin_filename:
            for i, img in enumerate(sub_images):
                tmp_path = join(write_path, str(tmp_digit * 36 + i * 4 + 1) + '.jpg')
                cv2.imwrite(tmp_path, cv2.resize(img, (256, 256)))  # 切割后的初始图片
                cv2.imwrite(join(write_path, str(tmp_digit * 36 + (i * 4 + 2)) + '.jpg'),
                            cv2.resize(cv2.flip(img, 1), (256, 256)))  # 水平翻转
                cv2.imwrite(join(write_path, str(tmp_digit * 36 + i * 4 + 3) + '.jpg'),
                            cv2.resize(cv2.flip(img, 0), (256, 256)))  # 垂直翻转
                cv2.imwrite(join(write_path, str(tmp_digit * 36 + i * 4 + 4) + '.jpg'),
                            cv2.resize(self.rotate(tmp_path), (256, 256)))  # 旋转180
        else:
            for i, img in enumerate(sub_images):
                ImageWorkSpace.grayNumber += 4
                tmp_path = join(write_path, str(tmp_digit * 36 + i * 4 + 1) + '.jpg')
                cv2.imwrite(tmp_path, self.gray(cv2.resize(img, (256, 256))))  # 切割后的初始图片
                cv2.imwrite(join(write_path, str(tmp_digit * 36 + (i * 4 + 2)) + '.jpg'),
                            self.gray(cv2.resize(cv2.flip(img, 1), (256, 256))))  # 水平翻转
                cv2.imwrite(join(write_path, str(tmp_digit * 36 + i * 4 + 3) + '.jpg'),
                            self.gray(cv2.resize(cv2.flip(img, 0), (256, 256))))  # 垂直翻转
                cv2.imwrite(join(write_path, str(tmp_digit * 36 + i * 4 + 4) + '.jpg'),
                            self.gray(cv2.resize(self.rotate(tmp_path), (256, 256))))  # 旋转180
                # print("灰度时间", ImageWorkSpace.grayTime)
                # print("归一化时间", ImageWorkSpace.norTime)
                # print("次数", ImageWorkSpace.grayNumber)


    def get_red(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low_hsv = np.array([0, 43, 46])
        high_hsv = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
        result = image.copy()
        result[mask != 255] = (255, 255, 255)
        return result

    def rotate(self, img_path):
        img = cv2.imread(img_path, 1)
        cv2.imshow('src', img)
        imgInfo = img.shape
        height = imgInfo[0]
        width = imgInfo[1]
        deep = imgInfo[2]
        matRotate = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), 180, 1)  # 旋转变化矩阵
        '''
        参数1 必选参数。用于设置旋转中心点，点坐标为OpenCV图像坐标系下的坐标。
        参数2 必选参数。用于设置旋转的角度，单位为度。
        参数3 必选参数。用于设置缩放系数，即对旋转的图像进行缩放。
        '''
        dst = cv2.warpAffine(img, matRotate, (width, height))  # 旋转
        '''
        参数2 变换矩阵：是一个2行3列的矩阵，由这个矩阵决定是何种变换
        参数3 变换后输出图像的大小:(width,height)-->宽和高(自己规定)
        '''
        return dst

    def gaussian(self, img):
        """高斯过滤"""
        return cv2.GaussianBlur(img, (3, 3), 0.8)

    def gray(self, img):
        """灰度,增强对比度(归一化)"""
        # img = cv2.imread("pm1-mark.jpg", cv2.IMREAD_COLOR)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # a = time.time()
                img[i, j] = 0.3 * img[i, j, 0] + 0.59 * img[i, j, 1] + 0.11 * img[i, j, 2]  # 灰度
                # b = time.time()
                # ImageWorkSpace.grayTime += (b - a)
                cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)  # 增强对比度，归一化
                # c = time.time()
                # ImageWorkSpace.norTime += (c - b)
        return img

    def clahe(self, img):
        """自适应直方图均衡化"""
        old_clahe = ""
        for m in range(img.shape[0]):
            for n in range(img.shape[1]):
                img[m, n] = 0.3 * img[m, n, 0] + 0.59 * img[m, n, 1] + 0.11 * img[m, n, 2]  # 灰度
                res_clahe = cv2.createCLAHE(clipLimit=2.0,
                                            tileGridSize=(8, 8))
                # 第二步：进行自适应直方图均衡化
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                old_clahe = res_clahe.apply(gray)
        return old_clahe


if __name__ == '__main__':
    a = ImageWorkSpace()
    a.file_name(ImageWorkSpace.OriginPath)

