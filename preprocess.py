"""
对原始训练数据进行处理：
1.分成图像与标签两个文件夹
2.读取mask进行孔洞填充，并统计mask面积大小
3.处理好的mask保存同图像一样的名字
4.读取存放头围结果的csv，读取像素实际大小和头围两列数据
5.保存训练所需的csv
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from utils.other_util import readCsv, printProgressBar
import pandas as pd
sep = '//'


def fill_inter_bone(mask):
    # 对一张图像做孔洞填充，读入的是一层
    mask = mask_fill = mask.astype(np.uint8)
    if np.sum(mask[:]) != 0:    # 即读入图层有值
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        len_contour = len(contours)
        contour_list = []
        for i in range(len_contour):
            drawing = np.zeros_like(mask, np.uint8)  # create a black image
            img_contour = cv2.drawContours(drawing, contours, i, (1, 1, 1), -1)
            contour_list.append(img_contour)
        mask_fill = sum(contour_list)
        mask_fill[mask_fill >= 1] = 1
    return mask_fill.astype(np.uint8)


def fill_inter_bone_2(mask):
    # 先对图像的上下边多加一行，确保封闭椭圆，再进行填充
    mask_append = np.append(np.ones([1, mask.shape[1]])*255, mask, axis=0)
    mask_append = np.append(mask_append, np.ones([1, mask.shape[1]])*255, axis=0)
    # 对一张图像做孔洞填充，读入的是一层
    mask_append_fill = fill_inter_bone(mask_append)
    mask_fill = mask_append_fill[1:-2, :]
    return mask_fill.astype(np.uint8)


train_labels_files_path = '/EDAN2021/data/training_set/labels'
pix_csv = r'/EDAN2021/data/training_set_pixel_size_and_HC.csv'
csv_save_path = '/EDAN2021/data/train.csv'                  # 生成的csv文件
save_folder = '/EDAN2021/data/training_set/labels_pre'      # 保存结果
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

csvlines = readCsv(pix_csv)
pix_header = csvlines[0]
pix_csv_data = csvlines[1:]
train_files_list = [i[0] for i in pix_csv_data]
# 加上label的后缀
labels_files_list = [i.split('.')[0]+'_Annotation.png' for i in train_files_list]

# 读取label进行填充，并计算label大小保存csv
label_nums = []
csv_data = []
iii = 1
# 几个比较特殊的,椭圆部分被边缘切断无法直接填充
special_ids = ['186_', '346_', '628_2', '787_']
for label_name in labels_files_list:
    label_file = train_labels_files_path+sep+label_name
    label = cv2.imread(label_file)
    label_gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)    # 转2维
    # 填充圆圈内部
    label_fill = fill_inter_bone(label_gray)    # 0-1
    # 特殊的几张额外进行处理
    if label_name.split('HC_Annotation.png')[0] in special_ids:
        label_fill = fill_inter_bone_2(label_gray)
    # 保存图像
    label_save = cv2.normalize(label_fill, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    save_path = save_folder+sep+label_name.replace('_Annotation', '')   # label的名字和image的一致
    cv2.imwrite(save_path, label_save)
    # csv数据
    label_num = np.sum(label_save) / 255
    csv_data.append([label_name.replace('_Annotation', ''), str(label_num)])
    printProgressBar(iii, content='')
    iii+=1

# 从提供的csv中获取像素对应的实际大小和头围
pix = [i[1] for i in pix_csv_data]
hc = [i[2] for i in pix_csv_data]
# 加在后面
for i in range(len(csv_data)):
    csv_data[i].append(pix[i])
    csv_data[i].append(hc[i])

# 训练所需的csv
header_name = ['ID', 'label_num', pix_header[1], pix_header[2]]
train_csv = pd.DataFrame(data=csv_data)
train_csv.to_csv(csv_save_path, header=header_name, index=None)

# 查看
# plt.figure(1)
# plt.imshow()
# plt.show()



