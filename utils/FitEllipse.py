import cv2
import math
import numpy as np
import SimpleITK as sitk


def getmaxcomponent(mask_array, num_limit=5):
    # sitk方法，更快,得到的是相当于ski的connectivity=3的结果
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    output_ex = cca.Execute(_input)
    labeled_img = sitk.GetArrayFromImage(output_ex)
    # num = cca.GetObjectCount()
    max_label = 0
    max_num = 0
    for i in range(1, num_limit):  # 不必全部遍历，一般在前面就有对应的label，减少计算时间
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    # print(str(max_label) + ' num:' + str(max_num) + 'from ' + str(num))  # 看第几个是最大的
    return np.array((labeled_img == max_label)).astype(np.uint8)


def fitelipse(mask):
    """
    opencv拟合椭圆
    mask-输入二值图像
    return：返回椭圆信息:中心位置，长短轴，旋转角
    """
    mask_max = getmaxcomponent(mask)
    contours, hierarchy = cv2.findContours(mask_max, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 轮廓查找
    if len(contours) == 0:            # 如果没能找到轮廓，就返回0
        return 0, 0, 0, 0, 0
    if contours[0].shape[0] < 6:       # 如果点数不够，就返回0
        return 0, 0, 0, 0, 0
    retval = cv2.fitEllipse(contours[0])  # 取轮廓拟合椭圆
    # 中心坐标xy，长短轴ab（注意不是半轴），旋转角angle
    x, y, a, b, angle = retval[0][0], retval[0][1], retval[1][0]/2, retval[1][1]/2, retval[2]
    return x, y, a, b, angle


def C_elipse(mask):
    """
    根据拉马努詹的近似公式,计算椭圆周长
    """
    _, _, a, b, _ = fitelipse(mask)
    if a == 0:
        return 0
    perimeter = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    return perimeter

# 测试
# from PIL import Image
# import numpy as np
# if __name__ == '__main__':
#     path = r'/EDAN2021/data/training_set/labels_pre/000_HC.png'
#     mask_i = Image.open(path)    # 读图
#     mask = np.array(mask_i)
#     mask = getmaxcomponent(mask, 3)
#     x, y, a, b, angle = fitelipse(mask)
#     C = C_elipse(a, b)
