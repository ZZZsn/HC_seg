"""一张一张图片的预测并计算头围信息,然后保存"""
# 设置cpu加速
import os
os.environ["MKL_NUM_THREADS"] = '8'
os.environ["NUMEXPR_NUM_THREADS"] = '8'
os.environ["OMP_NUM_THREADS"] = '8'
import time
from skimage.transform import resize
import segmentation_models_pytorch as smp
from utils.other_util import *
from utils.FitEllipse import *
from utils.test_preprocess import *
import pandas as pd
from data_loader_test import *
sep = os.sep


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    """处理的逻辑:
    1)读取图片
    2)图片预处理
    3)预测roi
    4)还原分割结果到原大小
    5)后处理，取单一连通域
    6)计算头围信息，计算FPS
    7)保存预测图片到指定文件夹，保存头围信息到指定csv
    """
    # 路径设置
    imgs_path = r'/data/medai05/EDAN2021/data/test_set/images'
    pix_csv_file = r'/data/medai05/EDAN2021/data/test_set_pixel_size.csv'
    save_mask_path = r'/data/medai05/EDAN2021/data/test_set/pre_mask'
    save_img_mask_path = r'/data/medai05/EDAN2021/data/test_set/img_mask'
    save_csv = r'/data/medai05/EDAN2021/data/test_set_result.csv'
    if not os.path.exists(save_mask_path):
        os.makedirs(save_mask_path)
        os.makedirs(save_img_mask_path)

    model_type = 0  # 经测试，torch与script的速度差不多，onnx模型的推理速度最快
    pytorch_weight = r'/data/medai05/EDAN2021/model/HC_mobileV3_100.pkl'    # 记得修改下面的模型
    script_weight = "/data/medai05/EDAN2021/model/HC_seg_32.pt"
    onnx_weight = "/data/medai05/EDAN2021/model/HC_mobileV2.onnx"
    input_size = 256
    batch_size = 2  # bz太大会影响推理速度，2的推理速度比较快

    # GPU
    # cuda_idx = 1
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_idx)

    device = torch.device('cpu')  # 不用gpu

    # 获取测试集的所有图片
    csvlines = readCsv(pix_csv_file)
    header = csvlines[0]
    csv_data = csvlines[1:]
    pix_list = [float(i[1]) for i in csv_data]
    test_list = [imgs_path + sep + i[0] for i in csv_data]
    test_data_load = get_loader(pix_list, test_list, input_size, batch_size=batch_size, num_workers=8)

    """加载模型"""
    if model_type == 0:
        # 加载pytorch模型
        with torch.no_grad():
            # 构建模型
            model = smp.Unet(encoder_name="timm-mobilenetv3_small_100", in_channels=1, classes=1)
            model.to(device)
            model.load_state_dict(torch.load(pytorch_weight, map_location=device))
            model.eval()
    elif model_type == 1:
        # 加载pytorch_script模型
        with torch.no_grad():
            model = torch.jit.load(script_weight)
    elif model_type == 2:
        # 加载onnx模型
        import onnxruntime
        model = onnxruntime.InferenceSession(onnx_weight)

    # 测试
    time_real_start = time.time()
    time_all = []
    for index1, sample in enumerate(test_data_load):
        img_file, img, pix = sample
        img = img.to(device)
        # 模型推理
        time_start = time.time()
        if model_type == 0 or model_type == 1:
            # pytorch和script模型
            mask_pre = model(img)
            time_all.append(time.time() - time_start)  # 测试耗时
            mask_pre = torch.sigmoid(mask_pre)
        elif model_type == 2:
            # onnx模型
            ort_inputs = {model.get_inputs()[0].name: to_numpy(img)}
            mask_pre = model.run(None, ort_inputs)
            time_all.append(time.time() - time_start)  # 测试耗时
            mask_pre = torch.sigmoid(torch.tensor(mask_pre[0]))

        for index2 in range(len(img_file)):
            # 获取最大联通域
            mask_pre_array = (torch.squeeze(mask_pre[index2, :, :, :])).data.cpu().numpy()
            mask_pre_array = (mask_pre_array > 0.5)
            mask_array_biggest = getmaxcomponent(mask_pre_array).astype(np.bool)
            # 原图大小的二值图
            img_ori_array = np.array(Image.open(img_file[index2]), dtype=np.float32)
            ori_shape = img_ori_array.shape
            final_mask = resize(mask_array_biggest, ori_shape, order=0).astype(np.uint8)
            # 计算头围以及拟合椭圆的相关信息
            HC = C_elipse(final_mask) * pix[index2] - pix[index2] * 5.4
            x, y, a, b, ang = fitelipse(final_mask)

            # 保存图像

            if save_mask_path is not None:
                # 拟合椭圆，画在原图上，需要int才能输入
                cv2.ellipse(img=img_ori_array, center=(int(x), int(y)), axes=(int(a), int(b)), angle=ang,
                            startAngle=0, endAngle=360, color=(255, 255, 255), thickness=2)
                # 画上预测结果的轮廓
                # contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # cv2.drawContours(img_ori_array, [contours[0]], 0, (255, 255, 255), thickness=2)
                final_save_img_mask_path = save_img_mask_path + sep + img_file[index2].split(sep)[-1]
                cv2.imwrite(final_save_img_mask_path, img_ori_array)
                # 预测的二值图
                final_mask = (final_mask * 255).astype(np.uint8)
                final_save_mask_path = save_mask_path + sep + img_file[index2].split(sep)[-1]
                im = Image.fromarray(final_mask)
                im.save(final_save_mask_path)

            # 保存椭圆相关信息
            csv_data[index1 * batch_size + index2].append(HC)
            csv_data[index1 * batch_size + index2].append(x)
            csv_data[index1 * batch_size + index2].append(y)
            csv_data[index1 * batch_size + index2].append(a)
            csv_data[index1 * batch_size + index2].append(b)
            csv_data[index1 * batch_size + index2].append(ang)

            # 进度条
            printProgressBar(1 + index1 * batch_size + index2, 335, content=img_file[index2].split(sep)[-1])

            # 释放内存，好使用更大的batch_size
            del im, final_mask, img_ori_array, mask_array_biggest, mask_pre_array

    new_header = [header[0], header[1], 'head circumference (mm)', 'x', 'y', 'a', 'b', 'angle']
    result_csv = pd.DataFrame(data=csv_data)
    result_csv.to_csv(save_csv, header=new_header, index=False)

    # fps的计算: 1/（测试完所有图片的总时长/图片数）
    time_real_all = time.time() - time_real_start
    print("\n all time use %.2f s" % time_real_all)
    fps = 1. / (np.sum(time_all) / 335)
    aver_use = np.mean(time_all) / batch_size
    print(" fps %.2f , average use %.4f s" % (fps, aver_use))
