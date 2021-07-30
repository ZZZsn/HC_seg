import torch
from PIL import Image
from torchvision import transforms as T
import numpy as np


def test_preprocess(image_path, outputsize =512):
    img = Image.open(image_path)
    Transform = T.Compose([T.Resize((outputsize, outputsize), interpolation=Image.BICUBIC), T.ToTensor()])  #
    img_tensor = Transform(img)
    img_array = np.array(img, dtype=np.float32)
    or_shape = img_array.shape  # 原始图片的尺寸

    return img_tensor, or_shape, img_array
