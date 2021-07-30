import random

from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F

from img_mask_aug import data_aug
from utils.other_util import char_color


class ImageFolder(data.Dataset):
    def __init__(self, pix_hc_list, GT_list, image_list, image_size=512, mode='train',
                 augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        # self.root = root

        # GT : Ground Truth
        # self.GT_paths = os.path.join(root, 'p_mask')
        self.GT_paths = GT_list
        self.image_paths = image_list
        self.pix_hc_list = pix_hc_list

        self.image_size = image_size
        self.mode = mode
        self.augmentation_prob = augmentation_prob

    # print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        pix, hc = self.pix_hc_list[index]
        image_path = self.image_paths[index]
        GT_path = self.GT_paths[index]

        image = Image.open(image_path)
        GT = Image.open(GT_path)

        Transform = []
        Transform_GT = []  # 注意,GT的插值需要最近邻nearest,但是采取非线性插值可能有奇效

        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            # ----------------------------------- 扩增操作 ------------------------------------
            [image, GT] = data_aug(image, GT)

            image = Image.fromarray(image)
            GT = Image.fromarray(GT)

        final_size = self.image_size
        Transform.append(T.Resize((final_size, final_size), interpolation=Image.BICUBIC))
        Transform_GT.append(T.Resize((final_size, final_size), interpolation=Image.NEAREST))

        Transform.append(T.ToTensor())
        Transform_GT.append(T.ToTensor())

        Transform = T.Compose(Transform)
        Transform_GT = T.Compose(Transform_GT)

        image = Transform(image)
        GT = Transform_GT(GT)
        # print(GT.shape)
        # 如果是rgb,则需要
        # Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # image = Norm_(image)

        return image_path, image, GT, float(pix), float(hc)

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(pix_hc_list, GT_list, image_list,
               image_size, batch_size,
                num_workers=2, mode='train', augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(pix_hc_list=pix_hc_list,
                          GT_list=GT_list,
                          image_list=image_list,
                          image_size=image_size,
                          mode=mode,
                          augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader
