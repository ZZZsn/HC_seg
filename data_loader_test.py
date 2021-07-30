from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import numpy as np


class ImageFolder(data.Dataset):
    def __init__(self, pix_list, image_list, image_size=512):
        """Initializes image paths and preprocessing module."""
        # self.GT_paths = os.path.join(root, 'p_mask')
        self.image_paths = image_list
        self.pix_list = pix_list
        self.image_size = image_size
        self.Transform = T.Compose([T.Resize((image_size, image_size), interpolation=Image.BICUBIC), T.ToTensor()])

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        pix = self.pix_list[index]
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        img_tensor = self.Transform(image)
        return image_path, img_tensor, pix,

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(pix_list, image_list, image_size, batch_size, num_workers=4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(pix_list=pix_list,
                          image_list=image_list,
                          image_size=image_size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  pin_memory=True)
    return data_loader
