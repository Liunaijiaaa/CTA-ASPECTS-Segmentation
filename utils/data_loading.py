import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import skimage.io as io

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

# adjust CT window width & window center
#     def get_window_size(self, window_type):
#         if self.window_type == 'lung':  # 肺窗
#             center = -600
#             width = 1200
#         elif self.window_type == 'Mediastinal':  # 纵膈窗
#             center = 40
#             width = 400
#         elif self.window_type == 'brian':
#             center = 40
#             width = 80
#         return center, width

    @classmethod
    def preprocess_img(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # img_ndarray = np.asarray(pil_img)
        img_ndarray = np.array(pil_img)

        rows = 512
        cols = 512
        center, width = 40, 80
        img_ndarray.flags.writeable = True
        min = (2 * center - width) / 2.0 + 0.5
        max = (2 * center + width) / 2.0 + 0.5
        dFactor = 255.0 / (max - min)
        for i in np.arange(rows):
            for j in np.arange(cols):
                img_ndarray[i, j] = int((img_ndarray[i, j] - min) * dFactor)
        min_index = img_ndarray < 0
        img_ndarray[min_index] = 0
        max_index = img_ndarray > 255
        img_ndarray[max_index] = 255

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        return img_ndarray

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        # img_ndarray = np.asarray(pil_img)
        # img_ndarray = np.array(pil_img)
        img_ndarray = np.array(np.uint8(pil_img))

        if img_ndarray.ndim == 2 and not is_mask:
            # rows = 512
            # cols = 512
            # center, width = 40, 80
            # img_ndarray.flags.writeable = True
            # min = (2 * center - width) / 2.0 + 0.5
            # max = (2 * center + width) / 2.0 + 0.5
            # dFactor = 255.0 / (max - min)
            # for i in np.arange(rows):
            #     for j in np.arange(cols):
            #         img_ndarray[i, j] = int((img_ndarray[i, j] - min) * dFactor)
            # min_index = img_ndarray < 0
            # img_ndarray[min_index] = 0
            # max_index = img_ndarray > 255
            # img_ndarray[max_index] = 255
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        # 无需调整窗宽窗位的情况
        if not is_mask:
            img_ndarray = img_ndarray / 255
        else:
        # img_ndarray = img_ndarray[np.newaxis, ...]
            img_ndarray[img_ndarray == 7] = 4
            img_ndarray[img_ndarray == 8] = 5
            img_ndarray[img_ndarray == 9] = 6
            img_ndarray[img_ndarray == 10] = 7
            img_ndarray[img_ndarray == 11] = 8
            img_ndarray[img_ndarray == 12] = 9
            img_ndarray[img_ndarray == 13] = 10
            img_ndarray[img_ndarray == 17] = 11
            img_ndarray[img_ndarray == 18] = 12
            img_ndarray[img_ndarray == 19] = 13
            img_ndarray[img_ndarray == 20] = 14
        # palette = [[0], [1], [2], [3],  [7], [8], [9], [10], [11], [12], [13], [17], [18], [19], [20]]
        # new_img_ndarray = []
        # for colour in palette:
        #     equality = np.equal(img_ndarray, colour)
        #     class_map = np.all(equality, axis=-1)
        #     new_img_ndarray.append(class_map)
        # img_ndarray = np.stack(new_img_ndarray, axis=-1).astype(np.float32)
        # img_ndarray = img_ndarray.transpose((2, 0, 1))
        # return img_ndarray
        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        # elif ext in ['.png']:
        #     return io.imread(filename, as_gray=True)
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
            # 'mask': torch.as_tensor(mask.copy()).int().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
