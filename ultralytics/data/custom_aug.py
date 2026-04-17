# custom_aug.py

from PIL import Image, ImageFilter
import torchvision.transforms as T
import numpy as np
import torch
import random

class CustomAugment:
    # 【改动1】将标志位改为类变量，所有实例共享，保证只打印一次
    has_printed = False  # <<< 改动：类变量，标记是否打印过

    def __init__(self):
        self.jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        # 【改动2】去掉实例变量 self.has_printed
        # self.has_printed = False  # 这个删除，因为用类变量替代了

    def add_gaussian_noise(self, img):
        img_np = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 15, img_np.shape)
        noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def random_erasing(self, img_np, p=0.5, sl=0.02, sh=0.2, r1=0.3):
        if random.uniform(0, 1) > p:
            return img_np

        h, w, c = img_np.shape
        area = h * w

        for _ in range(100):
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1/r1)

            h_erasing = int(round(np.sqrt(target_area * aspect_ratio)))
            w_erasing = int(round(np.sqrt(target_area / aspect_ratio)))

            if w_erasing < w and h_erasing < h:
                x1 = random.randint(0, h - h_erasing)
                y1 = random.randint(0, w - w_erasing)
                img_np[x1:x1+h_erasing, y1:y1+w_erasing, :] = np.random.randint(0, 256, (h_erasing, w_erasing, c), dtype=np.uint8)
                return img_np

        return img_np

    def __call__(self, data):
        img = data['img']
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        # 【改动3】只打印一次
        if not CustomAugment.has_printed:  # 通过类变量判断
            print("✅ CustomAugment called on image")
            CustomAugment.has_printed = True  # 打印后设置标志

        img = self.jitter(img)

        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            img = img.rotate(angle)

        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.0))

        if random.random() < 0.3:
            img = self.add_gaussian_noise(img)

        img_np = np.array(img)
        #img_np = self.random_erasing(img_np, p=0.5)

        data['img'] = img_np
        return data




