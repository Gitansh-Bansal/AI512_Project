# ------------------------------------------------------------------------------
# Modified for NightCity Dataset based on PIDNet Cityscapes implementation
# ------------------------------------------------------------------------------

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

class NightCity(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path,
                 num_classes=19,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=1024, 
                 crop_size=(512, 1024),
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4,
                 photo_aug=None):

        super(NightCity, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.photo_aug = photo_aug

        self.multi_scale = multi_scale
        self.flip = flip
        
        # Locate list file adaptively
        full_list_path = list_path
        if not os.path.isabs(full_list_path):
            candidates = [
                os.path.join(root, list_path),
                os.path.join(root, 'data', list_path),
                os.path.join('PIDNet', 'data', list_path),
                os.path.join('data', list_path),
                list_path
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    full_list_path = cand
                    break

        if not os.path.exists(full_list_path):
            raise FileNotFoundError(f"Could not find list file at: {full_list_path}")

        with open(full_list_path, 'r', encoding='utf-8') as f:
            self.img_list = [line.strip().split() for line in f if line.strip()]

        self.files = self.read_files()

        # Official Cityscapes 19-class mapping
        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
        
        # Fast vectorized lookup table (8x faster than Python loop)
        self.label_lut = np.full(256, ignore_label, dtype=np.int64)
        for k, v in self.label_mapping.items():
            if 0 <= k < 256:
                self.label_lut[k] = v
        
        # Device-agnostic class weights (compatible with CPU and GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_weights = torch.FloatTensor([
            0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
            0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 
            1.0865, 1.1529, 1.0507
        ]).to(device)
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item[0]
                name = os.path.splitext(os.path.basename(image_path))[0]
                files.append({
                    "img": image_path,
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item[0], item[1]
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
        
    def convert_label(self, label, inverse=False):
        if inverse:
            temp = label.copy()
            for v, k in self.label_mapping.items():
                label[temp == k] = v
            return label
        return self.label_lut[label].copy()

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        
        img_full_path = os.path.join(self.root, item["img"])
        if not os.path.exists(img_full_path):
            if "NightCity-images/NightCity-images" in item["img"]:
                single_nested = item["img"].replace("NightCity-images/NightCity-images", "NightCity-images")
                if os.path.exists(os.path.join(self.root, single_nested)):
                    img_full_path = os.path.join(self.root, single_nested)
        if not os.path.exists(img_full_path):
            if "images/" in item["img"]:
                no_prefix = "images/" + item["img"].split("images/", 1)[-1]
                if os.path.exists(os.path.join(self.root, no_prefix)):
                    img_full_path = os.path.join(self.root, no_prefix)
        if not os.path.exists(img_full_path):
            for candidate_folder in [os.path.join(self.root, 'NightCity-images'), os.path.join(self.root, 'nightcity')]:
                if os.path.exists(candidate_folder):
                    sub = os.path.join(candidate_folder, "images", item["img"].split("images/", 1)[-1] if "images/" in item["img"] else item["img"])
                    if os.path.exists(sub):
                        img_full_path = sub
                        break
        if not os.path.exists(img_full_path):
            # check relative to repo
            alt = os.path.join('AI512_Project', 'Data', item["img"])
            if os.path.exists(alt):
                img_full_path = alt
            else:
                raise FileNotFoundError(f"Image not found: {img_full_path} (root={self.root})")

        image = cv2.imread(img_full_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image at {img_full_path}")
        size = image.shape

        if self.photo_aug is not None:
            image = self.photo_aug(image)

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))
            return image.copy(), np.array(size), name

        lbl_full_path = os.path.join(self.root, item["label"])
        if not os.path.exists(lbl_full_path):
            if "NightCity-label/NightCity-label" in item["label"]:
                single_nested = item["label"].replace("NightCity-label/NightCity-label", "NightCity-label")
                if os.path.exists(os.path.join(self.root, single_nested)):
                    lbl_full_path = os.path.join(self.root, single_nested)
        if not os.path.exists(lbl_full_path):
            if "label/" in item["label"]:
                no_prefix = "label/" + item["label"].split("label/", 1)[-1]
                if os.path.exists(os.path.join(self.root, no_prefix)):
                    lbl_full_path = os.path.join(self.root, no_prefix)
        if not os.path.exists(lbl_full_path):
            for candidate_folder in [os.path.join(self.root, 'NightCity-label'), os.path.join(self.root, 'nightcity')]:
                if os.path.exists(candidate_folder):
                    sub = os.path.join(candidate_folder, "label", item["label"].split("label/", 1)[-1] if "label/" in item["label"] else item["label"])
                    if os.path.exists(sub):
                        lbl_full_path = sub
                        break
        if not os.path.exists(lbl_full_path):
            alt = os.path.join('AI512_Project', 'Data', item["label"])
            if os.path.exists(alt):
                lbl_full_path = alt
            else:
                raise FileNotFoundError(f"Label not found: {lbl_full_path} (root={self.root})")

        label = cv2.imread(lbl_full_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Failed to read label at {lbl_full_path}")

        label = self.convert_label(label)

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name
    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
