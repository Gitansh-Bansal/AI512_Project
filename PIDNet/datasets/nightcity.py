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

        norm_root = os.path.normpath(str(root))
        while os.path.basename(norm_root).lower() in ("nightcity-label", "nightcity-images", "label", "images"):
            norm_root = os.path.dirname(norm_root)
        self.root = norm_root
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

    def _find_file(self, rel_path, kind='img'):
        norm_rel = rel_path.replace('\\', '/')
        p = os.path.join(self.root, norm_rel)
        if os.path.exists(p):
            return p

        p_parent = os.path.join(os.path.dirname(self.root), norm_rel)
        if os.path.exists(p_parent):
            return p_parent

        for base in [self.root, os.path.dirname(self.root)]:
            if "NightCity-images/NightCity-images" in norm_rel:
                cand = os.path.join(base, norm_rel.replace("NightCity-images/NightCity-images", "NightCity-images"))
                if os.path.exists(cand):
                    return cand
            if "NightCity-label/NightCity-label" in norm_rel:
                cand = os.path.join(base, norm_rel.replace("NightCity-label/NightCity-label", "NightCity-label"))
                if os.path.exists(cand):
                    return cand

        sub_folder = "images" if kind == 'img' else "label"
        if f"{sub_folder}/" in norm_rel:
            suffix = sub_folder + "/" + norm_rel.split(f"{sub_folder}/", 1)[-1]
            for base in [self.root, os.path.dirname(self.root)]:
                for cand in [
                    os.path.join(base, suffix),
                    os.path.join(base, f"NightCity-{sub_folder}", suffix),
                    os.path.join(base, f"NightCity-{sub_folder}", f"NightCity-{sub_folder}", suffix),
                    os.path.join(base, "nightcity", suffix),
                    os.path.join(base, "nightcity-dataset", suffix)
                ]:
                    if os.path.exists(cand):
                        return cand

        for alt in [os.path.join('AI512_Project', 'Data', norm_rel), os.path.join('Data', norm_rel)]:
            if os.path.exists(alt):
                return alt

        fname = os.path.basename(norm_rel)
        search_dirs = [self.root, os.path.dirname(self.root)]
        if os.path.exists('/kaggle/input'):
            search_dirs.append('/kaggle/input')
        for sdir in search_dirs:
            if os.path.exists(sdir):
                for match in Path(sdir).rglob(fname):
                    return str(match)

        raise FileNotFoundError(f"{'Image' if kind == 'img' else 'Label'} not found: {p} (root={self.root})")

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        
        img_full_path = self._find_file(item["img"], kind='img')
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

        lbl_full_path = self._find_file(item["label"], kind='lbl')
        label = cv2.imread(lbl_full_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise ValueError(f"Failed to read label at {lbl_full_path}")

        label = self.convert_label(label)

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

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
