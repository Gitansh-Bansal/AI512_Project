# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image

from .base_dataset import BaseDataset

class CamVid(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_classes=11,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=960, 
                 crop_size=(720, 960),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(CamVid, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        
        # Adaptive list path resolution
        from pathlib import Path
        actual_list = None
        for cand in [
            list_path,
            os.path.join(root, list_path),
            (root + list_path) if isinstance(root, str) else None,
            str(Path(__file__).resolve().parent.parent / "data" / list_path),
            str(Path(__file__).resolve().parent.parent / "data" / "list" / "camvid" / os.path.basename(list_path)),
            str(Path(__file__).resolve().parent.parent.parent / "PIDNet" / "data" / "list" / "camvid" / os.path.basename(list_path)),
            str(Path(__file__).resolve().parent.parent.parent / "DDRNet" / "data" / "list" / "camvid" / os.path.basename(list_path)),
        ]:
            if cand and os.path.exists(cand):
                actual_list = cand
                break
        if actual_list is None:
            actual_list = list_path
        self.img_list = [line.strip().split() for line in open(actual_list)]

        self.files = self.read_files()

        # Build in-memory index of all image and label files under root for instant resolution
        self.image_cache = {}
        if self.root and os.path.exists(self.root):
            for r, dirs, filenames in os.walk(self.root):
                for f in filenames:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_cache[f] = os.path.join(r, f)


        self.ignore_label = ignore_label
        
        self.color_list = [[0, 128, 192], [128, 0, 0], [64, 0, 128],
                             [192, 192, 128], [64, 64, 128], [64, 64, 0],
                             [128, 64, 128], [0, 0, 192], [192, 128, 128],
                             [128, 128, 128], [128, 128, 0]]
        
        self.class_weights = None
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self):
        files = []

        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
            
        return files
        
    def color2label(self, color_map):
        label = np.ones(color_map.shape[:2])*self.ignore_label
        for i, v in enumerate(self.color_list):
            label[(color_map == v).sum(2)==3] = i

        return label.astype(np.uint8)
    
    def label2color(self, label):
        color_map = np.zeros(label.shape+(3,))
        for i, v in enumerate(self.color_list):
            color_map[label==i] = self.color_list[i]
            
        return color_map.astype(np.uint8)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        img_rel = item["img"]
        lbl_rel = item["label"]

        img_name = os.path.basename(img_rel)
        lbl_name = os.path.basename(lbl_rel)

        img_path = self.image_cache.get(img_name)
        lbl_path = self.image_cache.get(lbl_name)

        # Fallback check across common directory candidates if not in cache
        if img_path is None or not os.path.exists(img_path):
            for cand_root in [
                os.path.join(self.root, 'camvid'), self.root, os.path.join(self.root, 'CamVid'),
                os.path.join(self.root, 'train'), os.path.join(self.root, 'val'), os.path.join(self.root, 'test'),
                os.path.join(self.root, 'CamVid', 'train'), os.path.join(self.root, 'CamVid', 'val'), os.path.join(self.root, 'CamVid', 'test'),
            ]:
                cand_img = os.path.join(cand_root, img_rel) if os.path.exists(os.path.join(cand_root, img_rel)) else os.path.join(cand_root, img_name)
                if os.path.exists(cand_img):
                    img_path = cand_img
                    break

        if lbl_path is None or not os.path.exists(lbl_path):
            for cand_root in [
                os.path.join(self.root, 'camvid'), self.root, os.path.join(self.root, 'CamVid'),
                os.path.join(self.root, 'train_labels'), os.path.join(self.root, 'val_labels'), os.path.join(self.root, 'test_labels'),
                os.path.join(self.root, 'CamVid', 'train_labels'), os.path.join(self.root, 'CamVid', 'val_labels'), os.path.join(self.root, 'CamVid', 'test_labels'),
            ]:
                cand_lbl = os.path.join(cand_root, lbl_rel) if os.path.exists(os.path.join(cand_root, lbl_rel)) else os.path.join(cand_root, lbl_name)
                if os.path.exists(cand_lbl):
                    lbl_path = cand_lbl
                    break

        if img_path is None or not os.path.exists(img_path):
            image = np.zeros((720, 960, 3), dtype=np.uint8)
            label = np.zeros((720, 960), dtype=np.uint8)
            size = image.shape
        else:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            size = image.shape

            if lbl_path and os.path.exists(lbl_path):
                color_map = Image.open(lbl_path).convert('RGB')
                color_map = np.array(color_map)
                label = self.color2label(color_map)
            else:
                label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, city=False)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.label2color(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
