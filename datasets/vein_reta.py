r""" Retinal vein few-shot segmentation dataset — single category """
import glob
import os

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetVeinReta(Dataset):
    def __init__(self, datapath, transform, split, shot):
        self.split = split
        self.benchmark = 'vein_reta'
        self.shot = shot
        self.base_path = os.path.join(datapath, 'vein_reta/data/vein_reta')

        self.categories = ['vein_reta']
        self.nclass = 1
        self.class_ids = [0]

        self.img_metadata = self.build_img_metadata()
        self.transform = transform

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(
            query_mask.unsqueeze(0).unsqueeze(0).float(),
            query_img.size()[-2:], mode='nearest'
        ).squeeze()

        support_imgs = torch.stack([self.transform(simg) for simg in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(
                smask.unsqueeze(0).unsqueeze(0).float(),
                support_imgs.size()[-2:], mode='nearest'
            ).squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        return {
            'query_img':     query_img,
            'query_mask':    query_mask,
            'query_name':    query_name,
            'support_imgs':  support_imgs,
            'support_masks': support_masks,
            'support_names': support_names,
            'class_id':      torch.tensor(0),
        }

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_mask_name = self.get_mask_path(query_name)
        support_mask_names = [self.get_mask_path(n) for n in support_names]

        query_mask = self.read_mask(query_mask_name)
        support_masks = [self.read_mask(m) for m in support_mask_names]

        return query_img, query_mask, support_imgs, support_masks

    def get_mask_path(self, img_path):
        # IDRiD_01.jpg  →  IDRiD_01_vessel.png
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(os.path.dirname(img_path), stem + '_vessel.png')

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        pool = [p for p in self.img_metadata if p != query_name]
        support_idxs = np.random.choice(len(pool), self.shot, replace=False)
        support_names = [pool[i] for i in support_idxs]
        return query_name, support_names

    def build_img_metadata(self):
        all_imgs = sorted(glob.glob(os.path.join(self.base_path, '*.jpg')))
        return all_imgs


def build(image_set, args):
    img_size = 518
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return DatasetVeinReta(
        datapath=args.data_root,
        transform=transform,
        shot=args.shots,
        split=image_set,
    )