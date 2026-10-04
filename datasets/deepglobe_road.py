r"""DeepGlobe Road Extraction few-shot segmentation dataset."""
import glob
import os

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetDeepglobeRoad(Dataset):
    def __init__(self, datapath, transform, split, shot):
        self.split = split
        self.benchmark = 'deepglobe_road'
        self.shot = shot
        self.transform = transform

        self.img_dir  = os.path.join(datapath, 'data/deepglobe_road/images')
        self.mask_dir = os.path.join(datapath, 'deepglobe_road/masks')

        self.categories = ['road']
        self.nclass = 1
        self.class_ids = [0]

        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(
            query_name, support_names
        )

        query_img = self.transform(query_img)
        query_mask = F.interpolate(
            query_mask.unsqueeze(0).unsqueeze(0).float(),
            query_img.size()[-2:],
            mode='nearest'
        ).squeeze()

        support_imgs = torch.stack([self.transform(img) for img in support_imgs])

        support_masks_tmp = []
        for support_mask in support_masks:
            support_mask = F.interpolate(
                support_mask.unsqueeze(0).unsqueeze(0).float(),
                support_imgs.size()[-2:],
                mode='nearest'
            ).squeeze()
            support_masks_tmp.append(support_mask)
        support_masks = torch.stack(support_masks_tmp)

        return {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': query_name,
            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_names': support_names,
            'class_id': torch.tensor(0),
        }

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_mask = self.read_mask(self.get_mask_path(query_name))
        support_masks = [
            self.read_mask(self.get_mask_path(name)) for name in support_names
        ]
        return query_img, query_mask, support_imgs, support_masks

    def get_mask_path(self, img_path):
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.mask_dir, image_id + '.png')

    def read_mask(self, mask_path):
        mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        pool = [path for path in self.img_metadata if path != query_name]
        support_idxs = np.random.choice(len(pool), self.shot, replace=False)
        support_names = [pool[i] for i in support_idxs]
        return query_name, support_names

    def build_img_metadata(self):
        return sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))


def build(image_set, args):
    img_size = 518
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return DatasetDeepglobeRoad(
        datapath=args.data_root,
        transform=transform,
        shot=args.shots,
        split=image_set,
    )