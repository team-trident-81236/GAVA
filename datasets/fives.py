r""" FIVES retinal vessel few-shot segmentation dataset """
import glob
import os

import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetFives(Dataset):
    def __init__(self, datapath, transform, split, shot, episode_aug=None):
        self.split = split
        self.benchmark = 'fives'
        self.shot = shot
        # Optional episode-level augmentation (--thin_aug, training only);
        # None keeps the original behavior.
        self.episode_aug = episode_aug

        self.img_dir  = os.path.join(datapath, 'Fives/data/fives/images')
        self.mask_dir = os.path.join(datapath, 'Fives/data/fives/masks')

        # 4 disease categories: A, D, G, N
        self.categories = ['A', 'D', 'G', 'N']
        self.nclass = 4
        self.class_ids = list(range(4))

        self.img_metadata, self.labels = self.build_img_metadata(split)
        self.transform = transform

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names, class_id = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(
            query_name, support_names
        )

        if self.episode_aug is not None:
            imgs, masks = self.episode_aug([query_img] + support_imgs, [query_mask] + support_masks)
            query_img, support_imgs = imgs[0], imgs[1:]
            query_mask, support_masks = masks[0], masks[1:]

        query_img = self.transform(query_img)
        query_mask = F.interpolate(
            query_mask.unsqueeze(0).unsqueeze(0).float(),
            query_img.size()[-2:], mode='nearest'
        ).squeeze()

        support_imgs = torch.stack([self.transform(s) for s in support_imgs])

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
            'class_id':      torch.tensor(class_id),
        }

    def load_frame(self, query_name, support_names):
        query_img    = Image.open(os.path.join(self.img_dir, query_name)).convert('RGB')
        support_imgs = [
            Image.open(os.path.join(self.img_dir, n)).convert('RGB')
            for n in support_names
        ]
        query_mask    = self.read_mask(query_name)
        support_masks = [self.read_mask(n) for n in support_names]
        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, filename):
        path = os.path.join(self.mask_dir, filename)
        mask = torch.tensor(np.array(Image.open(path).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def get_category(self, filename):
        # train_001_A.png → 'A'
        return filename.split('_')[-1].split('.')[0]

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        query_cat  = self.get_category(query_name)
        class_id   = self.categories.index(query_cat)

        # support must be same category, different image
        pool = [
            p for p in self.img_metadata
            if p != query_name and self.get_category(p) == query_cat
        ]
        support_idxs = np.random.choice(len(pool), self.shot, replace=False)
        support_names = [pool[i] for i in support_idxs]

        return query_name, support_names, class_id

    def build_img_metadata(self, split):
        prefix = 'train' if split in ('trn', 'train') else 'test'
        all_imgs = sorted([
            os.path.basename(p)
            for p in glob.glob(os.path.join(self.img_dir, f'{prefix}_*.png'))
        ])
        labels = [self.categories.index(self.get_category(f)) for f in all_imgs]
        return all_imgs, labels


def build(image_set, args):
    img_size = 518
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if image_set == 'train':
        split = 'trn'
    else:
        split = 'test'

    episode_aug = None
    if split == 'trn' and getattr(args, 'thin_aug', False):
        from datasets.thin_structure_aug import ThinStructureEpisodeAug
        episode_aug = ThinStructureEpisodeAug()

    return DatasetFives(
        datapath=args.data_root,
        transform=transform,
        shot=args.shots,
        split=split,
        episode_aug=episode_aug,
    )