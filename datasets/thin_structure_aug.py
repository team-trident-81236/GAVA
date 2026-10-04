r""" Episode-level augmentation for thin / tubular structure few-shot training.

Opt-in via --thin_aug (training split only). Aims to stop the adapter from
learning a single-domain shortcut ("dark lines of retinal-vessel width on a
bright fundus background") so it transfers to other thin structures such as
cracks and roads:

  * Geometric, sampled independently per frame (support and query differ):
      flips, 90-degree rotations, and a log-uniform zoom in [1/max_zoom, max_zoom].
      Zoom-in crops and upsamples (thicker lines), zoom-out shrinks and pads
      (thinner lines), so line width varies within an episode and the model
      cannot rely on one fixed width.
  * Photometric, sampled once per episode and shared by all frames:
      contrast inversion, grayscale, channel permutation, color jitter, blur.
      Sharing keeps support and query consistent, so the support mask is the
      only reliable cue for what the target looks like in that episode.
"""
import math

import numpy as np
import PIL.Image as Image
import torch
from PIL import ImageFilter, ImageOps
from torchvision.transforms import functional as TF


class ThinStructureEpisodeAug:
    def __init__(self, work_size: int = 1036, max_zoom: float = 2.0,
                 p_invert: float = 0.5, p_gray: float = 0.3, p_channel_perm: float = 0.3,
                 p_jitter: float = 0.8, p_blur: float = 0.2):
        # Frames are first resized to work_size (2x the 518 model input) so a
        # 2x zoom-in still lands on native model resolution, and so the
        # augmentation stays cheap on the 2048px FIVES images.
        self.work_size = work_size
        self.max_zoom = max_zoom
        self.p_invert = p_invert
        self.p_gray = p_gray
        self.p_channel_perm = p_channel_perm
        self.p_jitter = p_jitter
        self.p_blur = p_blur

    def __call__(self, imgs, masks):
        """imgs: list of RGB PIL images; masks: list of HxW {0,1} uint8 tensors."""
        photo = self._sample_photometric()
        out_imgs, out_masks = [], []
        for img, mask in zip(imgs, masks):
            img, mask = self._geometric(img, mask)
            img = self._photometric(img, photo)
            out_imgs.append(img)
            out_masks.append(mask)
        return out_imgs, out_masks

    # ---------------- geometric (per frame) ----------------
    def _geometric(self, img, mask):
        ws = self.work_size
        m = Image.fromarray(mask.numpy().astype(np.uint8) * 255)
        img = img.resize((ws, ws), Image.BILINEAR)
        m = m.resize((ws, ws), Image.NEAREST)

        if np.random.rand() < 0.5:
            img, m = ImageOps.mirror(img), ImageOps.mirror(m)
        if np.random.rand() < 0.5:
            img, m = ImageOps.flip(img), ImageOps.flip(m)
        k = np.random.randint(4)
        if k:
            img, m = img.rotate(90 * k), m.rotate(90 * k)

        zoom = math.exp(np.random.uniform(-math.log(self.max_zoom), math.log(self.max_zoom)))
        if zoom > 1:   # zoom in: random crop, upsample back -> thicker structures
            c = max(1, int(round(ws / zoom)))
            x, y = np.random.randint(0, ws - c + 1), np.random.randint(0, ws - c + 1)
            img = img.crop((x, y, x + c, y + c)).resize((ws, ws), Image.BILINEAR)
            m = m.crop((x, y, x + c, y + c)).resize((ws, ws), Image.NEAREST)
        elif zoom < 1:  # zoom out: shrink, paste on black canvas -> thinner structures
            s = max(1, int(round(ws * zoom)))
            x, y = np.random.randint(0, ws - s + 1), np.random.randint(0, ws - s + 1)
            canvas_i = Image.new('RGB', (ws, ws))
            canvas_m = Image.new('L', (ws, ws))
            canvas_i.paste(img.resize((s, s), Image.BILINEAR), (x, y))
            canvas_m.paste(m.resize((s, s), Image.NEAREST), (x, y))
            img, m = canvas_i, canvas_m

        mask = torch.from_numpy((np.array(m) >= 128).astype(np.uint8))
        return img, mask

    # ---------------- photometric (per episode) ----------------
    def _sample_photometric(self):
        return {
            'invert': np.random.rand() < self.p_invert,
            'gray': np.random.rand() < self.p_gray,
            'perm': np.random.permutation(3) if np.random.rand() < self.p_channel_perm else None,
            'jitter': (np.random.uniform(0.7, 1.3), np.random.uniform(0.7, 1.3),
                       np.random.uniform(0.7, 1.3), np.random.uniform(-0.05, 0.05))
                      if np.random.rand() < self.p_jitter else None,
            'blur': np.random.uniform(0.1, 1.5) if np.random.rand() < self.p_blur else None,
        }

    def _photometric(self, img, p):
        if p['jitter'] is not None:
            b, c, s, h = p['jitter']
            img = TF.adjust_brightness(img, b)
            img = TF.adjust_contrast(img, c)
            img = TF.adjust_saturation(img, s)
            img = TF.adjust_hue(img, h)
        if p['perm'] is not None:
            img = Image.merge('RGB', [img.split()[i] for i in p['perm']])
        if p['gray']:
            img = ImageOps.grayscale(img).convert('RGB')
        if p['invert']:
            img = ImageOps.invert(img)
        if p['blur'] is not None:
            img = img.filter(ImageFilter.GaussianBlur(radius=p['blur']))
        return img
