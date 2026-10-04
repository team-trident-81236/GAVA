from .coco import build as build_coco, DatasetCOCO
from .lvis import build as build_lvis, DatasetLVIS
from .fss import build as build_fss, DatasetFSS
from .deepglobe import build as build_dg, DatasetDeepglobe
from .isic import build as build_isic, DatasetISIC
from .lung import build as build_lung, DatasetLung
from .pascal_part import build as build_pascal_part, DatasetPASCALPart
from .paco_part import build as build_paco_part, DatasetPACOPart
from .ade20k import build as build_ade20k, SemADE
from .veins import build as build_veins, DatasetVeins   # ← new
from .vein_reta import build as build_vein_reta, DatasetVeinReta   # ← new
from .fives import build as build_fives, DatasetFives 
from .deepglobe_road import build as build_deepglobe_road, DatasetDeepglobeRoad
from .deepcrack import build as build_deepcrack, DatasetDeepCrack

def build_dataset(dataset_file: str, image_set: str, args=None):
    if dataset_file == 'coco':
        return build_coco(image_set, args)
    if dataset_file == 'lvis':
        return build_lvis(image_set, args)
    if dataset_file == 'fss':
        return build_fss(image_set, args)
    if dataset_file == 'pascal_part':
        return build_pascal_part(image_set, args)
    if dataset_file == 'paco_part':
        return build_paco_part(image_set, args)
    if dataset_file == 'deepglobe':
        return build_dg(image_set, args)
    if dataset_file == 'isic':
        return build_isic(image_set, args)
    if dataset_file == 'lung':
        return build_lung(image_set, args)
    if dataset_file == 'ade20k':
        return build_ade20k(image_set, args)
    if dataset_file == 'veins':                         # ← new
        return build_veins(image_set, args)   
    if dataset_file == 'vein_reta':                              # ← new
        return build_vein_reta(image_set, args)
    if dataset_file == 'fives':                             # ← add this block
        return build_fives(image_set, args)    
    if dataset_file == 'deepglobe_road':
        return build_deepglobe_road(image_set, args)
    if dataset_file == 'deepcrack':
        return build_deepcrack(image_set, args)
    elif dataset_file == 'multi':
        from datasets.transform_utils import CustomConcatDataset
        ds_list = [build_dataset(name, image_set="train", args=args) for name in args.multi_train]
        return CustomConcatDataset(dataset_list=ds_list, dataset_ratio=args.ds_weight, samples_per_epoch=210000)
    raise ValueError(f'dataset {dataset_file} not supported')
