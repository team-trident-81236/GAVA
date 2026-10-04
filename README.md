# GAVA: A Graph-Aware Vessel Adapter for Few-Shot Segmentation of Thin Structures

GAVA is a lightweight **graph adapter** inserted into the frozen SAM2 image encoder of
[SANSA](README_SANSA.md). It runs in parallel to AdaptFormer and links each feature token to its most
similar neighbours inside a local window, so evidence travels **along** vessel-like structures instead
of only across square patches.

Motivation: pretrained few-shot segmenters are built for compact objects. On retinal vessels they
select the whole retina, not the vessel tree. GAVA adds the missing notion of *connectivity* while
keeping SAM2 frozen and training only ~1.7M parameters.

```
x_out = x + MLP(x) + s·AdaptFormer(x) + s·GAVA(x)        s = 0.1, SAM2 frozen
```

---

## Results

### Retinal vessels — RETA (54 unseen images, 1-shot)

| Method | Backbone | Trainable | mIoU | FB-IoU |
|---|---|---|---|---|
| PerSAM (training-free) | SAM ViT-H | 0 | 11.67 | 21.39 |
| PerSAM-F | SAM ViT-H | 2 / episode | 10.30 | 37.03 |
| SANSA universal (published) | SAM2-large | — | 14.58 | 51.21 |
| SAM2-large, no adaptation | SAM2-large | 0 | 39.69 | 66.56 |
| SAM2-tiny, no adaptation | SAM2-tiny | 0 | 42.52 | 67.74 |
| AdaptFormer, fine-tuned on FIVES | SAM2-tiny | 1.12M | 64.99 | 80.69 |
| GAVA + AdaptFormer, trained jointly | SAM2-tiny | 2.83M | 66.55 | 81.57 |
| **GAVA on a frozen AdaptFormer** | SAM2-tiny | **1.71M** | **66.71** | **81.66** |

All rows measured here on identical 1-shot episodes. PerSAM / PerSAM-F use the authors' official code.
5-shot shows the same ordering (AdaptFormer 64.92, GAVA 66.62).

![RETA comparison](output/sota_chart/reta_1shot_method_comparison.png)

### Retinal vessels — FIVES test (200 images, 1-shot)

| Method | mIoU | FB-IoU | AMD | DR | Glaucoma | Normal |
|---|---|---|---|---|---|---|
| SAM2-tiny, no adaptation | 43.42 | 68.62 | 46.97 | 42.88 | 35.86 | 47.98 |
| AdaptFormer | 74.18 | 86.05 | 76.15 | 73.76 | 72.72 | 74.09 |
| GAVA + AdaptFormer (joint) | 76.18 | 87.14 | 78.15 | 75.73 | 74.87 | 75.95 |
| **GAVA on frozen AdaptFormer** | **76.38** | **87.25** | 78.39 | 75.87 | 75.02 | 76.22 |

### Where the gain comes from

GAVA's improvement is concentrated on the thinnest vessels and on connectivity
(AdaptFormer → GAVA on a frozen AdaptFormer, 54 paired RETA images):

| Measure | AdaptFormer | GAVA | Δ | significance |
|---|---|---|---|---|
| Centreline recall, thin vessels (≤2 px, 62% of vessel length) | 47.7 | 50.8 | **+3.1** | p = 1.4e-9 |
| Centreline recall, medium (2–4 px) | 94.6 | 95.5 | +0.9 | |
| Centreline recall, thick (>4 px) | 99.5 | 99.7 | +0.1 | |
| clDice (connectivity) | 76.6 | 78.2 | **+1.6** | p = 2.4e-10 |
| Broken pieces per image (ground truth: 8.8) | 76.4 | 71.6 | −4.8 | |
| mIoU | 64.99 | 66.71 | **+1.72** | better on **54/54** images |

![Error maps](output/error_analysis/error_maps_reta_1shot.png)

### Cracks — DeepCrack test (237 unseen images, 1-shot)

| Model | Trained on FIVES | Trained on DeepCrack |
|---|---|---|
| SAM2-tiny, no adaptation | 60.54 | 60.54 |
| AdaptFormer | 55.35 | **73.82** |
| GAVA + AdaptFormer | 49.11 | 73.59 |
| GAVA on frozen AdaptFormer | 49.00 | 72.46 |

Two honest findings: the framework **transfers to a non-medical domain when retrained there**
(60.5 → 73.8, FB-IoU 86.3, close to fully supervised crack networks at 87.1), but **GAVA's gain is
vessel-specific** — it does not help on cracks, which are mostly single, non-branching lines.

---

## How GAVA works

Inside each adapted block, on the block's `[B, H, W, C]` feature map:

1. **Down-project** to a 0.3·C bottleneck (57 channels at stage 2, 115 at stage 3) + GELU.
2. **Two graph rounds**, each residual. Per round, in every 8×8 token window:
   - cosine similarity between all 64 tokens + a learnable confidence bias,
   - each token keeps its **top-8** neighbours (self and padding masked),
   - softmax over those 8 scores, weighted sum of neighbour features,
   - `node_proj(self) + edge_proj(neighbours)` → LayerNorm → stitch windows back.
3. **Up-project** back to C (zero-initialised) and scale by 0.1.

The graph is rebuilt from features on every forward pass (dynamic k-NN, EdgeConv/GAT-style), so no
pre-segmentation or predefined vessel graph is needed. Windows keep the cost linear in image size;
an earlier full-image graph ran out of memory.

![GAVA graph rounds](output/gava_graph_viz/gava_graph_rounds.png)

Code: [`models/sansa/vessel_adapter.py`](models/sansa/vessel_adapter.py), wired in at
[`models/sam2/modeling/backbones/hieradet.py`](models/sam2/modeling/backbones/hieradet.py).

---

## Setup

Environment and SAM2 weights follow the original SANSA instructions — see
[README_SANSA.md](README_SANSA.md). Extra packages used by the analysis scripts:

```bash
pip install scikit-image        # skeleton-based metrics (clDice, vessel width)
pip install timm                # only needed to run the PerSAM baseline
```

### Data layout

```
data/
├── Fives/data/fives/{images,masks}/     # 600 train_*.png, 200 test_*.png (2048²)
├── vein_reta/data/vein_reta/            # 54 images + *_vessel.png masks (1024²)
├── DeepCrack/{train_img,train_lab,test_img,test_lab}/   # 300 / 237 (544×384)
└── deepglobe_road/{images,masks}/       # 690 (1024²)
```

---

## Reproducing

Training uses 1 support + 1 query per episode, batch 2, 20 epochs, AdamW (lr 1e-4, cosine),
bf16, SAM2-tiny frozen. ~2.5 h per FIVES run on a 16 GB GPU.

```bash
# GAVA + AdaptFormer, trained jointly on FIVES
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python main.py --batch_size 2 --resume pretrain/adapter_sansa_universal.pth \
  --name_exp train_fives_adaptformer_plus_graph_stages_2_3 --data_root data \
  --dataset_file fives --adaptformer_stages 2 3 \
  --prompt mask --epochs 20 --sam2_version tiny --no_distributed

# AdaptFormer only (control): disable the graph branch
python main.py ... --vessel_adapter_stages -1

# GAVA only, on a frozen fine-tuned AdaptFormer (the clean ablation)
python main.py --resume pretrain/adaptformer_fives_stages_2_3_weights_only.pth \
  --name_exp train_fives_gava_only_frozen_adaptformer_stages_2_3 \
  --dataset_file fives --adaptformer_stages 2 3 --freeze_adapter ...

# Evaluate (1-shot) on an unseen dataset
python inference_fss.py --dataset_file vein_reta \
  --resume output/<run>/checkpoint0019.pth --name_exp eval_<name> \
  --shot 1 --adaptformer_stages 2 3 --prompt mask --sam2_version tiny --visualize
```

**DeepCrack note:** `--deepcrack_val holdout` keeps the official 237 test images unseen during
training by validating on 40 held-out training images instead.

Every command actually run in this project is recorded in [`t.txt`](t.txt).

### Flags added by this work

| Flag | Default | Effect |
|---|---|---|
| `--vessel_adapter_stages` | mirrors `--adaptformer_stages` | which stages get GAVA; `-1` disables it |
| `--freeze_adapter` | off | freeze AdaptFormer, train GAVA only |
| `--deepcrack_val` | `test` | `holdout` validates on train images, keeping the test set unseen |
| `--thin_aug` | off | zoom / inversion augmentation (negative result, see below) |

Defaults reproduce the original SANSA behaviour.

---

## Analysis tools

| Script | Produces |
|---|---|
| [`output/error_analysis/`](output/error_analysis/) | width-stratified recall, clDice, fragmentation, Wilcoxon + bootstrap CIs, error maps |
| [`output/comparison/`](output/comparison/) | side-by-side predictions and Grad-CAM (HiResCAM) figures |
| [`output/gava_graph_viz/`](output/gava_graph_viz/) | the learned graph, drawn on real features |
| [`output/metrics_explainer/`](output/metrics_explainer/) | what mIoU and FB-IoU measure, worked on a real prediction |
| [`output/sota_chart/`](output/sota_chart/) | the RETA method comparison chart |
| [`output/persam_baseline/`](output/persam_baseline/) | PerSAM / PerSAM-F baselines on RETA |
| [`docs/defense_qa.md`](docs/defense_qa.md) | Q&A sheet covering every result and caveat |

---

## Limitations

1. **Single training run per configuration** — the significance tests cover image-to-image
   variation, not seed variation.
2. **GAVA's gain is vessel-specific**: +1.7 mIoU on retina, −1.4 on cracks.
3. **Cross-domain transfer fails**: a FIVES-trained model is worse on cracks and roads than no
   adaptation at all (60.5 → 49.0 on DeepCrack).
4. **The support image is barely used** after fine-tuning: 1-shot and 5-shot scores are identical
   (66.71 vs 66.62), so the model behaves more like a fine-tuned segmenter than a true few-shot one.
5. **Augmentation did not help**: zoom + contrast-inversion augmentation (`--thin_aug`) lowered every
   score; it replaced one fixed prior with another.
6. **RETA and FIVES are the same modality** — RETA tests cross-*dataset*, not cross-domain,
   generalization.
7. **Adapters start from scratch**: the released SANSA adapter weights are for SAM2-large with
   `channel_factor 0.8` and do not load into SAM2-tiny, so only the backbone is pretrained.

### Future work

- Condition the graph on the support image, so the reference decides what to connect rather than the
  model learning one fixed vessel prior.
- Topology-aware training (clDice loss), multi-scale graph windows, and multi-seed confirmation on
  more vessel datasets (DRIVE, OCTA).

---

## Acknowledgements and citation

This repository extends **SANSA** (NeurIPS 2025 Spotlight); the original README, demos and
pretrained models are preserved in [README_SANSA.md](README_SANSA.md).

```bibtex
@inproceedings{cuttano2025sansa,
  title     = {SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation},
  author    = {Cuttano, Claudia and Trivigno, Gabriele and Averta, Giuseppe and Masone, Carlo},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```

Also builds on [SAM2](https://github.com/facebookresearch/sam2),
[AdaptFormer](https://github.com/ShoufaChen/AdaptFormer),
[VGN](https://github.com/syshin1014/VGN) (graph-based vessel segmentation),
[PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM) (baseline),
and the [FIVES](https://doi.org/10.1038/s41597-022-01564-3) and
[DeepCrack](https://doi.org/10.1016/j.neucom.2019.01.036) datasets.
