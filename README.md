# DROP

**Decouple Re-Identification and Human Parsing with Task-specific Features for Occluded Person Re-identification**
[Paper (arXiv:2401.18032)](https://arxiv.org/abs/2401.18032) · IEEE TBIOM 2026

This repository contains the HRNet-W32 + FPN image-based DROP implementation, with training, evaluation, human-parsing mask transforms, and dataset configurations. The part-based representation learning framework is adapted from BPBreID/Torchreid. The legacy `model.bpbreid` configuration namespace and `bpbreid` model alias are retained for compatibility; new configurations select `model.name: drop`.

## Installation

Use Python 3.10–3.12 with the pinned dependencies below. Create an isolated environment and install a matching PyTorch/torchvision pair for your machine before installing this project. For example, a CPU environment for the smoke tests:

```bash
git clone https://github.com/shuguang-52/2026-TBIOM-DROP.git
cd 2026-TBIOM-DROP
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
```

For training, install the corresponding CUDA builds from [PyTorch](https://pytorch.org/get-started/locally/) instead of the CPU wheels. The dependency bounds preserve the Albumentations 1.x and TorchMetrics 0.10 APIs used by this source. Logs and visualizations are saved locally by default. External experiment trackers are optional (`requirements-logging.txt`) and must be enabled explicitly; credentials belong in your environment.

## Data and pretrained backbone

Dataset images, human-parsing labels, and model weights are not bundled. Obtain the datasets and the [BPBreID human-parsing labels](https://github.com/VlSomers/bpbreid#download-human-parsing-labels) separately. Place each label archive's `masks/pifpaf_maskrcnn_filtering/` folder inside its corresponding dataset directory, retaining the train/query/gallery subfolders.

Use the BPBreID dataset layout rather than assuming raw dataset archives already match these loaders. The MSMT17 configuration expects the V1 mask layout.

Set `--root` to the directory expected by the dataset loader, containing its named dataset folder. The exact folder names are defined in `torchreid/data/datasets/image/`. For example, the Market1501 loader uses:

```text
reid-data/
└── Market-1501-v15.09.15/
    ├── bounding_box_train/
    ├── bounding_box_test/
    ├── query/
    └── masks/pifpaf_maskrcnn_filtering/
```

Download the HRNet-W32 ImageNet backbone from [HRNet Image Classification](https://github.com/HRNet/HRNet-Image-Classification). The loader expects `pretrained_models/hrnetv2_w32_imagenet_pretrained.pth`; use `model.bpbreid.hrnet_pretrained_path` to change the containing directory. `model.pretrained False` is available for random-initialization checks, but is not the pretrained training setup.

## Training

Run commands from the repository root:

```bash
python -m scripts_reid.main --config-file configs/drop/drop_occ_duke_train.yaml --root /path/to/reid-data
python -m scripts_reid.main --config-file configs/drop/drop_market1501_train.yaml --root /path/to/reid-data
```

Configurations are provided for Occluded-Duke, Occluded-ReID, P-DukeMTMC, Market1501, DukeMTMC-ReID, and MSMT17. Their dataset-specific image sizes, masks, and loss weights are specified in `configs/drop/`. Override settings at the end of the command, for example `data.workers 4 train.batch_size 32`. On Linux, prefix a command with `CUDA_VISIBLE_DEVICES=0` to select a GPU.

## Evaluation

Supply a trained DROP checkpoint explicitly:

```bash
python -m scripts_reid.main --config-file configs/drop/drop_occ_duke_test.yaml --root /path/to/reid-data model.load_weights /path/to/drop_checkpoint.pth.tar
```

For another dataset, use its training config and append `test.evaluate True model.load_weights /path/to/checkpoint.pth.tar model.pretrained False`. Use the same model, part grouping, and image dimensions as the checkpoint. External parsing labels supervise training; evaluation uses predicted attention masks by default. Missing evaluation checkpoints are rejected instead of evaluating random weights. Loading checkpoints with incompatible architectures is not a way to reproduce the paper results.

## Source layout and validation

- `torchreid/models/bpbreid.py`: DROP part representations and parsing head.
- `torchreid/models/hrnet.py`: HRNet backbone and FPN task-specific features.
- `torchreid/engine/image/part_based_engine.py`: training and visibility-aware evaluation.
- `torchreid/losses/`: identity/triplet and parsing objectives.
- `scripts_reid/` and `configs/drop/`: training/evaluation entry point and configurations.

This source-only release removes unrelated research branches and fixes execution defects in the supplied working tree: FPN tensor returns, feature-channel alignment, device placement, and task-prompt resizing for the configured image sizes. Existing parameter names/shapes are retained where possible. Checkpoint equivalence and published accuracy must be validated with the original trained weights and datasets; no weights or new benchmark results are claimed here.

```bash
python -m unittest discover -s tests -v
python -m scripts_reid.main --help
```

The default evaluator uses Python/NumPy. The retained Cython source is optional and not required for installation; precompiled binaries are not distributed.

## Citation

```bibtex
@article{dou2024drop,
  title={DROP: Decouple Re-Identification and Human Parsing with Task-specific Features for Occluded Person Re-identification},
  author={Dou, Shuguang and Jiang, Xiangyang and Tu, Yuanpeng and Gao, Junyao and Qu, Zefan and Zhao, Qingsong and Zhao, Cairong},
  journal={arXiv preprint arXiv:2401.18032},
  year={2024}
}
```

## Acknowledgment and license

Thanks to [BPBreID](https://github.com/VlSomers/bpbreid), [Torchreid](https://github.com/KaiyangZhou/deep-person-reid), and [HRNet](https://github.com/HRNet/HRNet-Image-Classification). Please also cite their work when applicable. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for retained upstream terms and attribution.
