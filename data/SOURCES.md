# Demo footage

- Title: Flying through downtown Woodstock
- Creator: Jordan Koschei / ConnectHV
- Source: https://connecthv.com/drone-footage/flying-through-downtown-woodstock
- Download: https://media.connecthv.com/preview/Flying%20through%20downtown%20Woodstock.mp4
- License: CC0, as stated at https://connecthv.com/drone-footage
- Creator's account of the collection: https://connecthv.com/blog/royalty-free-hudson-valley-drone-video-for-everyone/
- Accessed: 2026-11-10
- Local file: `data/raw/woodstock-drone-preview.mp4` (ignored by Git)
- SHA-256: `2aa5e705252cde67eaaf747ed0b5ff326798251c1fd2c076c863b767ef4fc51b`
- Purpose: pretrained visual smoke test only, separate from VisDrone training and evaluation.

The download is the creator's web preview, not the full-resolution original. Use the local file's decoded dimensions and frame rate when describing it.

# Pretrained model

- Model: Ultralytics YOLO26 nano, COCO-pretrained
- Official asset: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
- Local file: `models/yolo26n.pt` (ignored by Git)
- SHA-256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- Licensing: https://www.ultralytics.com/license

# Training dataset

- Dataset: VisDrone2019-DET, AISKYEYE team at Tianjin University
- Original project: https://github.com/VisDrone/VisDrone-Dataset
- Dataset configuration/conversion: the unmodified `VisDrone.yaml` bundled with the pinned Ultralytics release
- Archive mirror: `https://github.com/ultralytics/assets/releases/download/v0.0.0/`, using `VisDrone2019-DET-train.zip`, `VisDrone2019-DET-val.zip`, and `VisDrone2019-DET-test-dev.zip`
- Local root: `datasets/VisDrone/` (ignored by Git)
- Download sizes, archive checksums, CRC verification, extracted size, and split counts: `results/phase1-setup/dataset-manifest.json`

The upstream project provides a research dataset; its linked README does not include a standalone license grant. This repository does not redistribute the images, annotations, or archives. The CC0 description above applies only to the separate ConnectHV demo clip.
