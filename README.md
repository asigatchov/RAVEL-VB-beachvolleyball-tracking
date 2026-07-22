# TAPe-VB

TAPe-VB is a compact temporal detector for volleyball video. It detects
players and the ball, associates player regions across frames, and can render
an annotated video. The repository contains both OpenVINO and PyTorch
inference paths.

<p align="center">
  <img src="demo/ravel_vb_demo.gif" alt="TAPe-VB player and ball detection demo" width="960">
</p>

## Contents

```text
src/
├── model.py       checkpoint-compatible TAPe-VB model
├── backbone.py    model dependency
├── ball_grid.py   ball head dependency
└── postprocess.py decoding and player hysteresis
infer_torch.py    PyTorch .pt inference
infer_openvino.py OpenVINO inference
models/           example .pt and OpenVINO models
```

## Installation

Python 3.10+ is recommended. Install the dependencies with:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

For PyTorch inference, install a PyTorch build appropriate for the target
machine as well (CPU example: `uv pip install torch`).

## PyTorch quick start

Both bundled checkpoints can be run directly. Their clip lengths are encoded
in the checkpoint and are validated during startup:

```bash
uv run python infer_torch.py \
  models/ravel_vb_beach_player_ball_v1_9seq.pt input.mp4 \
  --output out/predictions_9.json \
  --output-video out/result_9.mp4

uv run python infer_torch.py \
  models/ravel_vb_beach_player_ball_v1_18seq.pt input.mp4 \
  --output out/predictions_18.json
```

Useful options include `--stride`, `--score-threshold`, `--close-threshold`,
`--no-ball`, `--device`, `--cpu-threads`, and `--show`. Run
`uv run python infer_torch.py --help` for the complete interface.

The output JSON contains source-frame pixel boxes, class names, confidence
scores, and short-term player track IDs. The model expects RGB clips resized
to 512×288; the 9-frame and 18-frame variants must use their matching clip
length.

## OpenVINO quick start

```bash
uv run python infer_openvino.py input.mp4 \
  --model models/ravel_vb_v1.xml \
  --output out/predictions.json \
  --output-video out/result.mp4
```

## Model notes

The included PyTorch checkpoints use architecture version 19 and are intended
for volleyball footage. Very small or blurred balls, occlusion, camera cuts,
and footage unlike the training data can reduce accuracy. The `ravel_vb_*`
filenames are retained only as checkpoint compatibility names; the published
working model name is TAPe-VB.
