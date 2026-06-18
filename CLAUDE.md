# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastWAM (Fast World Action Model) is a robotic control model that generates action sequences via flow matching, combining a Wan2.2-TI2V-5B video generation backbone with an ActionDiT (action diffusion transformer) fused through MoT (Mixture-of-Transformers). Unlike standard VLA models, it does not use an autoregressive language model — actions are sampled iteratively from noise using a continuous flow-matching scheduler.

Paper: https://arxiv.org/abs/2603.16666

## Common Commands

### Environment Setup
```bash
conda create -n fastwam python=3.10 -y && conda activate fastwam
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

### Preprocessing (required before first training)
```bash
# Preprocess ActionDiT backbone from Wan2.2 pretrained weights
python scripts/preprocess_action_dit_backbone.py --model-config configs/model/fastwam.yaml

# Cache T5 text embeddings to speed up data loading
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```

### Training
```bash
# 8-GPU training with DeepSpeed ZeRO-1
bash scripts/train_zero1.sh

# Multi-node ZeRO-2 training
bash scripts/train_zero2.sh

# Direct Hydra invocation (single-GPU or custom)
python scripts/train.py task=libero_uncond_2cam224_1e-4
```

### Evaluation
```bash
# LIBERO benchmark (multi-GPU)
python experiments/libero/run_libero_manager.py task=libero_uncond_2cam224_1e-4 ckpt=<path>

# RoboTwin benchmark (multi-GPU)
python experiments/robotwin/run_robotwin_manager.py task=robotwin_uncond_3cam_384_1e-4 ckpt=<path>

# Summarize LIBERO results
python experiments/libero/summarize_results.py <eval_dir>
```

## Architecture

### Model Stack

```
FastWAM = Video Expert (WanVideoDiT) + Action Expert (ActionDiT) + MoT Mixer
```

**Training flow**: Image → VAE → video latents; Prompt → T5 → text context; State → linear → token. Random noise → ActionDiT. MoT fuses video and action tokens via mixed cross-attention at each of 30 transformer layers. Loss is MSE on predicted velocity (flow matching).

**Inference flow**: Video branch computed once and cached as KV. Action sampled iteratively from noise via flow matching (10-20 denoising steps). Output is an action chunk of shape `[T, action_dim]`.

Only the DiT (MoT layers) and optional proprio_encoder are trainable — all other components (VAE, T5, video expert weights) are frozen.

### Key Source Layout

- `src/fastwam/models/wan22/fastwam.py` — main FastWAM model class with `infer_action()` for inference
- `src/fastwam/models/wan22/action_dit.py` — ActionDiT: 30-layer transformer (1024 hidden, 3072 attn dim)
- `src/fastwam/models/wan22/mot.py` — MoT: mixed attention between video and action experts
- `src/fastwam/models/wan22/wan_video_dit.py` — WanVideoDiT: video diffusion transformer
- `src/fastwam/models/wan22/wan_video_vae.py` — WanVideoVAE38: image/video latent encoder
- `src/fastwam/models/wan22/wan_video_text_encoder.py` — T5-based text encoder
- `src/fastwam/models/wan22/schedulers/` — WanContinuousFlowMatchScheduler
- `src/fastwam/trainer.py` — Wan22Trainer: main training loop with gradient accumulation, checkpointing, eval
- `src/fastwam/runtime.py` — `create_fastwam()` / `create_wan22_model()` factory functions
- `src/fastwam/datasets/lerobot/robot_video_dataset.py` — RobotVideoDataset (LeRobot format)
- `src/fastwam/datasets/lerobot/transforms/` — action/state normalization, rotation, image transforms

### Configuration System

Hydra with composable YAML configs under `configs/`:
- `configs/train.yaml` — base training config (batch size, LR, scheduler, wandb, mixed precision)
- `configs/model/` — model architecture configs (fastwam, fastwam_idm, fastwam_joint)
- `configs/data/` — dataset configs (libero_2cam, robotwin, r1_pro_chassis)
- `configs/task/` — task-level configs combining data + model + hyperparams

Task configs are the primary entry point for Hydra CLI (e.g., `task=libero_uncond_2cam224_1e-4`).

### Training Infrastructure

- Distributed training via Accelerate + DeepSpeed (ZeRO-1/ZeRO-2)
- Mixed precision: bfloat16
- Accelerate configs in `scripts/accelerate_configs/`, DeepSpeed configs in `scripts/ds_configs/`
- Checkpoints saved to `runs/{task_name}/{run_id}/checkpoints/` (weights + optimizer state)
- Experiment tracking via Weights & Biases

### Supported Datasets and Robots

- **LIBERO**: 4 suites (spatial, object, goal, 10-task), 2-camera, action_dim=7
- **RoboTwin**: dual-arm manipulation, 3-camera, action_dim=23
- **R1 Pro Chassis**: wheeled mobile manipulator, 3-camera, action_dim=23
- All use LeRobot-format HuggingFace datasets

### Model Variants

- `fastwam.py` / `mot.py` — standard version (video_expert and action_expert as nested modules)
- `fastwam2.py` / `mot2.py` — alternate version without nested module duplication (renamed backups, not primary)
- `fastwam_idm.py` — Inverse Dynamics Model variant
- `fastwam_joint.py` — Joint training variant (video + action losses simultaneously)

## Deployment

WebSocket server for online policy inference: `bt/fastwam_ws_server.py`
RoboTwin policy deployment interface: `experiments/robotwin/fastwam_policy/deploy_policy.py`

## Key Dependencies

PyTorch 2.7.1 (CUDA 12.8), Accelerate, DeepSpeed, HuggingFace Transformers (T5 text encoder), Hydra, einops, safetensors, wandb

## 写作时要注意的 

- 要写科普论文风格的技术文章, 要在纵向(时间维度,同类算法的历史发展维度)和横向(同时期的同类或类似算法)上做互相比较. 要严谨, 要符合实际.
- 画图表用mermaid, 数学相关的用LaTex.
- 可通过搜索来扩大思考范围与可参考的信息
- 可以用以下方法读取 pdf:
    - 用 python 代码调用 pypdf 包
    - 或者读取pdf对应的 `*_TeX_Source`,`TeX_Source`, `TeXSource` 或 `*_html` 文件夹里的内容, 也是一样的.
- 涉及计算的尽量通过编码或调用工具解决

## 关于原著的资料
- huggingface上的模型在: https://huggingface.co/yuanty/fastwam , huggingface上的数据集在: https://huggingface.co/datasets/yuanty/LIBERO-fastwam 和 https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam 
- 项目主页在: https://yuantianyuan01.github.io/FastWAM/ 
- 可以在网上搜索可信的比较相关的论文, 文章, 评论等等作为参考
- `b\d\FastWAM.pdf`是论文[Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/html/2603.16666v2)
- 它的github在: https://github.com/yuantianyuan01/FastWAM . 由于代码已经在本地, 所以一般不在github搜索和抓取代码, 而在github上搜索和抓取的一般是Commits, Issues, Pull Requests等等, 以及它的github上引用的其它链接url也需要关注.