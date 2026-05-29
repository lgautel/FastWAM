# 启动

## 1. Use scripts/precompute_text_embeds.py to precompute embeddings for each training task:

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=r1_pro_chassis_uncond_3cam_384_1e-4
```

## 2. Training

```bash
bash scripts/train_zero1.sh 8 task=r1_pro_chassis_uncond_3cam_384_1e-4
```

```bash
bash scripts/train_zero1.sh 8 task=r1_pro_chassis_uncond_3cam_384_1e-4 wandb.enabled=true wandb.mode=offline
```

# WS Server

健康检查
```bash
curl -s http://127.0.0.1:8000/healthz
```