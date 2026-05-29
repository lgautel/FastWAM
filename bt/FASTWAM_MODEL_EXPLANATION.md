# FastWAM 模型结构与推理流程说明

这份文档根据当前代码实现整理，重点解释 FastWAM 的输入输出、模型结构、ActionDiT、MoT、KV cache、flow matching 采样，以及 websocket 部署时的数据包格式。

相关代码主要在：

- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/action_dit.py`
- `src/fastwam/models/wan22/mot.py`
- `src/fastwam/models/wan22/schedulers/scheduler_continuous.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `bt/fastwam_ws_server.py`

## 1. 一句话概括

FastWAM 不是传统的 “图像 + prompt 进 VLM，再接 action head” 的 VLA 模型。

它更像是：

```text
Wan2.2 视频生成模型 + ActionDiT 动作扩散模型 + MoT mixed attention
```

整体流程是：

```text
图像 / prompt / proprio
      ↓
Wan video expert 提供视觉 token
ActionDiT 负责动作 token denoise
      ↓
MoT 让 action token attend 到 video token
      ↓
flow matching 逐步把随机 action noise 积分成动作
      ↓
输出未来一段 23 维动作序列
```

## 2. 当前任务的输入输出

以 `r1_pro_chassis` 配置为例，数据配置在 `configs/data/r1_pro_chassis.yaml`。

### 2.1 输入图像

训练数据有三路相机：

```text
head_rgb:        raw [3, 360, 640]
left_wrist_rgb:  raw [3, 480, 640]
right_wrist_rgb: raw [3, 480, 640]
```

模型最终吃的是拼好的单张图：

```text
[3, 384, 320]
```

`robotwin` 拼图规则是：

```text
head_rgb        -> 256 x 320
left_wrist_rgb  -> 128 x 160
right_wrist_rgb -> 128 x 160

left_wrist_rgb + right_wrist_rgb 横向拼成 128 x 320
再拼到 head_rgb 下面

最终图像: 384 x 320
```

所以 websocket 部署时，为了节省带宽，客户端可以直接发：

```text
head_rgb:        256 x 320 x 3 uint8
left_wrist_rgb:  128 x 160 x 3 uint8
right_wrist_rgb: 128 x 160 x 3 uint8
```

服务端会再做一次 resize / normalize，最终喂给模型：

```text
input_image: [1, 3, 384, 320]
```

### 2.2 proprio / state

`proprio` 是 proprioception，本体感知，也就是机器人当前自身状态。

在当前任务里，它就是 23 维 `state`：

```text
state / proprio: [23]
```

大致可以理解为：

```text
0-6:   left_arm
7-13:  right_arm
14:    left_gripper
15:    right_gripper
16-22: chassis
```

训练数据里它叫 `state`，经过 processor 后会变成 `proprio`。

模型内部只取当前第一步 proprio：

```text
proprio[:, 0, :]  # [B, 23]
```

然后通过一个线性层变成 context token：

```text
proprio [B, 23] -> Linear(23, 4096) -> proprio token [B, 1, 4096]
```

这个 proprio token 会拼到 prompt context 后面。

### 2.3 prompt

prompt 会先被包装成 Wan 风格文本：

```text
A video recorded from a robot's point of view executing the following instruction: {task}
```

然后经过 tokenizer + T5 text encoder，得到：

```text
context:      [B, L, 4096]
context_mask: [B, L]
```

如果有 proprio，则变成：

```text
context = [prompt tokens, proprio token]
```

### 2.4 输出动作

模型输出一段 action chunk：

```text
actions: [T, 23]
```

其中 `T = action_horizon`。

例如：

```text
action_horizon = 16 -> actions [16, 23]
action_horizon = 32 -> actions [32, 23]
```

服务端 websocket 下发的是已经反归一化后的物理动作，不是 normalized action。

## 3. 训练和推理的区别

### 3.1 训练时

训练时模型需要视频序列和动作序列。

当前配置：

```text
num_frames = 33
action_video_freq_ratio = 4
```

视频帧采样为：

```text
0, 4, 8, ..., 32
```

所以训练时 video shape 类似：

```text
video:  [B, 3, 9, 384, 320]
action: [B, 32, 23]
proprio:[B, 32, 23]
```

训练目标包括两个分支：

```text
loss_total = lambda_video * loss_video + lambda_action * loss_action
```

当前配置里 `lambda_action=1.0`，`lambda_video` 没显式写，代码默认也是 `1.0`。

### 3.2 推理时

推理动作时只需要当前一帧图像：

```text
input_image: [1, 3, 384, 320]
proprio:     [1, 23]
context:     prompt/proprio context
```

推理时不需要未来视频，也不需要 GT action。

## 4. FastWAM 的核心模块

FastWAM 在 `src/fastwam/models/wan22/fastwam.py` 中定义。

核心成员包括：

```text
video_expert      # Wan2.2 video DiT expert
action_expert     # ActionDiT
mot               # Mixture-of-Transformers
vae               # Wan VAE
text_encoder      # T5 text encoder，可选加载
tokenizer         # T5 tokenizer，可选加载
proprio_encoder   # Linear(proprio_dim, text_dim)
train/infer schedulers
```

配置入口是 `configs/model/fastwam.yaml`：

```yaml
_target_: fastwam.runtime.create_fastwam
model_id: Wan-AI/Wan2.2-TI2V-5B
tokenizer_model_id: Wan-AI/Wan2.1-T2V-1.3B
proprio_dim: ${data.train.processor.proprio_output_dim}
```

整体结构可以画成：

```text
prompt
  ↓ T5
context tokens ─────────────┐
                            │
proprio -> Linear -> token ─┤
                            ↓
image -> VAE -> video_expert.pre_dit -> video tokens
                            ↓
random action latent -> action_expert.pre_dit -> action tokens
                            ↓
                     MoT mixed attention
                            ↓
              action_expert.post_dit -> pred velocity
                            ↓
                    scheduler.step(...)
                            ↓
                       final actions
```

## 5. ActionDiT 结构

`ActionDiT` 在 `src/fastwam/models/wan22/action_dit.py`。

当前配置：

```yaml
action_dim: 23
hidden_dim: 1024
ffn_dim: 4096
num_heads: 24
attn_head_dim: 128
num_layers: 30
text_dim: 4096
freq_dim: 256
```

可以理解为一个动作序列 DiT：

```text
action [B, T, 23]
    ↓ Linear(23 -> 1024)
action tokens [B, T, 1024]
    ↓ 30 x DiTBlock
hidden tokens [B, T, 1024]
    ↓ Linear(1024 -> 23)
pred velocity [B, T, 23]
```

### 5.1 pre_dit 的作用

`pre_dit()` 是进入 DiT/MoT 前的编码和准备阶段。

它做的事情包括：

```text
1. action_tokens [B,T,23] -> action hidden tokens [B,T,1024]
2. context [B,L,4096] -> context_emb [B,L,1024]
3. timestep -> sinusoidal embedding -> t_mod [B,6,1024]
4. 准备 RoPE 位置编码 freqs
5. 准备 context attention mask
```

所以它不是最终推理，只是把输入整理成 transformer 能吃的形式。

### 5.2 DiTBlock 内部

ActionDiT 复用 Wan video DiT 的 `DiTBlock`。

每层结构大致是：

```text
x
 ↓ LayerNorm + timestep shift/scale
self-attention + gate
 ↓
cross-attention to context
 ↓
LayerNorm + timestep shift/scale
FFN + gate
 ↓
x_next
```

其中 context 是 prompt/proprio 条件。

### 5.3 注意力维度

ActionDiT 自身 hidden dim 是 1024，但 Q/K/V 的 attention 维度是：

```text
num_heads * attn_head_dim = 24 * 128 = 3072
```

这很关键，因为 Wan video expert 的 attention space 也是 3072。这样 ActionDiT 和 Wan video expert 才能在 MoT 中做 mixed attention。

## 6. prompt 给谁用

prompt 不是只给 VLM，也不是只给 Wan。

它先通过 T5 编成 `context`，然后同时给：

```text
video_expert.pre_dit(..., context, context_mask)
action_expert.pre_dit(..., context, context_mask)
```

也就是说：

```text
Wan video expert 可以 cross-attend prompt/proprio
ActionDiT 也可以 cross-attend prompt/proprio
```

这和很多 VLA 不太一样。

很多 VLA 可能是：

```text
image + prompt -> VLM -> hidden state -> action head
```

FastWAM 更像：

```text
image -> video token
prompt/proprio -> condition token
noisy action -> action token
video/action token 通过 MoT 混合
ActionDiT 在每一步 denoise 中直接读取 prompt/proprio
```

## 7. MoT 的作用

MoT 是 Mixture-of-Transformers，在 `src/fastwam/models/wan22/mot.py`。

它不是简单地把 Wan 输出接到 ActionDiT 前面，而是在 transformer attention 层面混合 video/action token。

训练时，MoT 会同时处理：

```text
video tokens
action tokens
```

每一层里：

```text
1. video expert 产生 video Q/K/V
2. action expert 产生 action Q/K/V
3. 拼接 Q/K/V 做 mixed attention
4. mixed attention 输出再切回 video/action 两部分
5. 各自走对应 expert 的 cross-attn 和 FFN
```

所以可以理解为：

```text
ActionDiT 不是独立生成动作；
它在 MoT 里通过 attention 读取 Wan video token。
```

更准确地说：

```text
Wan 提供视觉 token / world model 表征
MoT 让 action token attend 到这些视觉 token
ActionDiT 根据视觉、prompt、proprio 条件生成动作
```

## 8. 推理时 Wan/video 分支只算一次

在 `infer_action()` 中，video 分支只针对当前请求计算一次。

流程是：

```text
input_image
  ↓ VAE encode
first_frame_latents
  ↓ video_expert.pre_dit
video tokens
  ↓ mot.prefill_video_cache
video_kv_cache
```

`video_kv_cache` 保存的是每一层 video branch 的 K/V：

```text
[
  {"k": layer0_video_k, "v": layer0_video_v},
  {"k": layer1_video_k, "v": layer1_video_v},
  ...
]
```

后续 action denoise 的每一步都会复用这个 cache，不重复跑 Wan/video 分支。

注意：这个 cache 只对当前一次请求有效。下一次 websocket 请求换了新图像、prompt 或 proprio，就会重新算新的 video KV cache。

## 9. 推理时 action latent 从哪里来

推理时的 action latent 来自随机高斯噪声：

```python
latents_action = torch.randn(
    (1, action_horizon, action_dim)
)
```

当前任务里：

```text
latents_action: [1, T, 23]
```

例如：

```text
T=16 -> [1, 16, 23]
T=32 -> [1, 32, 23]
```

它不是从历史动作编码来的，而是从纯噪声开始，通过 flow matching 一步步积分成动作。

## 10. action expert 如何使用 video KV cache

每一个 denoise step 中，流程是：

```text
当前 latents_action
  ↓ action_expert.pre_dit
action tokens
  ↓ mot.forward_action_with_video_cache
updated action tokens
  ↓ action_expert.post_dit
pred velocity
  ↓ scheduler.step
下一步 latents_action
```

在 `forward_action_with_video_cache()` 内部，每层做：

```text
1. 当前 action tokens 生成 action Q/K/V
2. 取当前层 cached video K/V
3. 拼接:
   K = [video_K_cache, action_K]
   V = [video_V_cache, action_V]
4. 用 action_Q attend 到 [video + action]
5. 得到 mixed attention 后的 action token
6. 再走 action block 的 cross-attn / FFN
```

所以可以说：

```text
Action expert 每一步都借助 video KV cache 做 MoT mixed attention。
MoT 的 action-token 输出再经过 ActionDiT head，预测 flow matching velocity。
```

## 11. post_dit 和 scheduler.step 在干什么

### 11.1 post_dit

`post_dit()` 只是一个输出头：

```text
action_tokens [B,T,1024]
  ↓ Linear(1024 -> 23)
pred_action [B,T,23]
```

这里的 `pred_action` 不是最终动作，而是 flow matching 的预测速度，也就是 vector field。

可以理解为：

```text
pred_velocity = ActionDiT(latents_action, timestep, image/prompt/proprio)
```

### 11.2 scheduler.step

当前 scheduler 是 `WanContinuousFlowMatchScheduler`。

训练时构造 noisy sample：

```text
x_sigma = (1 - sigma) * x_clean + sigma * noise
```

训练目标是：

```text
target = noise - x_clean
```

也就是从 clean 指向 noise 的速度场。

推理时从 `sigma=1` 走到 `sigma=0`，所以 `delta` 通常是负数：

```text
delta = sigma_next - sigma_current
```

更新公式就是：

```text
actions = actions + delta * pred_velocity
```

代码等价于：

```python
sample = sample + model_output * delta
```

由于 `delta < 0`，而 `pred_velocity ≈ noise - clean`，所以积分方向实际是从 noise 往 clean 走。

完整采样可以写成：

```text
latents_action = randn([1,T,23])

for each denoise step:
    pred_velocity = model(latents_action, sigma, condition)
    latents_action = latents_action + delta_sigma * pred_velocity

final action = latents_action
```

## 12. websocket 部署数据流

`bt/fastwam_ws_server.py` 的在线推理流程是：

```text
客户端机器人:
  head_rgb
  left_wrist_rgb
  right_wrist_rgb
  state [23]
  prompt
      ↓ msgpack websocket

云服务器:
  resize / 拼图 / normalize 图像
  normalize state -> proprio
  prompt -> T5 context
      ↓
  FastWAM.infer_action()
      ↓
  normalized actions [T,23]
      ↓
  processor backward 反归一化
      ↓
  下发 actions [T,23]
```

返回数据包：

```python
{
    "actions": np.ndarray,  # [T, 23], float32, denormalized
    "policy_timing": {
        "infer_ms": float,
    },
    "server_timing": {
        "infer_ms": float,
        "prev_total_ms": float,  # 可选
    },
}
```

## 13. 和常见 VLA 的区别

常见 VLA 通常可以简化为：

```text
image + prompt
  ↓ VLM backbone
hidden state
  ↓ action head
action
```

FastWAM 更像：

```text
image -> Wan video token
prompt/proprio -> context token
random action noise -> ActionDiT token
  ↓
MoT mixed attention
  ↓
flow matching denoise
  ↓
action chunk
```

主要区别：

1. Prompt 不只是给 VLM，而是作为 context 直接给 video expert 和 ActionDiT。
2. 动作不是一次回归出来，而是从随机噪声通过 flow matching 多步采样出来。
3. Wan/video 分支提供视觉 token，ActionDiT 通过 MoT 读取视觉 token。
4. 输出是一段 action chunk，而不是单步动作。
5. 训练时同时学习 video latent 和 action latent 的生成目标。

## 14. 最简心智模型

如果只保留最重要的理解，可以这样记：

```text
FastWAM = Wan 视觉表征 + ActionDiT 动作扩散 + MoT 跨模态 attention
```

推理时：

```text
1. 当前图像经过 Wan/video branch，得到 video KV cache
2. prompt/proprio 变成 context，给 video/action 两边做 cross-attention
3. action 从随机噪声开始
4. 每一步 ActionDiT 都通过 MoT attend 到 video KV cache
5. ActionDiT head 输出 flow matching velocity
6. scheduler 用 actions = actions + delta * velocity 更新
7. 多步之后得到最终动作 chunk [T,23]
```

一句话总结：

> FastWAM 用 Wan 的视频生成能力提供视觉/world-model token，用 ActionDiT 在这些条件下做动作 flow matching 生成，MoT 是连接视频 token 和动作 token 的关键机制。
