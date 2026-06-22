# FastWAM WebSocket 推理性能优化说明

记录 `bt/fastwam_ws_server.py` 单帧推理从 **~320ms 优化到 ~80ms（约 4×）** 的诊断过程、改动与用法。

---

## 1. 背景 / 问题

FastWAM 的 WS 推理服务单帧约 **300ms**，而对照组 OldPi 的 Pi0.5 服务（`OldPi/pi/openpi/run_ws_server.sh`）只要 ~50ms。

两个 server 的 WebSocket / msgpack / `asyncio.to_thread` 协议代码几乎一致，**差距不在 server 层**，而在模型本身：

- **Pi0.5**：PaliGemma VLA + 小 action expert，跑在 **JAX/XLA**，整图 JIT 编译融合。
- **FastWAM**：基于 **Wan2.2-TI2V-5B** 的世界模型（MoT = 5B 视频专家 + 1B 动作专家），纯 **eager PyTorch bf16**。

---

## 2. 诊断结论

在 `infer_action` 内按阶段加 CUDA 同步计时（见下文 `FASTWAM_PROFILE`），优化前拆分：

```
[fastwam_profile] vae_encode=5.8ms ctx_prep=0.1ms video_prefill=27.8ms action_loop(x10)=295.5ms | total=329.2ms
```

**90% 的时间都在 10 步动作去噪循环**，每步约 29.5ms。而动作专家只有 32 个 token、1024 hidden、30 层，单层真实算力是微秒级——说明 **GPU 在空转，瓶颈是 eager 模式下 Python 派发 + kernel launch 的 CPU 开销**（每步约上千次 kernel launch）。

这是 CUDA graph / `torch.compile(reduce-overhead)` 的典型适用场景：动作循环每步 shape 完全固定，只有 `latents_action` 和 `timestep` 在变。

> 顺带澄清：服务日志里 `response ... total_ms=1600+ms` 那种大数字是 server 在 `await ws.recv()` 等客户端发下一帧的**墙钟时间**，不是计算耗时。文本编码（UMT5-XXL）也不是瓶颈——`total_ms` 与 `model_infer_ms` 仅差几 ms。

---

## 3. 改动清单

### 3.1 `src/fastwam/models/wan22/fastwam.py`

- 新增 `_PhaseTimer`（带 CUDA 同步的分阶段计时），由环境变量 `FASTWAM_PROFILE` 开关；未启用时零开销。
- 在 `infer_action` 内对 `vae_encode` / `ctx_prep` / `video_prefill` / `action_loop` 四段打点。
- 新增 `_get_action_step_fn()`：启用 `FASTWAM_COMPILE_ACTION` 时，用
  `torch.compile(self._predict_action_noise_with_cache, mode="reduce-overhead")`
  惰性编译动作去噪单步并缓存（借 CUDA graph 合并 kernel launch，**不改数值结果**）。
  动作循环改为调用该函数；不启用时行为与原来完全一致。

### 3.2 `src/fastwam/models/wan22/action_dit.py`（cudagraph 关键修复）

RoPE 频率原本是普通属性 `self.freqs = precompute_freqs_cis(...)`，**不会被 `model.to(cuda)` 搬运**，常驻 CPU。`pre_dit` 里 `self.freqs[:seq_len].to(tokens.device)` 在编译图中引入了 CPU 张量，导致 Inductor 直接放弃 cudagraph：

```
skipping cudagraphs due to cpu device (arg4_1) ... action_dit.py:286 ...
```

改为注册成**非持久 buffer**，让它随 `.to(device)` 上 GPU，编译区内不再有 CPU 张量：

```python
self.register_buffer("freqs", precompute_freqs_cis(attn_head_dim, end=1024), persistent=False)
```

`persistent=False` 不进 `state_dict`，不影响 checkpoint 加载。

### 3.3 `bt/fastwam_ws_server.py`（附带，非瓶颈）

`_encode_prompt` 增加按 prompt 字符串缓存 `(context, context_mask)`，prompt 不变时不再每帧重跑 UMT5-XXL（开销很小，属顺手优化）。

---

## 4. 用法

两个环境变量开关，默认全部关闭（= 原始行为）：

| 变量 | 作用 |
|---|---|
| `FASTWAM_PROFILE=1` | 打印 `infer_action` 分阶段耗时（CUDA 同步计时） |
| `FASTWAM_COMPILE_ACTION=1` | 用 `torch.compile(reduce-overhead)` 编译动作单步，启用 CUDA graph |

启动示例：

```bash
FASTWAM_PROFILE=1 FASTWAM_COMPILE_ACTION=1 bash bt/run_ws_server.sh
```

注意事项：
- **首次调用（warmup）会触发编译，耗时数十秒**，属正常；请保持 warmup 开启。
- cudagraph 模式下前 1–2 个真实帧可能仍在录制/预热，**第 3 帧往后才是稳态**。
- 需要 PyTorch 2.x + CUDA（实测 torch 2.7.1 / H200）。

---

## 5. 结果

| 阶段 | 优化前 (eager) | 优化后 (cudagraph) |
|---|---|---|
| vae_encode | 5.8 ms | 5.7 ms |
| video_prefill | 27.8 ms | 28.0 ms |
| **action_loop (×10)** | **295.5 ms** | **40.0 ms** |
| **total (infer)** | **~318 ms** | **~80 ms** |

动作循环 **~7.4×**，整体模型推理 **~4×**，已与 Pi0.5 的 ~50ms 同一量级。

> 上线前建议跑一遍 eval / 对比动作输出，确认 cudagraph + 复数 rope eager 回退路径下数值与优化前一致。

---

## 6. 后续可选优化（收益递减）

1. **`video_prefill`（~28ms）**：同样可对 5B 视频专家的单次预填充上 `torch.compile`。
2. **复数 RoPE**：`rope_apply` 用了复数 + float64，Inductor 不能 codegen 会回退 eager。改写成等价的**实数版 cos/sin 旋转**可让 Inductor 完整融合，进一步压低 `action_loop`。
3. **减少 `NUM_INFERENCE_STEPS`**：线性提速，但影响精度，需权衡（`NUM_INFERENCE_STEPS=5 bash bt/run_ws_server.sh`）。
