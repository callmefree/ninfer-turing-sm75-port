# NInfer 三元内核 → Tesla T10 (sm_75 / Turing) 移植推进计划

> 状态：**M0 进行中** ｜ 负责人：callmefree ｜ 创建：2026-09-28 ｜ 最近更新：2026-09-28
> 上游基线：`paicat1/Bonsai-27B-NInfer` @ `engine-main`（fork 自 `Ambolio/ninfer-4090-windows` v1.0.8）
> 目标设备：**Tesla T10 (TU102, sm_75, 16GB)**，宿主 fn36（Debian 12 bookworm / Linux + T10 卡）

---

## 0. 一句话目标

把上游引擎的三元 Bonsai-2-27B 推理能力从 sm_89 / 120a 移植到 **Turing sm_75**，使**单张 T10 (16GB) 能跑 ninfer 三元 27B**——既保住 2.125bpw 高密度（9.81GiB 装得进 16GB），又保住 ninfer 的 decode 吞吐优势（W8A16 社区 fork 27B ≈ 16.3GiB 装不下 T10，这正是必须走三元的根因）。

---

## 1. 决策依据（动手前必读的 5 个事实）

1. **改打包器没用，要改引擎内核**：`ninfer-ada-ternary` + `pack.py`（在 `main` 分支 `landing/tools/`）是 CPU 端量化+打包工具，与 GPU arch 零依赖；真正锁 sm_89 的是 `engine-main` 的 CUDA 内核。
2. **CUDA 13.1 不挡 sm_75**：NVIDIA 官方只砍 CC<7.5（Maxwell/Pascal/Volta），sm_75 = CC7.5 正好是边界，未被砍。继续用 CUDA 13.x 旁装即可。
3. **BF16 MMA → FP16 MMA 可移植**：三元 GEMM 的 `mma.sync m16n8k16` 当前写死 BF16；Turing 有 FP16 张量核（同形状 m16n8k16、同吞吐），权重 -1/0/1 天然落在 fp16 表示范围 → 换型即可，非物理不可能。
4. **decode 是显存带宽瓶颈**：三元 2.125bpw 权重搬运主导 decode tok/s，bf16→fp16 几乎不损速度；prefill（compute-bound）在 Turing 比 Ada 慢属预期。
5. **GitHub 无先例**：目前唯一图灵 ninfer fork 是 `mr-september/ninfer-2080ti-22g`，但仅 W8A16，从未做过三元→Turing 内核移植——本工作属净新内核工程。

---

## 2. 里程碑总览

| 里程碑 | 目标 | 关键交付 | 预估 |
|---|---|---|---|
| **M0 基线与环境** | 可复现的上游 + T10 能编 sm_75 | fork 锁定 + CUDA13 旁装 + T10 编出 sm_75 SASS | 0.5d |
| **M1 构建系统放行** | CMake 不再拒绝 sm_75 | 改 2 处 CMake，configure 通过 | 0.5d |
| **M2 三元 GEMM 内核改写（核心）** | 三元主 MMA 跑在 sm_75 | `mma_fp16` 接线 + 3 个 mma 变体换型 | 3–5d |
| **M3 注意力/投影内核 + 调优** | 完整推理链路不 OOM | bf16 内核转 fp16（仅命中项）+ 共享内存 carveout | 3–7d |
| **M4 验证与基准** | 正确性/速度达标 | PPL≤6.8 + 六项验证链 PASS + 速度快照 | 2–3d |

**决策门**：M1 后确认能放行即继续；M2 后跑通最小档（主 mma + decode + PPL≤6.8）即投完整档，否则评估放弃 ninfer 改走 PrismML-llama.cpp fork。

---

## 3. 详细 TODO List

### M0 — 基线与环境 [0.5d]
- [x] **T0.1** 锁定上游 `paicat1/Bonsai-27B-NInfer` @ `engine-main` 基线 commit **`4c9a4f5`**（2026-09-27，`perf: port sched3 A3 token-grid scheduling (s8/wide prefill)`）；本地浅克隆参考副本 `bonsai-upstream-engine/` 已就位（见 `baseline.md`）。GitHub fork（`callmefree/Bonsai-27B-NInfer`）作为开发分支载体，于 M1 前完成。
- [ ] **T0.2** 本地配 CUDA 13.x 旁装（与上游同工具链）；fn36/T10 宿主是 Linux/Debian，需 `nvidia-toolkit` + 驱动支持 sm_75 编译。
- [ ] **T0.3** 写最小 sm_75 测试核（一段 `mma.sync .f16 m16n8k16`），`nvcc -arch=sm_75` 编译 + `cuobjdump --list-gpubins` 确认产物含 sm_75 SASS（防 JIT 伪装）。
- [ ] **验收**：能编出含 sm_75 SASS 的最小 CUDA 程序；上游 `engine-main` 在本地可 `cmake` configure（先不改，确认基线能配）。

### M1 — 构建系统放行 [0.5d]
- [ ] **T1.1** 根 `CMakeLists.txt` 架构白名单 `^(120a|89)$` → `^(120a|89|75)$`，错误文案同步加 75。
- [ ] **T1.2** `NINFER_SM89` 宏匹配由 `89|120a` 扩为 `89|120a|75`（让 75 走与 89 相同的"通用内核路径"）。
- [ ] **T1.3** 核查 `src/CMakeLists.txt`：确认 `nvfp4_w4a4` / `sm120_kv` 的 `else()` stub 对 75 自动生效（不改动，仅验证 stub 可链接不报未定义符号）。
- [ ] **验收**：`cmake -DCMAKE_CUDA_ARCHITECTURES=75` configure **不** FATAL_ERROR；`ninfer_ops` 目标成功生成（含三元/bf16/w8 通用内核）。

### M2 — 三元 GEMM 内核改写（核心）[3–5d]
- [ ] **T2.1** `src/ops/common/mma.cuh`：确认/补 FP16 `mma_fp16`（m16n8k16 PTX `mma.sync .f16` 封装）——**唯一必须新写/接线的张量核原语**。
- [ ] **T2.2** `src/ops/linear/ternary/ternary_rowsplit_mma.cuh`（主 MMA，硬编码 bf16）：
  - `__nv_bfloat16*` staging 数组 → `__half`；
  - `mma_bf16(...)` → `mma_fp16(...)`；
  - 解码末端 `__float22bfloat162_rn` → `__float22half2_rn`（fp32→fp16 送 MMA）；
  - **输出仍可存 bf16**（`__float2bfloat16_rn`，store 不需张量核；下游激活是 bf16，不连带改）；
  - fp32 累加器不变。
- [ ] **T2.3** `ternary_rowsplit_mma_wide_t.cuh`（宽瓦片 prefill）：同 T2.2 手法换型。
- [ ] **T2.4** `ternary_rowsplit_mma_small_t.cuh`（小 T 瓦片 verify/spec）：同 T2.2 换型。
- [ ] **T2.5** `ternary_rowsplit_mma_s8.cuh`（int8 MMA）：Turing 有 int8 TC，**大概率免改**，仅读码确认 dtype 一致。
- [ ] **T2.6** `ternary_rowsplit_gemm.cu`（实例化层，无 `__CUDA_ARCH__` 守卫）：读码确认免改。
- [ ] **验收**：能编出三元主 MMA 的 sm_75 SASS（`cuobjdump` 见 `m16n8k16.f16` 指令）；`ninfer_ops` 全量编译 75 通过。

### M3 — 注意力/投影内核 + 调优 [3–7d，视范围]
- [ ] **T3.1** 确认 Bonsai 实际命中的 op：`cuobjdump` 反查 / 运行日志 / 读 `src/ops/linear/linear.cpp` 的 dispatch——决定要转哪些 bf16 内核（这是工作量从 1 周涨 3 周的主要变量）。
- [ ] **T3.2** 对命中项按 T2.2 手法转 fp16：`bf16_gemm_mma.cuh`、`bf16_attn_input_*.cu`、`bf16_linear_add_*.cu`、`gdn_*` 等（仅命中者）。
- [ ] **T3.3** 共享内存 carveout：Turing TU102 每 SM 共享内存上限 **64KB**（sm_89/86 是 48KB）。核对 `wide_t` prefill 瓦片的静态共享内存不超 64KB，必要时调 `TernaryMmaSchedule` tile/流水，或用 `cudaFuncSetAttribute` 设最大动态共享。
- [ ] **验收**：完整推理链路在 T10 上能分配显存、不 OOM；Bonsai-2-27B 三元制品（9.81GiB）成功加载。

### M4 — 验证与基准 [2–3d]
- [ ] **T4.1** 正确性：跑六项验证链（pack / assembly / row_order 等）全 PASS；**PPL ≤ 6.8**（paicat1 基线 6.1129）。
- [ ] **T4.2** 架构产物：`cuobjdump --list-gpubins` 确认同时含 sm_75 / 89 / 120a SASS（120a/89 照常，证明未回归）。
- [ ] **T4.3** 速度基准：T10 decode tok/s（受 ~403–448 GB/s 带宽限），对比 Ada 基线（4080S decode ~226 t/s、5080 ~396 t/s）记录落差。
- [ ] **T4.4** 显存快照：16GB 占用 + KV Cache 余量（预期余 ~6–9GB，可开 32K+ 上下文）。

---

## 4. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| bf16 注意力/投影内核命中范围大 | 工作量↑至 3 周 | T3.1 先确认命中再扩大，最小档(M2)不受此影响 |
| `wide_t` 共享内存超 Turing 64KB | prefill 链接/运行失败 | T3.3 carveout / 调 tile / 动态共享 |
| FP16 MMA 数值 ≠ BF16 | PPL 上升 | magic-bias 已用 fp16，理论对等；T4.1 PPL 验证兜底 |
| 上游改版漂移 | 合并冲突 | T0.1 pin 基线 SHA，fork 内单独分支开发 |
| CUDA 13 旁装与宿主驱动不匹配 | 编译/运行失败 | T0.2 提前在 T10 宿主验证工具链 |

---

## 5. 工作量汇总

| 范围 | 内容 | 估计 |
|---|---|---|
| **最小可用档（推荐先做）** | M0–M2 + T4.1/4.2 最小验证；prefill 先用 s8/慢路径 | **~1 周** |
| **完整档** | M0–M4 全量（4 个 mma 变体 + bf16 注意力/投影 + 共享内存调优 + 全程验证） | **~2–3 周** |

---

## 6. 决策门（Go / No-Go）

- **门1（M1 后）**：CMake 能放行 sm_75 → 继续；否则排查工具链。
- **门2（M2 后，最小档）**：主 mma 跑通 + PPL≤6.8 + decode 不塌 → 投入完整档；否则评估放弃 ninfer，改走 **PrismML-llama.cpp fork + Bonsai-2 三元 GGUF**（已能跑，5.95GB 轻松进 16GB，几十 t/s）。

---

## 7. 仓库组织建议

```
ninfer-turing-sm75-port/
├─ README.md                      # 门面：项目简介 + 导航
├─ PLAN.md                        # 本文件：详细推进计划 / TODO list
├─ 技术细节-文件级改动.md          # 文件级改动清单与代码片段（fork 自原分析）
└─ patches/                       # (后续) 放 .patch / 分支差异
```

---

## 8. 战略提示

- **为什么值得做 vs 直接 PrismML**：W8A16 27B 装不下 T10；三元 9.81GiB 能进且 decode 显存带宽瓶颈 → 移植后 T10 跑真·27B 且 decode 不塌，ninfer 三元路径在吞吐/工程特性上仍更优。
- **为什么不必为"Bonsai 落地"而做**：PrismML fork + 三元 GGUF 已解决落地。走 ninfer-on-Turing 的动机是"想保留 ninfer 的速度/工程特性"，而非"让 Bonsai 落地"——门2 即此判断点。
