# NInfer 三元内核 → Tesla T10 (sm_75 / Turing) 移植推进计划

> 状态：**M1 已完成 / M2 分析中（发现 cp.async 障碍，见 AUDIT-M0-M2.md）** ｜ 负责人：callmefree ｜ 创建：2026-09-28 ｜ 最近更新：2026-09-28
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
| **M2 三元 GEMM 内核改写（核心）** | 三元主 MMA 跑在 sm_75 | T2.0 memory.cuh 同步回退(咽喉点) + `mma_f16` 接线 + 3 个 mma 变体换型 + 激活 bf16→fp16 | 1–2 周 |
| **M3 注意力/投影内核 + 调优** | 完整推理链路不 OOM | bf16 内核转 fp16（仅命中项，同样受 cp.async 牵连）+ 共享内存确认 | 2–4 周 |
| **M4 验证与基准** | 正确性/速度达标 | PPL≤6.8 + 六项验证链 PASS + 速度快照 | 2–3d |

**决策门**：M1 后确认能放行即继续；M2 后跑通最小档（主 mma + decode + PPL≤6.8）即投完整档，否则评估放弃 ninfer 改走 PrismML-llama.cpp fork。

---

## 3. 详细 TODO List

### M0 — 基线与环境 [0.5d]
- [x] **T0.1** 锁定上游 `paicat1/Bonsai-27B-NInfer` @ `engine-main` 基线 commit **`4c9a4f5`**（2026-09-27，`perf: port sched3 A3 token-grid scheduling (s8/wide prefill)`）；本地浅克隆参考副本 `bonsai-upstream-engine/` 已就位（见 `baseline.md`）。GitHub fork（`callmefree/Bonsai-27B-NInfer`）作为开发分支载体，于 M1 前完成。
- [ ] **T0.2** 本地配 CUDA 13.x 旁装（与上游同工具链）；fn36/T10 宿主是 Linux/Debian，需 `nvidia-toolkit` + 驱动支持 sm_75 编译。
- [ ] **T0.3** 写最小 sm_75 测试核，**覆盖完整链路**：`memory.cuh` 同步回退的 `cp_async`/`cp_commit`/`cp_wait` + `ldmatrix` + `mma_f16`（而非仅裸 MMA）；`nvcc -arch=sm_75` 编译 + `cuobjdump --list-gpubins` 确认产物**只含 sm_75 SASS、无 cp.async**（验证回退生效、无 PTX JIT 伪装）。
- [ ] **验收**：能编出含 sm_75 SASS 的最小 CUDA 程序；上游 `engine-main` 在本地可 `cmake` configure（先不改，确认基线能配）。

### M1 — 构建系统放行 [0.5d] ✅ 已完成
- [x] **T1.1** 根 `CMakeLists.txt` 架构白名单 `^(120a|89)$` → `^(120a|89|75)$`，错误文案同步加 75。→ **已改**（L11 正则 + L13 文案，见 `patches/M1-build-system-sm75.patch`）。
- [x] **T1.2** `NINFER_SM89` 宏匹配由 `89|120a` 扩为 `89|120a|75`（让 75 走与 89 相同的"通用内核路径"）。→ **已改**（L38，见 patch）。
- [x] **T1.3** 核查 `src/CMakeLists.txt`：确认 `nvfp4_w4a4` / `sm120_kv` 均 `if(MATCHES "^120")…else()→*_stubs.cpp`（L78-86 / L93-101），sm_75 不匹配 `^120` 自动落 stub，**无需改动**；`w8_sm120` 段（L109-116）对所有架构编译通用 w8 内核，也不限 120。
- [ ] **验收**：`cmake -DCMAKE_CUDA_ARCHITECTURES=75` configure **不** FATAL_ERROR；`ninfer_ops` 目标成功生成（含三元/bf16/w8 通用内核）。→ *本地无 CUDA toolkit，configure 验证推迟至 T0.2/T0.3 远程 T10 机器。*

### M2 — 三元 GEMM 内核改写（核心）[复核后 ~1–2 周]
> ⚠️ **复核重大发现（见 AUDIT-M0-M2.md）**：`cp.async` 是 Ampere(sm_80)+ 专属，**Turing sm_75 不支持**；`memory.cuh` 的 `cp_async`/`cp_commit`/`cp_wait` 无 `__CUDA_ARCH__` 守卫，对 sm_75 编译必挂。但 `ldmatrix`（Turing 引入）与 `mma_f16` 均可用。修复杠杆极高——`memory.cuh` 是**唯一咽喉点**，一处回退覆盖全树 75+ 文件。
- [x] **T2.1** `src/ops/common/mma.cuh`：已确认 **`mma_f16`（`m16n8k16 .f16` PTX 封装，L42-49）已存在**，另含 `mma_f16_f16acc`（f16 累加变体）。Turing(sm_75) 原生 fp16 TC，直接复用即可 —— **无需新写任何张量核原语**。
- [ ] **T2.0（新增·咽喉点）** `src/ops/common/memory.cuh`：给 `cp_async`/`cp_commit`/`cp_wait` 加 `__CUDA_ARCH__ < 800` 同步回退——`cp_async` 改 register 中转的 `ld.global`+`st.shared`（16B 块，激活侧逐元素 `__bfloat162half2` 转 fp16 落 fp16 Bs）；`cp_commit`→no-op；`cp_wait<N>`→`__syncthreads()`。此一处修复令全树搬运管线一次性 sm_75 兼容。**同时验证 `#include <cuda_pipeline.h>` 在 sm_75 下仅 `pipe_*` 受影响（本内核未用），不挡编译。**
- [ ] **T2.2** `src/ops/linear/ternary/ternary_rowsplit_mma.cuh`（主 MMA）— 解码 `ternary_mma_decode_byte` 已用 fp16 magic(`0x6400`=1024.0)，去掉末端 L104-105 的 fp16→bf16 回退（lo/hi_bits 直接取 `ha`/`hc` 的 fp16 位）；`As`/`Bs` staging 指针类型 `__nv_bfloat16*` → `__half*`；**激活转换并入 T2.0 的同步回退**（Bs 走 fp16，避免 bf16 位模式被误读为 fp16）；MMA 调用 `mma_bf16` → `mma_f16`；输出 `out` 保持 `__float2bfloat16_rn`(bf16，兼容下游)；共享内存字节数不变（fp16/bf16 均 2B）。
- [ ] **T2.3** `ternary_rowsplit_mma_wide_t.cuh`（宽瓦片 prefill）：同 T2.2 手法换型。
- [ ] **T2.4** `ternary_rowsplit_mma_small_t.cuh`（小 T 瓦片 verify/spec）：同 T2.2 换型。
- [ ] **T2.5** `ternary_rowsplit_mma_s8.cuh`（int8 MMA）：Turing 有 int8 TC，**MMA 指令本身免改**；但若该内核也用 `cp_async` 搬运（grep 确认 ternary 目录均用 cp_async），仍需 T2.0 回退——"免改"仅指 MMA，搬运不豁免。读码确认。
- [ ] **T2.6** `ternary_rowsplit_gemm.cu`（实例化层，无 `__CUDA_ARCH__` 守卫）：读码确认免改。
- [ ] **验收**：能编出三元主 MMA 的 sm_75 SASS（`cuobjdump` 见 `m16n8k16.f16` 指令、**无 cp.async**）；`ninfer_ops` 全量编译 75 通过。

### M3 — 注意力/投影内核 + 调优 [3–7d，视范围]
- [ ] **T3.1** 确认 Bonsai 实际命中的 op：`cuobjdump` 反查 / 运行日志 / 读 `src/ops/linear/linear.cpp` 的 dispatch——决定要转哪些 bf16 内核（这是工作量从 1 周涨 3 周的主要变量）。
- [ ] **T3.2** 对命中项按 T2.2 手法转 fp16：`bf16_gemm_mma.cuh`、`bf16_attn_input_*.cu`、`bf16_linear_add_*.cu`、`gdn_*` 等（仅命中者）。
- [ ] **T3.3** 共享内存 carveout：**复核降级**——主 prefill 内核 `static_assert(kSharedBytes <= 48*1024)`（L145）已由作者控制在 48KB；Turing TU102 每 SM 上限 **64KB**（≥48KB），断言自动通过，**主内核无溢出风险**。仅需在改 `wide_t` 变体时确认其 `kSharedBytes` 同样 ≤48KB（grep 见 wide_t 亦参与 staging，按同断言约束）。`cudaFuncSetAttribute` 在 CUDA graph capture 内非法，保持静态预算即可。
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
| **`cp.async` 是 Ampere 专属，sm_75 无** | 全树搬运管线编译失败（原漏算） | **T2.0 在 `memory.cuh` 单点加 sm_75 同步回退**，一处覆盖 75+ 文件 |
| bf16 注意力/投影内核命中范围大 | 工作量↑ | T3.1 先确认命中再扩大，最小档(M2)不受此影响 |
| `wide_t` 共享内存超 Turing 64KB | ~~原以为风险~~ **复核降级**：主内核静态 ≤48KB < Turing 64KB，断言自动过 | T3.3 仅确认 wide_t 变体同样 ≤48KB |
| FP16 MMA 数值 ≠ BF16 | PPL 上升 | magic-bias 已用 fp16，理论对等；T4.1 PPL 验证兜底 |
| `#include <cuda_pipeline.h>` 在 sm_75 | 未用 `pipe_*` 但含头，须确认不挡编译 | T2.0 内验证；必要时条件包含 |
| 上游改版漂移 | 合并冲突 | T0.1 pin 基线 SHA，fork 内单独分支开发 |
| CUDA 13 旁装与宿主驱动不匹配 | 编译/运行失败 | T0.2 提前在 T10 宿主验证工具链 |

---

## 5. 工作量汇总

| 范围 | 内容 | 估计 |
|---|---|---|
| **最小可用档（推荐先做）** | M0–M2(min) + T4.1/4.2 最小验证；prefill 先用 s8/慢路径 | **~2 周** |
| **完整档** | M0–M4 全量（4 个 mma 变体 + bf16 注意力/投影 + 共享内存确认 + 全程验证） | **~4–6 周** |

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
├─ AUDIT-M0-M2.md                 # 代码级复核报告（cp.async 障碍发现 + 估算修订）
├─ baseline.md                    # 上游基线 SHA pin 记录
├─ 技术细节-文件级改动.md          # 文件级改动清单与代码片段（fork 自原分析）
└─ patches/                       # 放 .patch / 分支差异（M1 已入）
```

---

## 8. 战略提示

- **为什么值得做 vs 直接 PrismML**：W8A16 27B 装不下 T10；三元 9.81GiB 能进且 decode 显存带宽瓶颈 → 移植后 T10 跑真·27B 且 decode 不塌，ninfer 三元路径在吞吐/工程特性上仍更优。
- **为什么不必为"Bonsai 落地"而做**：PrismML fork + 三元 GGUF 已解决落地。走 ninfer-on-Turing 的动机是"想保留 ninfer 的速度/工程特性"，而非"让 Bonsai 落地"——门2 即此判断点。
