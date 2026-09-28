# M0–M2 工作审查报告（代码级复核）

> 审查日期：2026-09-28 ｜ 审查人：WorkBuddy ｜ 方法：实地读取 fork 实际改动 + 上游引擎源码，不采信摘要
> 上游基线：`paicat1/Bonsai-27B-NInfer` @ `engine-main` `4c9a4f5`

---

## 0. 总判定

| 项 | 结论 | 说明 |
|---|---|---|
| **M0（基线锁定）** | ✅ 正确 | fork 已建、基线 SHA 一致、本地/远程提交对齐 |
| **M1（构建放行）** | ✅ 正确 | CMake 三处改动到位且精确，sm_75 能放行 |
| **M2（内核分析）** | ⚠️ 方向对，但**严重低估** | 漏掉 `cp.async` 是 Ampere 专属的真实障碍；`ldmatrix` 反而可用；共享内存担忧不成立；估算过低 |

---

## 1. M0 审查 ✅

- **fork**：`callmefree/Bonsai-27B-NInfer` 已建；`sm75-port` 分支 `fe84565` 正确坐在基线 `4c9a4f5` 之上（gh API 验证）。
- **基线 pin**：`baseline.md` 记录 `4c9a4f5`（2026-09-27，`perf: port sched3 A3 token-grid scheduling (s8/wide prefill)`）—— 该 commit 正在动 `s8`/`wide_t` prefill 内核，pin 它防漂移是正确的。
- **本地副本**：`bonsai-upstream-engine/` 浅克隆就位，与远程一致。
- 无问题。

## 2. M1 审查 ✅

fork 中 `CMakeLists.txt` 实测三处改动均精确落地：

| 行 | 改动 | 验证 |
|---|---|---|
| L11 | `"^(120a\|89)$"` → `"^(120a\|89\|75)$"` | ✅ `75` 能正确匹配放行 |
| L13 | 错误文案加 `75` | ✅ 同步 |
| L38 | `NINFER_SM89` 宏匹配 `89\|120a` → `89\|120a\|75` | ✅ 让 75 走与 89 相同的通用内核路径 |

- `patches/M1-build-system-sm75.patch` 与 fork 实际 diff **逐字节一致**（已比对）。
- **T1.3 验证成立**：`src/CMakeLists.txt` 的 `nvfp4_w4a4`/`sm120_kv`/`w8_sm120` 均走 `if(MATCHES "^120")…else()→*_stubs.cpp`，sm_75 不匹配 `^120` 自动落 stub，**无需改动**。
- **遗留小项**：fork 的 `main` 分支是 `2a20a7e`（不是 engine-main），`engine-main` 分支未推到 fork。不影响 `sm75-port` 工作分支，但建议后续把 `engine-main` 也推到 fork 作为干净 base。

## 3. M2 审查 ⚠️（核心发现）

### 3.1 已核实正确的部分

- **T2.1（`mma_f16` 已存在）✅**：`src/ops/common/mma.cuh` **L42-49** 确有 `mma_f16`（`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`），另含 `mma_f16_f16acc`（L54）、`mma_s8`（L62）。**Turing 原生 fp16 张量核，直接复用，零新内核要写。** PLAN 原"唯一必须新写的原语"被推翻——这是净好消息。
- **三元解码本就是 fp16 算术 ✅**：`ternary_rowsplit_mma.cuh` L90-91 的 `kMagic=0x6400`/`kBias=0x6401` 是 **fp16 的 1024.0/1025.0**；L98-102 用 `__half2` 算术得出 fp16 权重×scale。仅在 **L104-105** 把 fp16 结果转回 bf16 存 A tile。所以改 fp16 MMA 非常自然。
- **输出保持 bf16 正确 ✅**：L380-381 `out` 是 `__nv_bfloat16*`、`__float2bfloat16_rn` 存盘，store 不需张量核，保留兼容下游。

### 3.2 🔴 重大遗漏：`cp.async` 是 Ampere 专属，Turing 根本没有

- 内核大量使用 `cp_async`/`cp_commit`/`cp_wait`（主内核 L237/240/262/267/322/327/365）。
- `src/ops/common/memory.cuh` 的 `cp_async`（L36-48）、`cp_commit`（L67）、`cp_wait`（L70-73）**无条件发射 `cp.async.*` PTX，没有任何 `__CUDA_ARCH__` 守卫**。
- 官方确认：`cp.async` / `LDGSTS` 是 **CUDA 11.0 / sm_80（Ampere）引入**。**Turing sm_75 不支持 cp.async** → 对 sm_75 编译会在 PTX→SASS 阶段直接失败。
- **PLAN 完全漏算这一项。** M2 不是"换 MMA + 换指针类型"那么轻。

**但这是可控的、高杠杆的修复**：`memory.cuh` 是**唯一咽喉点**。给 `cp_async`/`cp_commit`/`cp_wait` 加 `__CUDA_ARCH__ < 800` 的同步回退（register 中转的 `ld.global` + `st.shared`；`cp_commit`→no-op；`cp_wait<N>`→`__syncthreads()`），即可让**全树 75+ 个文件**的搬运管线一次性 Turing 兼容，无需逐文件改。

### 3.3 🟡 激活 bf16→fp16 转换必须并进同步回退

- 激活 `x` 是 `__nv_bfloat16*`（主内核 L163）。当前 `Bs` 是 bf16，`cp_async` 原样拷贝字节 → ldmatrix → `mma_bf16`。
- 目标 `mma_f16` 要 fp16 操作数。若把 `Bs` 直接改成 `__half*` 而拷贝仍走"字节原样"，bf16 位模式会被错误解释成 fp16 → 数值错。
- **正确做法**：让 `Bs` 为 `__half*`，在同步回退拷贝里对激活逐元素 `__bfloat162half2` 转 fp16 再写共享内存（**选 B：转换发生在 staging 一次，比在 ldmatrix 后每 k-step 转更省**）。权重 A 侧同理（`As` 改 `__half*`，去掉 L104-105 的 fp16→bf16 回退）。

### 3.4 ✅ 好消息：`ldmatrix` 在 sm_75 可用

- 官方资料确认 `ldmatrix` 是 **Turing(sm_75) 引入**，sm_75 及以下需所有线程提供有效地址（本内核满足）。
- 主内核 L342/349 的 `ldmatrix_x4`/`ldmatrix_x2` 在 Turing 直接可用，**不必替换**为手动 `ld.shared`。这是 M2 工作量的重要对冲。

### 3.5 ⚪ 共享内存担忧不成立（PLAN T3.3 需降级）

- 主 prefill 内核 `static_assert(kSharedBytes <= 48*1024)`（L145）—— 这是引擎**静态预算**，已由作者控制在 48KB 内。
- Turing TU102 每 SM 共享内存上限 **64KB**（≥48KB），故该断言在 sm_75 上**自动通过，无溢出风险**。
- PLAN T3.3"wide_t 超 Turing 64KB" **对主 prefill 内核不成立**（其 ≤48KB<64KB）。`wide_t` 变体若也 ≤48KB 则同样无虞——需单独确认，但风险远低于原描述。

### 3.6 M2 子任务状态精修

| 子项 | 原 PLAN | 复核后 |
|---|---|---|
| T2.1 | mma_f16 需新写 | ✅ 已存在，完成 |
| T2.0（**新增**） | — | `memory.cuh` 加 sm_75 同步拷贝回退（咽喉点，覆盖全树） |
| T2.2 | 换 MMA + 换指针 | + 必须并进 bf16→fp16 转换（激活侧选 B） |
| T2.3/2.4 | wide_t/small_t 同手法 | 继承 T2.0+T2.2，同样受 cp.async 牵连（grep 确认二者亦用 cp_async/ldmatrix） |
| T2.5 | s8 免改 | s8 用 int8 MMA（Turing 有），但**若 s8 内核也用 cp_async 搬运，仍需 T2.0 回退**——"免改"仅指 MMA 指令，搬运不豁免 |

---

## 4. 工作量重新估算（修订）

| 范围 | 原估计 | 复核后 | 差异原因 |
|---|---|---|---|
| M0（基线+环境） | 0.5d | 0.5d | 不变（远程 T10 机器待定，T0.2/0.3 未做） |
| M1（构建放行） | 0.5d | 0.5d | 已做，精确 |
| **M2（核心内核）** | **3–5d** | **~1–2 周** | 加 T2.0（memory.cuh 回退）+ bf16→fp16 转换整合；非仅换 MMA |
| M3（attention/linear + 调优） | 3–7d | ~2–4 周 | 全树 bf16 内核同样受 cp.async + bf16→fp16 牵连，命中项更多 |
| **最小可用档（M0–M2 min + T4.1/4.2）** | **~1 周** | **~2 周** | cp.async 回退 + 激活转换拉高下限 |
| 完整档（M0–M4） | 2–3 周 | 4–6 周 | M2/M3 双双上调 |

> 注：T2.0 是一次性咽喉点修复，摊薄到所有内核；M3 的"命中多少 bf16 内核"仍是最大变量（T3.1 须先反查 Bonsai 实际 dispatch 哪些 op）。

---

## 5. 给远程编译验证（T0.2/T0.3）的修正要求

原 PLAN 的 T0.3 只要求"写最小 sm_75 测试核跑 `mma.sync .f16 m16n8k16`"。**复核后必须升级**：

1. T0.3 测试核要覆盖**完整链路**：`memory.cuh` 的同步回退 `cp_async`/`cp_commit`/`cp_wait` + `ldmatrix` + `mma_f16` 三段在 sm_75 上能编出 SASS。
2. 必须验证 `cuobjdump --list-gpubins` 产物**只含 sm_75 SASS、无 cp.async**（确认回退生效、无 PTX JIT 伪装）。
3. 在 T10 宿主上真正 `cmake -DCMAKE_CUDA_ARCHITECTURES=75` 跑通 `ninfer_ops` 全量编译（M1 验收的远程部分），这是 T0.2/T0.3 的核心交付。

---

## 6. 结论

- **M0、M1 可确认无误**，已 push，可作为后续开发坚实基础。
- **M2 分析方向正确但漏了 `cp.async` 这个真实障碍**；好在它是 `memory.cuh` 单点咽喉，修复杠杆极高，且 `ldmatrix`/`mma_f16` 已可用，整体可行、未被推翻。
- 需要**立刻补一个 T2.0 子任务**（memory.cuh sm_75 同步回退）并把估算上调；这不是"放弃 ninfer"的信号，而是把工作量说准。
- 远程编译验证前，务必先拿到 T10 宿主（IP/系统/驱动/CUDA toolkit），且验证脚本要覆盖完整 staging+MMA 链路。
