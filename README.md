# NInfer 三元内核 → Tesla T10 (sm_75 / Turing) 移植

> 把 [`paicat1/Bonsai-27B-NInfer`](https://github.com/paicat1/Bonsai-27B-NInfer) 引擎分支 `engine-main` 的**三元 Bonsai-2-27B** 推理，从 sm_89 / 120a 移植到 **Tesla T10 (sm_75, 16GB)**。

## 目标

让单张 T10（16GB）跑 ninfer 三元 27B——保住 **2.125bpw 高密度**（三元制品 9.81GiB 装得进 16GB），且 decode 不受 W8A16 社区 fork（27B ≈ 16.3GiB 装不下 T10）的限制。

## 状态

🟡 **规划中** —— 详细分阶段推进计划见 [PLAN.md](./PLAN.md)

## 核心结论（为什么可行）

- 改的是**引擎内核**而非打包器；
- **CUDA 13.1 不挡 sm_75**（CC=7.5 边界未被 NVIDIA 砍）；
- 三元 GEMM 的 BF16 MMA → **FP16 MMA** 可移植（Turing 有 fp16 张量核 `m16n8k16`）；
- decode 是显存**带宽瓶颈**，换 fp16 几乎不损速度；
- GitHub 上**无三元→Turing 先例**（net-new kernel work，你是第一个）。

## 文件导航

| 文件 | 内容 |
|---|---|
| [PLAN.md](./PLAN.md) | 分阶段详细推进计划：里程碑 M0–M4、逐文件 TODO、验收标准、决策门、风险与工作量 |
| [技术细节-文件级改动.md](./技术细节-文件级改动.md) | 文件级改动清单与代码片段（fork 自前期技术分析） |

## 里程碑一览

| 里程碑 | 目标 | 预估 |
|---|---|---|
| M0 基线与环境 | fork 锁定 + CUDA13 旁装 + T10 能编 sm_75 | 0.5d |
| M1 构建系统放行 | CMake 不再拒绝 sm_75 | 0.5d |
| M2 三元 GEMM 内核改写（核心） | 主 MMA + 变体转 FP16 | 3–5d |
| M3 注意力/投影内核 + 调优 | bf16 内核转 fp16 + 共享内存 carveout | 3–7d |
| M4 验证与基准 | PPL≤6.8 + 六项验证链 + 速度快照 | 2–3d |

> 最小可用档（M0–M2 + 最小验证）≈ **1 周**；完整档 ≈ **2–3 周**。

## 上游

- `paicat1/Bonsai-27B-NInfer` @ `engine-main`（fork 自 `Ambolio/ninfer-4090-windows` v1.0.8）

## License

待定（跟随上游；ninfer 引擎为作者专有，本仓库以 patch / fork 分支形式存在，不重新分发任何二进制制品）。
