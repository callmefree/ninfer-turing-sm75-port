# 上游基线锁定（baseline pin）

- 上游仓库：`paicat1/Bonsai-27B-NInfer`
- 移植基线分支：`engine-main`
- **锁定 commit SHA**：`4c9a4f53e4ac0f14be47f5c21743639fc826f07c`
- 提交时间：2026-09-27T14:59:13Z
- 提交信息：`perf: port sched3 A3 token-grid scheduling (s8/wide prefill)`
- 锁定日期：2026-09-28

## 为何 pin

该 commit 涉及 `s8` / `wide_t` prefill 调度内核（正是 M2 移植目标 `ternary_rowsplit_mma_s8.cuh` / `ternary_rowsplit_mma_wide_t.cuh`），上游仍在活跃改动这些文件。pin 此 SHA 可防止后续漂移导致移植补丁冲突。

## 本地参考副本

- 工作区浅克隆：`bonsai-upstream-engine/`（仅本地参考，不推送上游）
- GitHub fork（开发分支载体）：`callmefree/Bonsai-27B-NInfer`（于 M1 前 fork 完成，承载 `sm75-port` 分支）

## 同步方法

后续如需拉取上游更新：

```bash
git -C bonsai-upstream-engine fetch --depth 1 origin engine-main
# 对比 diff 后，人工 cherry-pick 到 sm75-port 分支
```
