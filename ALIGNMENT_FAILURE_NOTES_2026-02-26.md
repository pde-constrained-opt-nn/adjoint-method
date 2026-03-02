# 对齐失败复盘（2026-02-26）

## 现象
1. 对齐后 `high-osci` 的 NN 结果明显劣化：
- 之前（较好）：`Adjoint+Fourier` 可到 `~1e-2` 量级（甚至更低）。
- 本轮（变差）：`Adjoint+Fourier` 最终约 `5e-2`，中途出现明显 spike。

2. VP smoke 初始无法运行，出现依赖缺失与版本冲突。

## 主要失败点
1. **训练配方漂移**（核心）
- 高频任务误用了 `fourier + exponential`。
- 先前较优配置是 `fourier_fixed + cosine(warmup)`。
- 导致高频任务稳定性下降、收敛到较差局部最优。

2. **环境耦合导致复现实验失真**
- 为 VP 安装依赖后，`jax/numpy/scipy` 被升级（与主实验环境分叉）。
- 同一机器上“终端环境”与“notebook kernel环境”不一致，容易误判“同环境”。

3. **VP 可选依赖未提前补齐**
- `matplotlib`、`vp_solver` 缺失导致 VP smoke 失败。
- 修复后 VP 能跑，但会进一步影响环境版本。

4. **结果口径变化与配方变化叠加**
- 对齐口径（`interior/exclude_t0`）本身会带来小幅数值差异；
- 但本轮大幅退化主要不是口径，而是配方/环境变化叠加。

## 已确认的结论
1. `problem4` 仍是 `high-osci` 别名，不是新 PDE。
2. 这次“效果变差”主要是实验设置变化，不是数学模型本身失效。

## 下次对齐前置约束（必须）
1. **双环境隔离**
- `pde-adjoint`：只跑 problem2/high-osci 主实验（固定版本）。
- `pde-vp`：只跑 VP smoke 与 Mark VP 相关流程。

2. **冻结高频基线配方**
- 高频默认：`arch=fourier_fixed`, `lr_schedule=cosine`, `warmup_ratio=0.10`, `seed=42`。
- 报告同时给 `final_loss` 与 `best_loss`（避免仅看末尾抖动点）。

3. **先验收再扩展**
- 先通过：`problem2` 不退化 + `high-osci` 恢复到历史量级。
- 再做 VP 接入与跨仓库汇总。

4. **Notebook 启动即打印环境指纹**
- 打印 `sys.executable`、`jax/flax/optax/numpy/scipy` 版本，写入每次结果头部。

## 建议恢复路线
1. 先回到对齐前代码与主环境，确认基线性能恢复。
2. 再以“最小改动”逐项对齐：
- 先问题别名 + 指标 schema；
- 再口径切换；
- 最后 VP 接入。
