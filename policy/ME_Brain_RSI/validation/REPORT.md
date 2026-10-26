> 当前资源目录结构与迁移验证见 [资源目录统一记录](RESOURCE_LAYOUT.md)。本页保留前期结构重构阶段的证据与限制。

# 重构与验证记录

策略目录：`XPolicyLab/policy/ME_Brain_RSI`。报告不记录部署机器的挂载根目录和人员目录。

结构重构已落盘。原始版本目录未修改。后续隐私清理仅调整当前目录的路径绑定、私有配置和身份元数据。本次没有创建提交、推送代码或发起拉取请求。

## 实现

- `model.py` 通过 `ME_Brain_Adaptor` 统一注册、初始化、观测、动作、重置与清理。
- 全部 54 个任务采用同一个注册表，包括 39 个脚本策略注册项、1 个脚本任务别名和 14 个 VLA 任务。
- 公共能力按感知、几何、运动、规划、观测和控制组织；原来按人员、交付包和历史版本选择整份代码库的入口已经移除。
- 1,402 次模块引用合并为 607 份源码模块。不同感知方法分别保留；字节相同的源码共享存储，各任务仍具有独立的模块、全局变量和资源布局。
- 任务策略与专用部署循环保留原实现。生成器策略仍按其原来的部署位置执行，原生动作协议与动作时序保持原样。
- 结构重构阶段保留单份 Focus-VLWA 源码，当时检查的 60 个源码及配置文件与原目录逐字节一致。后续隐私清理只将文档中的绝对路径示例改为相对路径。VLA 检查点统一通过公共解析器定位，并保留绝对路径检查点的原有优先级。
- 结构重构阶段提取的 1,384 次原算法模块引用保留计算逻辑。13 个文件仅删除末尾多余空行，修改前后 AST 完全一致，详见 `formatting_only.json`。后续路径治理涉及资源定位与外部私有配置绑定，其校验值和验证结果由最新检查记录给出。不同任务的感知与控制算法分别保留。

## 结构重构阶段已通过的验证

以下记录是结构重构阶段取得的结果。路径与隐私清理后的最终结果以最新验证记录为准。

| 检查 | 结果 |
| --- | --- |
| 54 项任务注册、别名及原动作协议 | 通过 |
| 910 个 Python 文件语法编译 | 通过 |
| 633 个框架与能力文件的提交空白检查 | 通过 |
| 10 项框架回归测试 | 全部通过 |
| 35 项任务的 206 个感知、几何及运动辅助函数探针 | 迁移前后输出全部一致 |
| 普通 RGB 调试闭环 | 通过 |
| 编码图像调试闭环 | 通过 |
| 模型及部署入口导入 | 38 项通过；方程任务因现有环境缺少 `pinocchio` 未通过 |

调试闭环均使用 `press_by_number`，每种输入完成 10 个调试回合、每回合 20 步。该结果验证通信、输入解码与动作接口，不代表机器人任务成功率。

下面是相同接口的可移植复现示例。历史验证使用当时可用的运行环境，机器专属解释器路径已从报告中移除。

```bash
cd XPolicyLab/policy/ME_Brain_RSI
export ME_BRAIN_RUNTIME_CONFIG=./runtime_config.local.json
export ME_BRAIN_POLICY_PYTHON=./strategy_env/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
export ME_BRAIN_EVAL_PYTHON="$ME_BRAIN_POLICY_PYTHON"
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo press_by_number none arx_x5 auto 0 0 0 uv base
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo press_by_number none arx_x5 auto 0 0 0 uv base
python3 preflight.py
python3 -m unittest discover -s tests -v
```

完整命令输出分别保存在 `debug_plain.log`、`debug_encoded.log`、`unit_tests.log`。结构、行为、源码和运行环境证据分别保存在 `preflight.json`、`behavior_parity.json`、`source_parity.json`、`static_checks.json`、`runtime_preflight.json`。

## 提交要求与剩余限制

适配器文件、`ModelTemplate` 接口、标准脚本参数、`deploy.yml` 标准键、公共检查点解析器及使用说明已对齐仓库中的 `CONTRIBUTING.md` 和 `demo_policy`。根仓库说明文档未改动。

当前不能声明完成全部提交验收：

1. 尚未重跑全任务仿真，无法用本次结果保证完整成功率与运行时性能不变。辅助函数一致性与源码一致性证明的范围如上。
2. 当前检查所用环境缺少方程任务所需的 `pinocchio`、`ftfy`；原生 CUDA、SAM3、DINO 及所有权重的实际加载兼容性没有得到完整验证。资源预检只检查文件或模块存在，不替代运行测试。
3. 为遵循保留原感知能力的要求，原策略内部的独立 PNG 服务协议、BGR 工具接口及诊断图像输出保持原样。静态扫描记录了 27 处相关历史调用，其中含直接 `cv2.imdecode`；这些调用尚未满足仓库图像规则的完整审查。适配器 `model.py` 自身不解码、不交换颜色通道。完整清单见 `static_checks.json`，正式提交前仍需处理或取得维护者认可。
4. 训练与数据转换入口保留，但本次没有重新训练或验证训练结果。正式排行榜发布仍需提供可下载的检查点和正式仿真结果。

当前交付记录了结构重构及其上述验证，尚未完成排行榜验收。目录内的机器资源配置仅保留中性相对路径，机器专属资源映射和可选在线服务的私有配置放在策略目录之外。`runtime_config.local.json` 仍不应公开提交。历史日志、异常堆栈与运行产物需脱敏后归档，不能仅依赖 `.gitignore`。

## 审查规则来源

仓库的 [xpolicylab-adapter-check/SKILL.md](https://github.com/XPolicyLab/XPolicyLab/blob/feat/offline_eval/.agents/skills/xpolicylab-adapter-check/SKILL.md) 将以下情况列为阻塞项：

> cv2.imdecode anywhere outside `utils/process_data.py`

本次已读取服务器上同一路径的工作流文件。由于原策略仍含这类历史调用，报告没有将其标为通过；这是明确的检查要求。为了保留原感知与控制能力，本次没有在缺少完整仿真证据时替换这些内部图像协议。

## 本次隐私与路径清理

本次清理及最新验证见 [敏感信息与相对路径清理记录](PRIVACY_CLEANUP.md)。本文件之前列出的测试数字属于结构重构阶段历史结果。
