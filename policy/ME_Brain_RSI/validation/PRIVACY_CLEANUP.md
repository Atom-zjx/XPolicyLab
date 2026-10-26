> 本页为路径与隐私清理阶段记录；后续资源目录统一的当前验证见 [资源目录统一记录](RESOURCE_LAYOUT.md)。

# 敏感信息与相对路径清理记录

清理对象：`ME_Brain_RSI/`。

## 已完成

- 清除了策略目录内已识别的人员目录名、内部服务域名、内部模型默认值、公司请求头默认值和凭据文件位置。
- 模型、机器人配置、资源挂载、解释器与文档示例采用相对引用；根配置中的资源路径统一以策略根目录为基准，`includes` 以配置文件所在目录为基准。
- 本机权重保持原位置，通过策略目录外的中性资源绑定访问；私有服务设置与原始备份保存在目录外，并由本地忽略规则排除提交。目录内 `runtime_config.local.json` 只保留中性的相对引用。
- 清理了 7 份资源来源清单并重算内容哈希，保留数值数据和原始模型文件校验值；194 个资源源码硬链接已核实与公共源码保持同步。
- 旧运行产物已移出策略目录，历史验证日志已脱敏，编译缓存已清除。没有读取真实凭据文件，没有发出在线模型请求。
- 各任务感知、几何、运动与任务策略分别保留；本次变动为路径解析、配置注入、元数据与日志清理。

## 最新验证

| 检查 | 结果 |
| --- | --- |
| 29 项回归测试 | 全部通过 |
| 35 项任务、206 个辅助函数探针 | 与原实现输出一致 |
| 83 个运行资源引用 | 均解析到原来的文件 |
| 普通／编码图像调试闭环 | 均通过；从策略目录之外启动，使用相对配置和解释器路径 |
| 913 个 Python 文件 | 语法编译通过 |
| 提交空白检查、资源哈希、链接完整性 | 通过 |
| 已知人员／内部服务标记扫描 | 未发现残留 |
| Python／JSON 中真实资源绝对路径字面量 | 未发现残留 |
| 符号链接目标 | 全部为相对路径 |
| 26 个二进制及压缩数组资源 | 未检出已知私人路径标记 |

完整证据见 `privacy_path_audit.json`、`unit_tests_after_privacy.log`、`behavior_parity_after_privacy.json`、`relative_resource_bindings.json`、`debug_relative_plain.log`、`debug_relative_encoded.log`。

## 使用方式

```bash
cd XPolicyLab/policy/ME_Brain_RSI
export ME_BRAIN_RUNTIME_CONFIG=./runtime_config.local.json
PYTHONDONTWRITEBYTECODE=1 python3 preflight.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

资源路径与私有服务设置可通过配置文件或环境变量替换。公开发布时以 `runtime_config.example.json` 为模板准备自己的资源配置，目录外的机器私有配置不属于发布内容。

## 保留的语义与验证范围

CAD/USD 的 prim、mesh、material 标识保留原值，它们不是文件系统路径。系统设备路径、解释器 shebang、HTTP 路由也保留各自语义。4 份相关几何元数据已验证原字节保持不变。

扫描涵盖当前目录的文本文件、资源链接以及二进制资源中的已知私人标记；不会读取目录外权重内容或真实凭据文件。公开上游的许可证与作者归属未删除。

本次未执行全任务仿真或训练。运行资源预检仍报告方程任务环境缺少 `pinocchio`、`ftfy`；原先的图像接口提交审查限制也仍然保留在 `REPORT.md` 中。历史阶段证据已标注，不能代替本次或完整仿真验收。
