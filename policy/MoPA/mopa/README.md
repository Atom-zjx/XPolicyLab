# MoPA

[MoPA: Coordinated Mobile Manipulation via Subsystem-Specific Perception Alignment](https://mopa-policy.github.io/)
通过子系统专属的视觉 query 与动作专家建模感知和动作。本实现提供机械臂策略：
保留一套 **8 个 manipulation query**，关闭 base 分支，生成 **4 步关节动作**。
输入为 RGB 图像、语言指令和机械臂/夹爪状态，无需底盘状态或场景 context。

## 模型

视觉语言骨干为 Qwen3-VL-4B-Instruct-Action，保留 M-RoPE。动作头使用
16 层 DiT-B，注意力宽度为 768、头数为 12，状态编码和动作解码 MLP
隐藏维度为 1024。训练采用均匀 flow matching，每个样本重复采样 8 次噪声；
推理采用 4 步 Euler 积分。默认配置见
[configs/model.json](configs/model.json)。

```text
mopa/
├── pyproject.toml
├── requirements.txt
├── README.md
├── models/                     # Qwen、query 策略和动作头
├── configs/model.json
├── common.py                   # 关节布局、RGB 缩放和归一化
├── data/                       # 数据准备和训练数据集
├── training/                   # 训练入口、优化器和 checkpoint
├── runtime.py                  # 独立 checkpoint 推理
└── integrations/               # 可选的宿主协议适配
```

## 安装

在本目录创建 Python 3.11 环境并安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

本目录可独立复制和安装。原生数据准备、训练与推理不依赖宿主项目。
准备本地 Qwen3-VL-4B-Instruct-Action 目录，包含权重、config、tokenizer、
processor 和 chat template。训练时通过 `--base-vlm` 或 `MOPA_BASE_VLM` 指定。
依赖使用 `transformers==4.57.0`，注意力实现为 SDPA。

## 数据准备

每个原始 episode 为一个 NPZ 文件，字段如下：

| 字段 | 格式 |
| --- | --- |
| `state` | float32 `[T,D]`，机械臂/夹爪状态 |
| `action` | float32 `[T,D]`，关节动作标签 |
| `instruction` | 非空标量字符串 |
| `image_0`、`image_1`、… | uint8 RGB `[T,H,W,3]`，按 metadata 的相机顺序排列 |

双臂状态/动作顺序为左臂、左夹爪、右臂、右夹爪。单臂使用机械臂、夹爪。
`metadata.json` 声明关节维度和相机布局，例如：

```json
{
  "action_type": "joint",
  "robot_action_dim_info": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
  "cameras": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
  "image_size": [224, 224]
}
```

```bash
mopa-prepare --source /path/to/raw_episodes \
  --metadata /path/to/metadata.json --output /path/to/dataset
```

输出包含 episode NPZ、`metadata.json` 和 `dataset_statistics.json`。
图像缩放为指定尺寸，保持 RGB。q01/q99 统计仅来自真实帧；训练时将状态和
动作归一化到 `[-1,1]`，常量维度映射为零。episode 末尾通过重复最后一个动作
补齐动作块。已有数据目录不会被覆盖。

## 训练

```bash
mopa-train --dataset /path/to/dataset --output /path/to/checkpoint \
  --base-vlm /path/to/Qwen3-VL-4B-Instruct-Action --seed 0 --device cuda
```

也可运行 `python -m mopa.training.cli`。默认训练 100000 步，batch size 为 8，
骨干学习率为 `1e-5`，其余参数为 `1e-4`。`--config` 接受模型参数 JSON；
query 数量、动作长度和关闭 base 的约束保持固定。完整选项见 `mopa-train --help`。
CPU 使用 `--device cpu`，骨干精度自动设为 float32。

训练保存解析后的模型配置、关节布局、相机顺序、数据路径、随机种子与训练参数。
复现时使用相同数据、Qwen 资产、依赖、配置和种子；GPU 数值结果仍可能随硬件变化。
训练为单进程，不恢复 optimizer/scheduler。非空输出目录不会被覆盖。

## 推理

checkpoint 包含以下三个文件，须一起保留：

```text
config.json
model.pt
dataset_statistics.json
```

```python
import numpy as np
from mopa.runtime import Policy

policy = Policy("/path/to/checkpoint", device="cuda")
# rgb_views: 按 policy.cameras 排列的 uint8 RGB 图像列表。
# joint_state: 按关节布局排列的原始状态向量 [D]。
actions = policy.predict(
    images=[rgb_views],
    instructions=["Place the bowl on the plate."],
    states=np.asarray([joint_state], dtype=np.float32),
)
# actions.shape == (1, 4, policy.action_dim)，已恢复为原始动作量纲。
```

Qwen 资产移动后可设置 `Policy(..., base_vlm="/new/path")`，资产内容须与训练时一致。
输入支持 batch，推理采用与训练相同的图像预处理和统计量。
checkpoint 格式为 `xpl-mopa-arm-v1`；包含 base 分支或双套 query 的权重无法直接加载。
