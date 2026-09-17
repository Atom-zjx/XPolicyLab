# MoPA

[MoPA: Coordinated Mobile Manipulation via Subsystem-Specific Perception Alignment](https://mopa-policy.github.io/)
的 RoboDojo 机械臂策略使用一套 **8 个 manipulation query**，关闭 base 分支，预测 **4 步 joint 动作**。
输入为 RGB 图像、语言指令和机械臂/夹爪关节状态。

完整模型项目位于小写 [`mopa/`](mopa/README.md)，包含安装配置、模型、数据处理、训练和推理。
它可以独立安装和运行。本层仅提供 XPolicyLab 入口：

```text
policy/MoPA/
├── mopa/                    # 独立模型项目
│   ├── pyproject.toml
│   ├── README.md
│   ├── models/、data/、training/
│   └── runtime.py
├── model.py                 # ModelTemplate 入口、共享路径和维度解析
├── process_data.py          # 数据转换命令入口
├── train.py                 # 训练命令入口
├── deploy.py / deploy.yml
└── *.sh                     # 安装、数据、训练和评估启动脚本
```

模型结构、原生数据格式及独立训练/推理见 [模型文档](mopa/README.md)。
通用参数、运行命名和部署方式见
[XPolicyLab README](../../README.md)。

## Installation

```bash
conda create -n mopa python=3.11 -y
conda activate mopa
cd XPolicyLab/policy/MoPA
bash install.sh
```

安装脚本安装内层 `mopa` 包和 XPolicyLab。准备包含权重、tokenizer、processor
与 chat template 的本地 Qwen3-VL-4B-Instruct-Action 目录。

## Data Processing

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> joint [expert_data_num] [options]

bash process_data.sh RoboDojo stack_bowls arx_x5 joint \
  --source /path/to/data/RoboDojo/stack_bowls/arx_x5/data
```

动作接口仅为 `joint`，命令以已注册的 `arx_x5` 为例；维度通过共享机器人配置读取。
默认输入为 `<parent>/data/<bench_name>/<ckpt_name>/<env_cfg_type>/data`，
`SOURCE_DATA` 或 `--source` 可覆盖。输出为
`data/<bench_name>-<ckpt_name>-<env_cfg_type>-joint/`，`--output` 可覆盖；已有目录不会被覆盖。

转换读取 HDF5 的 `state/`、`action/` 和 `vision/`，通过共享 `decode_image_bit`
解码为 RGB，交给内层数据处理生成 NPZ 和统计量，不使用 LeRobot 格式。
默认相机顺序为 `cam_head cam_left_wrist cam_right_wrist`，尺寸为 224×224；
可用 `--cameras`、`--image-size HEIGHT WIDTH` 设置。缺失指令时使用 `--instruction`，
多个语言改写取第一项。

## Training

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> joint <seed> <gpu_id> [options]

export MOPA_BASE_VLM=/path/to/Qwen3-VL-4B-Instruct-Action
bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0
```

入口通过共享维度工具核对数据，使用内层[默认配置](mopa/configs/model.json)
和训练器。`--dataset`、`--output` 可指定路径；其余选项见 `python train.py --help`。
checkpoint 保存到 `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-joint-<seed>/`：

```text
config.json
model.pt
dataset_statistics.json
```

三个文件须一起保留；格式为 `xpl-mopa-arm-v1`，加载时核对机器人和关节布局。
含 base 分支或双套 query 的权重无法直接加载，需要训练对应的机械臂策略。
当前训练为单进程，不恢复 optimizer/scheduler，非空输出目录不会被覆盖。

## Evaluation

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> joint <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 joint 0 \
  0 0 mopa base
```

评估需要上述 checkpoint 和父工作区中的 `env_cfg/`。`EVAL_ENV_TYPE=debug`
使用接口调试环境，`DEBUG_OBS_ENCODED=1` 启用编码图像传输。图像由服务端解码，
模型接收 RGB 数组；batch 观测与动作按 `env_idx` 对齐。

`deploy.yml` 模型选项：

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `checkpoint_path` | null | checkpoint 目录或 `model.pt`；省略时使用标准路径解析 |
| `base_vlm` | null | 同一组 Qwen 资产的新目录 |
| `device` | cuda | 推理设备；cpu 自动使用 float32 |
| `dtype` | bfloat16 | GPU 骨干精度；动作头为 float32 |
| `execute_steps` | null | 每次执行动作数，范围 1–4，默认完整 4 步 |
| `request_timeout_s` | 120 | RPC 超时秒数 |

语言字段为 `instruction`，兼容 `instructions`；不支持 `ee` 动作。
