# 资源目录统一记录

`resources/` 已按任务统一资源上下文，任务策略包使用 `coap/`，不再通过历史包名或修订目录区分实现。

```text
resources/
  make_kong/
    coap/
      tasks/
      assets/
      data/
  <task>/
    coap/
      adapter/       # Optional protocol adapters
      runtime/       # Optional task runtime resources
    configs/         # Optional task configuration
    vendor/          # Optional upstream dependency resources
```

实际子目录按任务需求保留；不为每个任务创建空的模板目录。SAM3 等上游依赖保留自己的包名和资源层级，模型内容哈希保持不变。

## 变更范围

- 调整 301 个资源条目的位置，更新 39 份任务模块清单中的资源上下文引用；全部资源任务均有 `coap/` 根目录。
- 607 份能力源码逐文件校验，与目录迁移前完全一致；内部导入别名保留兼容，不提供历史版本切换。
- 194 个读取相邻资源的 Python 文件继续与 `capabilities/` 中的源码共享硬链接，没有新增一份独立策略代码。
- 600 个资源链接重新计算相对目标，外部权重继续经过目录外的中性资源绑定；静态资源内容未改变。
- 各任务原有感知、几何、运动和任务实现分别保留。统一的是资源布局与调度方式。

`capabilities/` 存放实现，`assets/` 存放按内容哈希去重的静态资源，`resources/` 为各任务提供源码相邻资源的兼容布局。加载器执行能力源码，并将资源上下文绑定到对应任务的新目录。

## 验证

验证结果与迁移计数见 `resource_layout_audit.json`、`resource_layout_migration.json`。任务辅助函数对比见 `behavior_parity_after_layout.json`；调试入口日志见 `debug_layout_plain.log` 与 `debug_layout_encoded.log`。

资源布局验证覆盖全部 1,402 条模块映射、194 个源码硬链接、600 个相对链接及静态资源哈希。迁移前可用的静态资源引用均继续可用；原本缺少的可选权重与运行期输出不计为本次迁移新增问题。

本次验证包含回归测试、35 个任务的 206 个确定性辅助函数探针，以及普通和编码图像的调试闭环。未执行全任务仿真、GPU 模型推理或训练，不能据此声明完整仿真成功率。

## 后续维护

新增任务的资源上下文从 `resources/<task>/coap/` 开始组织，通过 `tasks/<task>/modules.json` 绑定到能力源码。移动资源时同步更新清单、重算相对链接，并验证原有相邻资源读取。外部机器资源通过运行配置绑定，公开目录内不记录真实机器路径。
