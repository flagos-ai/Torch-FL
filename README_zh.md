<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# torch-fl

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.x-orange.svg)](https://pytorch.org/)
[![CI](https://github.com/flagos-ai/PyTorch-Plugin-FL/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/flagos-ai/PyTorch-Plugin-FL/actions/workflows/ci.yml?query=branch%3Amain)

[文档](docs/) · [安装](docs/getting-started/installation.md) · [快速开始](docs/getting-started/quickstart.md) · [兼容性](docs/reference/compatibility.md) · [English](README.md)

FlagOS 软件栈的 PyTorch 设备插件。torch-fl 对外提供统一的 `flagos` 设备，并在可复用原生内核、可移植编译器内核、厂商原生实现和显式 CPU 回退之间路由算子。

## 概览

不同加速器厂商提供的运行时、编译器栈和 PyTorch 集成方式各不相同。torch-fl 通过统一的运行时和算子路由层屏蔽这些差异，提供符合 PyTorch 使用习惯的设备接口。

用户只需使用标准 PyTorch API 和 `flagos` 设备。插件根据平台能力和配置为每个算子选择内核实现，无需将不同厂商暴露为不同设备名称，也无需在迁移加速器时修改工作负载。

## 设计理念

torch-fl 遵循五项原则：

1. **PyTorch 原生接口** — 标准 PyTorch API 无需修改；用户面向 `flagos` 设备编程，而不是使用厂商专用扩展。

2. **统一逻辑设备** — 单一设备名称（`flagos`）抽象厂商差异，平台相关路由在算子层透明完成。

3. **分层算子后端** — 每个操作可以分发到不同实现。路由决策以算子为粒度，而不是以设备或模型为粒度。

4. **优先复用而非重写** — 在 dispatch 和 ABI 边界允许的情况下集成成熟内核与编译器栈，避免重复实现已有能力。

5. **明确能力边界** — 对不支持的操作和 CPU 回退路径进行明确说明，不将其描述为完整原生覆盖；通过状态等级区分已验证支持与实验性集成。

### 执行路径

torch-fl 主要支持三类算子执行策略：

- **厂商原生内核**：直接调用厂商运行时和算子库（ACLNN、mudnn、topsaten），由插件生成对应厂商 C/C++ API 的绑定代码。

- **兼容性 boxing**：当厂商栈提供可与 PrivateUse1 共存的独立 PyTorch dispatch key 时，以零拷贝方式转换张量元数据。CUDA boxing 通过外部 `libtorch_cuda.so` 复用 NVIDIA 内核。

- **可移植编译器内核**：使用 Triton 或兼容编译器后端生成的 FlagGems 内核，在多个加速器系列之间复用，无需逐平台重写。

同一平台可以组合多种执行路径。这些路径属于内部实现策略，而不是用户选择的产品等级。

## 架构

```text
PyTorch API
    |
flagos 设备（PrivateUse1）
    |
设备运行时 + 按算子路由
    |
FlagGems/编译器内核 | 兼容性 boxing | 厂商原生内核 | CPU 回退
    |
加速器运行时
```

上图为概念架构。组件设计、dispatch 内部机制、分布式集合通信、编译集成和 profiler 设计详见 [docs/architecture/](docs/architecture/)。

## 能力

torch-fl 在项目层面提供以下能力：

- PyTorch eager 张量操作和设备管理
- Autograd 与模型训练
- `torch.compile` 集成
- 通过 `ProcessGroupFlagOS` 支持分布式集合通信和 DDP
- `torch.profiler` 集成
- FlagGems/Triton 算子集成
- 在 PyTorch 语义允许时，为未覆盖操作提供显式 CPU 回退

各平台的能力可用性和验证状态并不相同。某项功能存在于 torch-fl 代码库中，并不代表每个平台都已实现或验证该功能。平台详情请参阅[兼容性矩阵](docs/reference/compatibility.md)。

## 硬件支持

| 平台 | 执行路径 | 已验证能力 | 状态 | 指南 |
|---|---|---|---|---|
| NVIDIA CUDA | 基于外部 `libtorch_cuda.so` 的 CUDA boxing | Eager、autograd、分布式（FlagCX/NCCL）、profiler（CUPTI）、FlagGems（Python + C++） | **稳定** | [CUDA](docs/vendors/cuda/installation.md) |
| MetaX | 通过 cu-bridge 进行 CUDA boxing，或使用 MetaX 原生内核 | Eager、autograd | **稳定** | [MetaX](docs/vendors/metax/installation.md) |
| Ascend | 原生 ACLNN 后端，通过 FlagTree（Triton 3.5）使用 FlagGems | Eager、autograd、RNG 套件 | **Beta** | [Ascend](docs/vendors/ascend/installation.md) |
| PPU | 针对 PPU CUDA 13 兼容 SDK 的 CUDA boxing | Eager、autograd | **实验性** | [PPU](docs/vendors/ppu/installation.md) |
| 海光 DCU | 基于 hipify DTK torch 的 CUDA boxing | Eager、autograd、profiler | **Beta** | [DCU](docs/vendors/dcu/installation.md) |
| 燧原 GCU | 原生 topsaten 后端，未路由及 int64 算子使用 CPU 回退 | Eager | **实验性** | [GCU](docs/vendors/gcu/installation.md) |
| 摩尔线程 MUSA | FlagGems 优先的 Triton 内核，原生 mudnn 回退，未路由算子使用 CPU 回退 | Eager、FP16/BF16 AMP | **实验性** | [MUSA](docs/vendors/musa/installation.md) |
| 地平线 BPU | 无 eager 内核；通过 hbdk4 使用 `torch.compile` 图执行路径 | 仅图编译 | **仅运行时** | [BPU](docs/vendors/bpu/installation.md) |
| 清微智能 | 已提供运行时构建选择器 | 尚无安装文档 | **仅运行时** | — |

**状态定义：**

- **稳定**：关键路径持续接受测试，并已记录受支持的版本组合。
- **Beta**：主要路径已经验证，但覆盖范围、打包或发布流程尚未稳定。
- **实验性**：已在特定配置、模型或硬件环境中完成验证；接口或构建流程仍可能变化。
- **仅运行时**：已提供设备运行时支持，但该平台不是通用 eager 算子后端。

Eager 执行、训练、编译、分布式、profiler、FlagGems 等能力的详细拆分以及真实硬件验证结果，请参阅[兼容性矩阵](docs/reference/compatibility.md)。

## 兼容性

| 组件 | 支持范围 | 说明 |
|---|---|---|
| Python | 3.8 或更高版本 | 平台 SDK 和可用 wheel 可能要求更窄的版本范围。 |
| PyTorch | 2.10.x（`>=2.10,<2.11`） | 生成的 ATen 绑定与该次版本线绑定。 |
| FlagGems | 取决于平台 | 仅在平台路由使用 FlagGems 时，从 PyPI 或厂商兼容构建安装。 |
| Triton/编译器 | 取决于平台 | 使用所选加速器要求的编译器发行版。 |

### ATen 次版本绑定

torch-fl 根据 PyTorch 内部 ATen 算子注册表生成原生绑定。这些绑定对 C++ ABI 和算子 schema 变化敏感，因此项目固定使用一个 PyTorch 次版本线。当前固定版本线为 **2.10.x**。

使用其他 PyTorch 次版本（例如 2.11.x）会导致构建或运行时失败。同一次版本线内的补丁版本（例如 2.10.0 到 2.10.1）兼容。

厂商 SDK、CUDA toolkit 及其他平台特定版本要求记录在各平台安装指南中。

## 快速开始

先在[安装指南](docs/getting-started/installation.md)中选择平台，然后运行：

```python
import torch
import torch_fl

# 在 flagos 设备上创建张量
x = torch.randn(4, 4, device="flagos:0")

# 操作会路由到适合当前平台的内核
y = torch.relu(x @ x)

# 将结果移回 CPU
print(y.cpu())
```

算子具体路由到 FlagGems、厂商内核、兼容性 boxing 或 CPU 回退，由平台检测和运行时配置决定。以上代码在所有受支持加速器上保持不变。

设备查询、同步和多设备用法请参阅[快速开始指南](docs/getting-started/quickstart.md)。

## 文档

### 入门

- [安装](docs/getting-started/installation.md) — 平台选择和源码构建
- [快速开始](docs/getting-started/quickstart.md) — 基本使用方式

### 参考

- [兼容性矩阵](docs/reference/compatibility.md) — 各平台能力验证与状态
- [环境变量](docs/reference/environment-variables.md) — 构建和运行时配置

### 架构

- [分布式集合通信](docs/architecture/distributed-flagcx.md) — ProcessGroupFlagOS、FlagCX 和厂商回退
- [Profiler 集成](docs/architecture/profiler.md) — torch.profiler 对齐与 CUPTI 集成
- [torch.compile 集成](docs/architecture/torch-compile-integration.md) — Inductor GPU 设备注册

### 平台指南

- [CUDA（NVIDIA）](docs/vendors/cuda/installation.md)
- [MetaX](docs/vendors/metax/installation.md)
- [Ascend（华为）](docs/vendors/ascend/installation.md)
- [PPU](docs/vendors/ppu/installation.md)
- [海光 DCU](docs/vendors/dcu/installation.md)
- [燧原 GCU](docs/vendors/gcu/installation.md)
- [摩尔线程 MUSA](docs/vendors/musa/installation.md)
- [地平线 BPU](docs/vendors/bpu/installation.md)

## 参与贡献

欢迎参与以下方向的贡献：

- **算子**：补充缺失算子，或针对特定后端优化现有实现。
- **运行时和平台集成**：将 torch-fl 移植到新加速器，或改进现有后端支持。
- **编译器集成**：扩展 torch.compile 支持、改进 Triton 代码生成或增加编译后端。
- **分布式与 profiler**：增强集合通信后端或 profiling 集成。
- **测试**：增加算子正确性测试、模型集成测试或性能基准。
- **文档**：改进安装指南、故障排查文档或厂商集成说明。

开发流程、代码生成、测试要求和 PR 规范详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

**所有 GitHub 对外文本必须使用英文**，包括 PR 标题、PR 描述、commit message、issue 内容和 code review 评论。本仓库的代码、注释和历史记录均使用英文，PR 也由不阅读其他语言的贡献者审阅。

## 致谢

torch-fl 基于多个上游项目构建：

- **PyTorch** — PrivateUse1 扩展机制和 ATen 算子接口
- **FlagGems** — 基于 Triton 的可移植算子内核
- **Triton**（通过 FlagTree 和厂商发行版）— GPU 内核编译基础设施
- **FlagCX** — 异构集合通信库
- **厂商运行时和算子库** — ACLNN（Ascend）、mudnn（MUSA）、topsaten（GCU）、cu-bridge（MetaX）、DTK（DCU）等

以上致谢不代表所列项目或厂商对 torch-fl 的认可或背书。

## 许可证

torch-fl 使用 [Apache License 2.0](LICENSE) 许可证。
