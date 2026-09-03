# Resilient Transformers Testing - Quick Start

## 新功能：Resilient Mode

Resilient模式让测试harness在遇到crash/hang时依然能完成测试并自动提issue。

### 核心改进

1. **分批运行** - 每批20个test，crash只影响当前批次
2. **增量输出** - 边跑边写，不等全部完成
3. **自动恢复** - Crash后自动继续下一批
4. **端到端自动化** - 从测试到提issue零人工介入

## 快速开始

### 单个模型（推荐）

```bash
# 测试bert并自动提issue
bash scripts/transformers_auto_sweep.sh bert gcu GCU flagos-ai/Torch-FL

# 测试qwen3
bash scripts/transformers_auto_sweep.sh qwen3 gcu GCU flagos-ai/Torch-FL
```

**参数说明**：
- `bert` / `qwen3` - 模型名称
- `gcu` - 设备名称（torch.flagos的device参数）
- `GCU` - 芯片名称（用于issue标题，如 `[AI][GCU]`）
- `flagos-ai/Torch-FL` - GitHub仓库

### 批量运行（bert + qwen3）

```bash
# 一次运行两个模型
bash scripts/transformers_batch_sweep.sh gcu GCU flagos-ai/Torch-FL
```

### 手动分步运行

如果需要更细粒度控制：

```bash
MODEL=bert
DEVICE=gcu
CHIP=GCU

# Step 1: 运行测试（resilient模式）
python tests/manual/transformers_hf_tests.py \
    --model ${MODEL} \
    --device ${DEVICE} \
    --resilient \
    --batch-size 20 \
    --batch-timeout 900 \
    --out /tmp/${MODEL}-results.json

# Step 2: Triage
python scripts/transformers_triage.py \
    /tmp/${MODEL}-results.json \
    --out /tmp/${MODEL}-classified.json

# Step 3: Verify
python scripts/transformers_verify.py \
    /tmp/${MODEL}-classified.json \
    --out /tmp/${MODEL}-verified.json \
    --test-source-dir tests/transformers/models/${MODEL} \
    --workers 4

# Step 4: Deduplicate
python scripts/transformers_deduplicate.py \
    /tmp/${MODEL}-verified.json \
    --out /tmp/${MODEL}-new.json \
    --coverage-file docs/reference/hf-coverage.md \
    --repo flagos-ai/Torch-FL

# Step 5: Preview
python scripts/transformers_preview_issues.py \
    /tmp/${MODEL}-new.json \
    --chip ${CHIP} \
    --transformers-version 4.47.0 \
    --torch-fl-commit $(git rev-parse --short HEAD) \
    --issue-bodies-dir /tmp/${MODEL}-issues \
    --out /tmp/${MODEL}-preview.md

# Step 6: Review and file
cat /tmp/${MODEL}-preview.md
python scripts/transformers_file_issues.py \
    /tmp/${MODEL}-new.json \
    --approve-all \
    --repo flagos-ai/Torch-FL
```

## Resilient模式参数

### `--resilient`
启用resilient模式（分批运行）

### `--batch-size N`
每批测试数量（默认：20）

**建议**：
- 20: 标准配置，平衡速度和隔离
- 10: 如果频繁crash
- 50: 如果测试稳定，想更快

### `--batch-timeout N`
每批超时时间，秒（默认：900 = 15分钟）

**建议**：
- 900: 标准配置
- 1800: 大模型（llama, qwen等）
- 600: 小模型（bert, distilbert等）

## 典型场景

### 场景1：燧原S60首次测试bert

```bash
# 在S60机器上
cd /path/to/PyTorch-Plugin-FL
bash scripts/transformers_auto_sweep.sh bert gcu GCU
```

**预期**：
- 自动分批运行所有bert测试
- 遇到crash继续下一批
- 最终自动提issue到GitHub

### 场景2：调试某个crash

如果知道某个test导致crash：

```bash
# 正常模式运行单个test
python tests/manual/transformers_hf_tests.py \
    --model bert \
    --pytest-arg "tests/models/bert/test_modeling_bert.py::BertModelTest::test_forward"
```

### 场景3：查看中间结果

Resilient模式的所有中间文件保存在 `/tmp/transformers-auto-sweep-<model>-<timestamp>/`：

```bash
ls -lh /tmp/transformers-auto-sweep-bert-*/

# 输出：
# test-results.json   - 原始测试结果
# classified.json     - 分类后的findings
# verified.json       - 隔离验证后
# new.json           - 去重后新issues
# preview.md         - Issue预览
# issues/*.md        - 各个issue的body
```

### 场景4：只运行测试，不提issue

修改 `transformers_auto_sweep.sh` 的第6步，或者手动运行前5步。

## 与旧模式对比

| 特性 | 旧模式 | Resilient模式 |
|-----|--------|--------------|
| 运行方式 | 一次性运行全部 | 分批运行 |
| Crash影响 | 整个挂掉 | 只影响当前批次 |
| 结果输出 | 全部完成后 | 边跑边写 |
| 超时处理 | 整体超时 | 每批独立超时 |
| 适用场景 | 稳定测试 | 新芯片/不稳定环境 |

## 故障排查

### 问题1：没有生成test-results.json

**原因**：模型名称错误或collect失败

**解决**：
```bash
# 检查模型名称
python tests/manual/transformers_hf_tests.py --list-models | grep bert

# 测试collect
python tests/manual/transformers_hf_tests.py --model bert --collect-only
```

### 问题2：所有batch都crash

**原因**：环境问题（驱动、依赖等）

**解决**：
```bash
# 检查设备
python -c "import torch, torch_fl; print(torch.flagos.device_count())"

# 检查依赖
pip list | grep -E "transformers|torch|accelerate"

# 运行sanity check
python -c "
import torch
import torch_fl
x = torch.randn(2, 3, device='flagos')
print('Device OK:', x.device)
"
```

### 问题3：Batch很慢

**原因**：Batch size太大或timeout太长

**解决**：
```bash
# 减小batch size
--batch-size 10

# 减小timeout
--batch-timeout 600
```

### 问题4：Issue没有提交

**原因**：GitHub认证或权限问题

**解决**：
```bash
# 检查gh CLI认证
gh auth status

# 测试dry-run
python scripts/transformers_file_issues.py \
    /tmp/bert-new.json \
    --approve-all \
    --dry-run \
    --repo flagos-ai/Torch-FL
```

## 配置建议

### 不同芯片的配置

**燧原GCU S60**:
```bash
--batch-size 20
--batch-timeout 900
```

**摩尔线程MUSA**:
```bash
--batch-size 20
--batch-timeout 900
```

**华为Ascend**:
```bash
--batch-size 15  # 如果内存受限
--batch-timeout 1200
```

### 不同模型的配置

**小模型 (bert, distilbert)**:
```bash
--batch-size 50
--batch-timeout 600
```

**中模型 (gpt2, t5)**:
```bash
--batch-size 20
--batch-timeout 900
```

**大模型 (llama, qwen3)**:
```bash
--batch-size 10
--batch-timeout 1800
```

## 最佳实践

1. **首次运行新芯片**：使用resilient模式 + 小batch (10-20)
2. **稳定环境**：可以用正常模式（更快）
3. **调试crash**：先用resilient找到crash的batch，再单独运行那批
4. **批量测试**：使用 `transformers_batch_sweep.sh`
5. **保留日志**：工作目录自动带时间戳，方便追溯

## 后续扩展

计划添加的功能：

- [ ] Preflight checks（环境预检）
- [ ] 进度实时显示
- [ ] Web dashboard
- [ ] 支持diffusers模型
- [ ] 多机并行运行

## 问题反馈

遇到问题请提issue到 flagos-ai/Torch-FL，附上：
1. 完整的命令行
2. 工作目录路径
3. test-results.json（如果有）
4. 错误信息
