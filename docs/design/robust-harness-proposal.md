# Robust Transformers Test Harness - 设计方案

## 问题定义

**现状**：燧原S60运行 transformers 测试时，遇到严重错误（segfault/hang）导致整个进程挂掉，无法完成测试，也就无法进入自动提issue流程。

**目标**：让测试harness足够鲁棒，即使遇到crash/hang也能：
1. 尽可能多地完成测试
2. 保存已完成的结果
3. 自动triage并提issue

## 设计原则

1. **Fail-safe**: 部分失败不阻塞整体
2. **Progressive output**: 边跑边写，不等全部完成
3. **Automatic recovery**: Crash后自动继续
4. **Zero manual intervention**: 全程自动化到提issue

## 核心改进

### 1. 增加 `--resilient` 模式

在 `transformers_hf_tests.py` 中添加新模式：

```python
def run_tests(model, source, args):
    if args.resilient:
        return run_tests_resilient(model, source, args)
    else:
        return run_tests_original(model, source, args)  # 当前实现

def run_tests_resilient(model, source, args):
    """Resilient mode: 即使crash也继续运行"""
    
    # Step 1: Collect all tests (不执行，只列出)
    collected = collect_all_tests(model, source, args)
    if not collected["tests"]:
        return empty_result("NO_TESTS_COLLECTED")
    
    all_test_nodeids = collected["tests"]
    print(f"Collected {len(all_test_nodeids)} tests for {model}")
    
    # Step 2: 分批运行
    BATCH_SIZE = 50  # 每批50个test，可配置
    results = []
    crashed_batches = []
    
    for batch_idx, start in enumerate(range(0, len(all_test_nodeids), BATCH_SIZE)):
        batch = all_test_nodeids[start:start+BATCH_SIZE]
        batch_name = f"batch_{batch_idx+1}"
        
        print(f"[{batch_name}] Running {len(batch)} tests...")
        
        try:
            batch_result = run_test_batch(
                model, source, batch, args,
                timeout_per_batch=args.batch_timeout or 600
            )
            
            results.extend(batch_result["tests"])
            print(f"[{batch_name}] ✓ {len(batch_result['tests'])} results")
            
        except BatchCrashError as e:
            # 这批crash了，记录并继续
            print(f"[{batch_name}] ✗ CRASHED: {e}")
            crashed_batches.append({
                "batch_idx": batch_idx,
                "test_nodeids": batch,
                "error": str(e)
            })
            
            # 添加占位结果
            for nodeid in batch:
                results.append({
                    "nodeid": nodeid,
                    "status": "BATCH_CRASHED",
                    "detail": f"Batch {batch_name} crashed: {e}"
                })
            
            # 尝试重置设备
            try:
                reset_device_context(args.device)
            except Exception as reset_error:
                print(f"Warning: Device reset failed: {reset_error}")
        
        # Step 3: 增量写入（重要！）
        if args.out:
            append_batch_to_json(args.out, {
                "batch_idx": batch_idx,
                "tests": results[-len(batch):],
                "crashed": batch_name in [b["batch_idx"] for b in crashed_batches]
            })
    
    # Step 4: 汇总结果
    return {
        "run": {
            "status": "COMPLETED_WITH_CRASHES" if crashed_batches else "COMPLETED",
            "crashed_batches": crashed_batches,
            "total_batches": (len(all_test_nodeids) + BATCH_SIZE - 1) // BATCH_SIZE,
            "completed_tests": len([r for r in results if r["status"] != "BATCH_CRASHED"])
        },
        "tests": results,
        "summary": summarize_statuses(results, [])
    }

def collect_all_tests(model, source, args):
    """Only collect tests, don't run them"""
    target = test_dir(source, model)
    workdir = prepare_workdir(model)
    env = child_env(source, args.device, workdir / "dummy.jsonl", args.offline)
    
    command = pytest_command(target, args.marks, ["--collect-only", "-q"], True)
    
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True,
            env=env, cwd=str(workdir), timeout=60
        )
        
        # 解析pytest --collect-only输出
        test_nodeids = parse_collected_tests(proc.stdout)
        return {"tests": test_nodeids, "collected": len(test_nodeids)}
    except Exception as e:
        return {"tests": [], "error": str(e)}

def run_test_batch(model, source, batch_nodeids, args, timeout_per_batch):
    """运行一批tests"""
    workdir = prepare_workdir(f"{model}_batch")
    report = workdir / "report.jsonl"
    report.touch()
    
    env = child_env(source, args.device, report, args.offline)
    
    # 用pytest的nodeid selection运行特定tests
    command = pytest_command(
        test_dir(source, model),
        args.marks,
        ["-p", "hf_report_plugin"] + batch_nodeids,  # 直接传nodeid
        collect_only=False
    )
    
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(workdir),
            timeout=timeout_per_batch
        )
        
        records = read_report(report)
        reduced = reduce_records(records)
        
        # 检查是否crash
        if pytest_process_crashed(proc.returncode):
            raise BatchCrashError(
                f"Process crashed with return code {proc.returncode}"
            )
        
        return reduced
        
    except subprocess.TimeoutExpired:
        raise BatchCrashError(f"Batch timed out after {timeout_per_batch}s")
    finally:
        # 清理临时目录
        shutil.rmtree(workdir, ignore_errors=True)

class BatchCrashError(Exception):
    pass

def append_batch_to_json(filepath, batch_data):
    """增量写入batch结果"""
    filepath = Path(filepath)
    
    # 读取现有数据
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
    else:
        data = {"batches": [], "tests": []}
    
    # 添加新batch
    data["batches"].append(batch_data)
    data["tests"].extend(batch_data["tests"])
    
    # 原子写入
    atomic_write(filepath, json.dumps(data, indent=2))

def reset_device_context(device):
    """尝试重置设备状态"""
    import torch
    
    if device == "flagos":
        try:
            torch.flagos.empty_cache()
            torch.flagos.synchronize()
            # 可能的额外重置逻辑
        except Exception as e:
            raise RuntimeError(f"Device reset failed: {e}")
```

### 2. 自动化端到端脚本

创建 `scripts/transformers_auto_sweep.sh`:

```bash
#!/bin/bash
# 端到端自动化：测试 → triage → verify → dedupe → file issues

set -e

MODEL=$1
DEVICE=${2:-gcu}
CHIP=${3:-GCU}  # 用于issue标题
REPO=${4:-flagos-ai/Torch-FL}

if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model> [device] [chip] [repo]"
    echo "Example: $0 qwen3 gcu GCU flagos-ai/Torch-FL"
    exit 1
fi

WORK_DIR=/tmp/transformers-auto-sweep-${MODEL}
mkdir -p ${WORK_DIR}

echo "================================================"
echo "Transformers Auto Sweep + Issue Filing"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Chip: $CHIP"
echo "Repo: $REPO"
echo "Work Dir: $WORK_DIR"
echo "================================================"

# Step 1: Run tests (resilient mode)
echo ""
echo "[1/6] Running tests (resilient mode)..."
python tests/manual/transformers_hf_tests.py \
    --model ${MODEL} \
    --device ${DEVICE} \
    --resilient \
    --out ${WORK_DIR}/test-results.json \
    --timeout 1800 \
    || echo "Warning: Tests completed with errors"

# 检查是否有结果
if [ ! -f ${WORK_DIR}/test-results.json ]; then
    echo "ERROR: No test results generated"
    exit 1
fi

TEST_COUNT=$(python -c "import json; d=json.load(open('${WORK_DIR}/test-results.json')); print(len(d.get('tests', [])))")
echo "Captured ${TEST_COUNT} test results"

if [ "$TEST_COUNT" -eq 0 ]; then
    echo "ERROR: No tests completed"
    exit 1
fi

# Step 2: Triage
echo ""
echo "[2/6] Triaging failures..."
python scripts/transformers_triage.py \
    ${WORK_DIR}/test-results.json \
    --out ${WORK_DIR}/classified.json

# Step 3: Verify (parallel)
echo ""
echo "[3/6] Verifying failures in isolation..."
python scripts/transformers_verify.py \
    ${WORK_DIR}/classified.json \
    --out ${WORK_DIR}/verified.json \
    --test-source-dir tests/transformers/models/${MODEL} \
    --workers 4 \
    --timeout 120 \
    || echo "Warning: Verification completed with errors"

# Step 4: Deduplicate
echo ""
echo "[4/6] Deduplicating against baseline and GitHub..."
python scripts/transformers_deduplicate.py \
    ${WORK_DIR}/verified.json \
    --out ${WORK_DIR}/new.json \
    --coverage-file docs/reference/hf-coverage.md \
    --repo ${REPO}

NEW_COUNT=$(python -c "import json; d=json.load(open('${WORK_DIR}/new.json')); print(len(d.get('findings', [])))")
echo "Found ${NEW_COUNT} new findings"

if [ "$NEW_COUNT" -eq 0 ]; then
    echo "No new issues to file. Done!"
    exit 0
fi

# Step 5: Preview
echo ""
echo "[5/6] Generating issue previews..."
python scripts/transformers_preview_issues.py \
    ${WORK_DIR}/new.json \
    --chip ${CHIP} \
    --transformers-version $(python -c "import transformers; print(transformers.__version__)") \
    --torch-fl-commit $(git rev-parse --short HEAD) \
    --issue-bodies-dir ${WORK_DIR}/issues \
    --out ${WORK_DIR}/preview.md

echo ""
echo "Issue preview:"
cat ${WORK_DIR}/preview.md

# Step 6: File issues
echo ""
echo "[6/6] Filing issues to GitHub..."
read -p "File ${NEW_COUNT} issues to ${REPO}? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/transformers_file_issues.py \
        ${WORK_DIR}/new.json \
        --approve-all \
        --repo ${REPO} \
        --issue-bodies-dir ${WORK_DIR}/issues
    
    echo ""
    echo "✓ Issues filed successfully!"
else
    echo "Skipped issue filing. Use --dry-run to test:"
    echo "  python scripts/transformers_file_issues.py ${WORK_DIR}/new.json --approve-all --dry-run"
fi

echo ""
echo "================================================"
echo "All done! Results in: ${WORK_DIR}"
echo "================================================"
```

使用：
```bash
# 单个模型
bash scripts/transformers_auto_sweep.sh qwen3 gcu GCU

# 批量运行所有模型
for model in bert gpt2 qwen3 llama; do
    bash scripts/transformers_auto_sweep.sh $model gcu GCU || true
done
```

### 3. 增强 triage 工具处理不完整结果

修改 `transformers_triage.py` 接受部分结果：

```python
def triage_failures(test_json):
    """Triage可以处理不完整的结果"""
    run_info = test_json.get("run", {})
    tests = test_json.get("tests", [])
    
    # 检查是否是resilient模式的结果
    if "crashed_batches" in run_info:
        print(f"Warning: {len(run_info['crashed_batches'])} batches crashed")
        print("Triaging completed tests only...")
    
    # 过滤掉BATCH_CRASHED状态的测试
    valid_tests = [t for t in tests if t.get("status") != "BATCH_CRASHED"]
    
    failures = [t for t in valid_tests if t.get("status") == "FAIL"]
    
    # ... 继续正常triage流程
```

### 4. 命令行参数扩展

```python
# transformers_hf_tests.py 添加参数
parser.add_argument(
    "--resilient",
    action="store_true",
    help="Resilient mode: continue running even if batches crash"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=50,
    help="Tests per batch in resilient mode (default: 50)"
)
parser.add_argument(
    "--batch-timeout",
    type=int,
    default=600,
    help="Timeout per batch in seconds (default: 600)"
)
```

## 实施优先级

### P0 - 立即实现（本周）
1. ✅ `--resilient` 模式基础实现
2. ✅ 分批运行 + 增量输出
3. ✅ `transformers_auto_sweep.sh` 端到端脚本

### P1 - 短期（下周）
4. Triage处理不完整结果
5. 设备重置逻辑（per-chip）
6. 更好的错误分类（OOM, SEGFAULT等）

### P2 - 中期（本月）
7. Preflight checks
8. 进度监控和日志
9. 跨芯片配置文件

## 测试计划

### 1. 单元测试
```bash
# 测试collect功能
pytest tests/unit/test_transformers_hf_tests.py::test_collect_all_tests

# 测试分批逻辑
pytest tests/unit/test_transformers_hf_tests.py::test_run_test_batch
```

### 2. 集成测试
```bash
# 模拟crash场景
python tests/manual/transformers_hf_tests.py \
    --model bert \
    --resilient \
    --batch-size 10 \
    --out /tmp/bert-resilient.json

# 验证部分结果能triage
python scripts/transformers_triage.py \
    /tmp/bert-resilient.json \
    --out /tmp/bert-triage.json
```

### 3. 真实场景测试
```bash
# 燧原S60实测
ssh gcu-s60-box
cd /path/to/torch-fl
bash scripts/transformers_auto_sweep.sh qwen3 gcu GCU
```

## 成功标准

1. ✅ 燧原S60能完整跑完qwen3测试（即使有crash）
2. ✅ 有部分结果就能自动提issue
3. ✅ Crash不阻塞后续测试
4. ✅ 端到端零人工介入

## 风险和缓解

| 风险 | 影响 | 缓解措施 |
|-----|------|---------|
| 分批运行太慢 | 测试时间×2-3 | 默认关闭，只在crash时启用 |
| Device poison跨批次 | 后续批次全失败 | 每批后尝试reset device |
| 磁盘空间不足 | 多次pytest启动占用大 | 及时清理workdir |
| 某批hang住 | 整体卡住 | 每批独立timeout |

## 后续扩展

- [ ] 支持diffusers模型
- [ ] 分布式运行（多机并行）
- [ ] Web dashboard监控进度
- [ ] 自动bisect找出首个failing commit
