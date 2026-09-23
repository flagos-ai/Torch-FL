#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Run one Qwen-Image-2.1 manual test and report the readings that matter on a
# chip that has not run it before. Everything the run prints is kept, including
# the backend's own stderr logs, so the numbers under the summary are a
# measurement of this run rather than a recollection of an earlier one.
#
# Usage:
#   tests/manual/qwen_image_21/run.sh infer  [--stage ... --device ...]
#   tests/manual/qwen_image_21/run.sh sweep  [--device ... --only ...]
#   tests/manual/qwen_image_21/run.sh bench  [--device ... --batch N [N ...] ...]
#   tests/manual/qwen_image_21/run.sh table  JSON... [--baseline LABEL]
#   tests/manual/qwen_image_21/run.sh numerics      run the per-op comparison
#   tests/manual/qwen_image_21/run.sh side-by-side --left cuda --right flagos
#   tests/manual/qwen_image_21/run.sh compare --a REF.png --b OUT.png
#   tests/manual/qwen_image_21/run.sh census [LOG]  re-read a saved log
#
# bench with one --batch value writes the one record it always did. With several
# it writes one record per value -- bench-b1.json, bench-b2.json, ... -- and this
# script summarises every one of them, because a throughput is a curve rather
# than a point: whether a card saturates, and at what batch, is the difference
# between arithmetic-bound and host-bound. --peak-tflops and
# --peak-bandwidth-gbs add the two achievement ratios (MFU for the denoise loop,
# MBU for the elementwise probe). Both are recorded, and without them the FLOPs
# and the GB/s are still measured while the ratios read n/a.
#
# All device runs also accept wrapper-only `--run-dir DIR` and `--log FILE`.
# They are removed before the remaining arguments are passed to Python, so callers
# do not need to export OUT_DIR or LOG merely to choose the artifact paths.
#
# Images land in one directory per backend under OUT_DIR -- <OUT_DIR>/flagos,
# <OUT_DIR>/musa, <OUT_DIR>/cuda -- so the vendor run and the flagos run of the
# same sweep do not overwrite each other, and side-by-side pairs them into
# <OUT_DIR>/compare.
#
# bench writes one JSON per backend into its own directory; table reads those
# JSONs and needs no torch and no device, so it runs anywhere the files can be
# read -- including a laptop, from files carried off the chip.
#
# compare and side-by-side are model-agnostic: they read PNGs and a manifest and
# know nothing about torch, so they live once, in the Qwen-Image-2512 directory,
# and this script dispatches to them. Two copies would drift.
#
# Environment:
#   PYTHON    interpreter to run; it must be the environment that has torch_fl
#             for the chip under test (default: python3)
#   HF_HOME   the cache that holds the model, when the model is a hub id. Not
#             defaulted here: leaving it unset means the huggingface_hub
#             default, which is usually NOT where a shared model cache lives,
#             and pointing it at an empty directory looks exactly like a
#             missing model.
#   QWEN_IMAGE_21_MODEL     the weights: a directory or a hub id
#   QWEN_IMAGE_21_DIFFUSERS optional diffusers source checkout that defines the
#                           2.1 classes; not needed when the installed package
#                           contains a compatible backport
#   OUT_DIR   where images and logs land (default: ./qwen-image-21-out)
#   LOG       log path override (default: $OUT_DIR/<mode>-<timestamp>.log)
#
#   FLAGOS_LOG is exported for the run. It defaults to `fallback` for bench,
#   which measures, and `dispatch,fallback` for infer and sweep, which are
#   censuses -- the dispatch census costs real time per operator call and the
#   README records it as 12% of the denoise loop on the 2512 workload. Set it
#   explicitly to override either default; bench records what it actually saw.
#   The older per-diagnostic variables are retired and do nothing.
#
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is exported too, and on a
#   40 GB card it is not optional: the pipeline reserves 38.6 GiB of 39.5 GiB
#   usable, and without it the allocator fragments and the run dies in the loop
#   reporting "reserved but unallocated". See the README, §1.2.
#
# The procedure this drives, with the per-stage expectations:
#   tests/manual/qwen_image_21/README.md

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SUBDIR=tests/manual/qwen_image_21
SHARED=tests/manual/qwen_image_2512
PYTHON=${PYTHON:-python3}
OUT_DIR=${OUT_DIR:-$PWD/qwen-image-21-out}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

usage() {
    echo "usage: $(basename "${BASH_SOURCE[0]}") <infer|sweep|bench|table|numerics|side-by-side|compare|census> [args...]" >&2
    echo "see the header of this file for the modes and the environment" >&2
    exit 2
}

has_flag() {
    local want=$1
    shift
    local arg
    for arg in "$@"; do
        case "$arg" in
        "$want" | "$want"=*) return 0 ;;
        esac
    done
    return 1
}

# The value of --flag, whether written as "--flag value" or "--flag=value".
value_of_flag() {
    local want=$1
    local default=$2
    local prev=""
    local arg
    shift 2
    for arg in "$@"; do
        case "$arg" in
        "$want"=*) echo "${arg#*=}"; return ;;
        esac
        [ "$prev" = "$want" ] && { echo "$arg"; return; }
        prev=$arg
    done
    echo "$default"
}

# grep -c prints 0 and exits 1 when nothing matches, so the value is taken from
# the output and never from the status.
count() {
    local n
    n=$(grep -c -- "$1" "$LOG" 2>/dev/null)
    echo "${n:-0}"
}

summarise() {
    echo
    echo "---- readings ----"
    echo "log                : $LOG"
    echo "cpu_fallback ops   : $(count 'flagos cpu_fallback')"
    echo "dispatch records   : $(count 'flagos dispatch')"
    echo "distinct ATen ops  : $(grep -o 'dispatch\] [^ ]*' "$LOG" 2>/dev/null | sort -u | wc -l)"
    echo "libentry failures  : $(count 'failed to compile')"

    echo
    echo "---- operator calls by backend ----"
    grep -o 'flagos dispatch\] [^ ]* -> [a-z_]*' "$LOG" 2>/dev/null |
        awk '{print $NF}' | sort | uniq -c | sort -rn

    echo
    echo "---- distinct operators by backend ----"
    grep -o 'flagos dispatch\] [^ ]* -> [a-z_]*' "$LOG" 2>/dev/null |
        sort -u | awk '{print $NF}' | sort | uniq -c | sort -rn

    echo
    echo "---- which operators ----"
    for backend in $(grep -o 'flagos dispatch\] [^ ]* -> [a-z_]*' "$LOG" 2>/dev/null |
        awk '{print $NF}' | sort -u); do
        echo "$backend:"
        grep -o "flagos dispatch\] [^ ]* -> $backend\$" "$LOG" 2>/dev/null |
            awk '{print $3}' | sort -u | tr '\n' ' ' | fold -s -w 76 | sed 's/^/  /'
        echo
    done
    echo
    echo "a non-zero cpu_fallback count names ops that fell off the accelerator"
    echo "entirely; the op list itself is in the log, next to each count."
}

# bench measures, so it reports the measurement rather than a census: the JSON it
# wrote is the record, and re-reading it here is what keeps the summary and the
# artifact from disagreeing. The keys are read by name so a missing one prints
# "n/a" instead of failing the run that produced it.
#
# The path comes in as $1 rather than being assumed, so an explicit --out is
# summarised from where it actually went.
summarise_bench() {
    local json=$1
    echo
    echo "---- readings ----"
    echo "log                : $LOG"
    echo "json               : $json"
    if [ ! -f "$json" ]; then
        echo "no JSON was written; the run's own output above is the record"
        return
    fi
    "$PYTHON" - "$json" "$ROOT/$SUBDIR" <<'PY'
import json
import sys

sys.path.insert(0, sys.argv[2])
import cost

record = json.load(open(sys.argv[1], encoding="utf-8"))


def show(label, *keys, fmt="{:.2f}", suffix=""):
    value = record
    for key in keys:
        if not isinstance(value, dict):
            value = None
            break
        value = value.get(key)
    print(f"{label:<20} : {'n/a' if value is None else fmt.format(value)}{suffix}")


show("batch              ", "config", "batch", fmt="{}")
show("latency            ", "latency", "per_image_s", "median_s", suffix=" s/image (median)")
show("latency mean       ", "latency", "per_image_s", "mean_s", suffix=" s/image")
show("latency std        ", "latency", "per_image_s", "std_s", fmt="{:.3f}", suffix=" s")
show("latency p90        ", "latency", "per_image_s", "p90_s", suffix=" s/image")
show("latency p99        ", "latency", "per_image_s", "p99_s", suffix=" s/image")
show("measured calls     ", "protocol", "runs", fmt="{:.0f}")
show("loop per step      ", "phases", "loop per step", "median_s", fmt="{:.3f}", suffix=" s")
show("throughput         ", "throughput", "images_per_s", fmt="{:.4f}", suffix=" images/s")
show("transformer FLOPs  ", "compute", "flops_per_image", fmt="{:.4e}", suffix=" FLOP/image")
show("loop per image     ", "compute", "loop", "seconds_per_image", suffix=" s/image (loop)")
show("achieved           ", "compute", "loop", "tflops", suffix=" TFLOP/s (loop)")
show("MFU                ", "compute", "loop", "mfu", fmt="{:.1%}")
gbs = cost.probe_summary((record.get("compute") or {}).get("bandwidth"))
print(
    f"{'probe bandwidth    ':<20} : "
    + ("n/a" if gbs is None else f"{gbs['op']} {gbs['median_gb_per_s']:.0f} GB/s")
    + ("" if gbs is None or gbs["mbu"] is None else f"  MBU {gbs['mbu']:.1%}")
)
if gbs is not None:
    print(f"{'probe spread       ':<20} : {cost.probe_spread(gbs)}")
show("peak memory        ", "memory", "peak_gib", suffix=" GiB")
show("memory counter     ", "memory", "source", fmt="{}")
print(f"{'determinism        ':<20} : {record['determinism']['identical']}")
print(f"{'prompt             ':<20} : {record['prompt']['id']} sha256 {record['prompt']['sha256'][:16]}...")
print(f"{'FLAGOS_LOG         ':<20} : {record['environment']['log'] or 'unset'}")
print(f"{'triton cache       ':<20} : {record['environment']['triton_cache_dir'] or 'default'}")
for note in record.get("notes", []):
    print(f"{'note               ':<20} : {note}")
PY
    echo
    echo "render several of these together, on any interpreter:"
    echo "  bash $ROOT/$SUBDIR/run.sh table <json>... --baseline <label>"
}

MODE=${1:-}
[ -n "$MODE" ] || usage
shift || true

# Diagnostics are one comma-separated list, FLAGOS_LOG. The three separate
# variables this script used to set -- FLAGOS_LOG_DISPATCH, FLAGOS_LOG_FALLBACK,
# FLAGOS_CACHE_STATS -- are retired and read by nothing, so exporting them would
# silently produce no logging at all while the summary reported zero operators
# as though the workload had not run any.
#
# bench measures, the other modes count operators. `dispatch` writes one
# unbuffered line per operator call and costs real time per call -- 12% of the
# denoise loop on the 2512 workload -- so it is left out exactly where a number
# is going to be quoted, and included everywhere the log is the point.
# `fallback` is cheap: it prints only when an operator actually leaves the
# accelerator, so it stays on for every mode. An explicit FLAGOS_LOG in the
# caller's environment wins, and bench records what it actually saw.
case "$MODE" in
bench) DEFAULT_LOG=fallback ;;
*) DEFAULT_LOG=dispatch,fallback ;;
esac
export FLAGOS_LOG=${FLAGOS_LOG:-$DEFAULT_LOG}

case "$MODE" in
infer) SCRIPT=$SUBDIR/infer.py ;;
sweep) SCRIPT=$SUBDIR/sweep.py ;;
bench) SCRIPT=$SUBDIR/bench.py ;;
# table reads bench JSONs and knows nothing about torch or the device it ran on,
# so it is dispatched before any of the per-backend directory work below.
table) SCRIPT=$SUBDIR/bench.py ;;
numerics) SCRIPT=$SUBDIR/numerics.py ;;
# Model-agnostic, so they live with the 2512 flow and are dispatched to from
# here rather than copied: see the header.
compare) SCRIPT=$SHARED/compare.py ;;
side-by-side) SCRIPT=$SHARED/side_by_side.py ;;
census)
    LOG=${1:-${LOG:-}}
    # census takes a log path, not flags, so anything dash-led is a mistake
    # rather than a filename that has not been written yet.
    case "$LOG" in -*) usage ;; esac
    [ -n "$LOG" ] || usage
    [ -f "$LOG" ] || { echo "no such log: $LOG" >&2; exit 1; }
    summarise
    exit 0
    ;;
*) usage ;;
esac

if [ "$MODE" = "table" ]; then
    # No device, no OUT_DIR, no log: this reads files and prints a table. Called
    # directly rather than through the backend-directory machinery below, which
    # would need a --device it does not have an opinion about.
    exec "$PYTHON" "$ROOT/$SCRIPT" --table "$@"
fi

# `--run-dir` and `--log` belong to this wrapper rather than to the individual
# Python tools. Accept both spellings and forward every other argument unchanged.
FORWARDED=()
while [ "$#" -gt 0 ]; do
    case "$1" in
    --log)
        [ "$#" -ge 2 ] || { echo "--log requires a path" >&2; exit 2; }
        LOG=$2
        shift 2
        ;;
    --log=*)
        LOG=${1#*=}
        shift
        ;;
    --run-dir)
        [ "$#" -ge 2 ] || { echo "--run-dir requires a path" >&2; exit 2; }
        OUT_DIR=$2
        shift 2
        ;;
    --run-dir=*)
        OUT_DIR=${1#*=}
        shift
        ;;
    *)
        FORWARDED+=("$1")
        shift
        ;;
    esac
done
# `${FORWARDED[@]+...}` rather than a bare expansion, matching ARGS below: an
# empty array under `set -u` is an unbound variable before bash 4.4.
set -- ${FORWARDED[@]+"${FORWARDED[@]}"}

# Each backend writes its own directory, so the vendor run and the flagos run of
# the same sweep sit next to each other instead of overwriting one another.
DEVICE=$(value_of_flag --device flagos "$@")
DEVICE_DIR=$OUT_DIR/$DEVICE
mkdir -p "$DEVICE_DIR"

# Fill in an output path only when the caller did not name one, so an explicit
# --output / --output-dir still wins.
ARGS=("$@")
if [ "$MODE" = "infer" ] && ! has_flag --output "$@"; then
    ARGS+=(--output "$DEVICE_DIR/$(value_of_flag --stage full "$@").png")
fi
if [ "$MODE" = "sweep" ] && ! has_flag --output-dir "$@"; then
    ARGS=(--output-dir "$DEVICE_DIR" ${ARGS[@]+"${ARGS[@]}"})
fi
if [ "$MODE" = "bench" ] && ! has_flag --out "$@"; then
    ARGS+=(--out "$DEVICE_DIR/bench.json")
fi
if [ "$MODE" = "numerics" ] && ! has_flag --out "$@"; then
    ARGS+=(--out "$DEVICE_DIR/numerics.pt")
fi
if [ "$MODE" = "side-by-side" ] && ! has_flag --root "$@"; then
    ARGS=(--root "$OUT_DIR" ${ARGS[@]+"${ARGS[@]}"})
fi

LOG=${LOG:-$OUT_DIR/$MODE-$(date +%Y%m%d-%H%M%S).log}
echo "python : $PYTHON"
echo "script : $ROOT/$SCRIPT"
case "$MODE" in
infer | sweep | bench | numerics) echo "output : $DEVICE_DIR" ;;
esac
echo "log    : $LOG"
echo

(cd "$ROOT" && "$PYTHON" "$SCRIPT" ${ARGS[@]+"${ARGS[@]}"}) 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

# Compute the summary completely before appending it.  summarise reads LOG, so
# piping it directly through `tee -a "$LOG"` would make its later grep calls
# observe a file that is changing underneath them.
if [ "$MODE" = "bench" ]; then
    SUMMARY=
    # bench prints one "wrote <path>" line per record, which is a list whenever
    # --batch named several values. Reading them back out of this run's own log is
    # what keeps the summary and the artifact from disagreeing: deriving the names
    # here would be a second opinion about where bench.py put them, and the two
    # would drift the first time the naming changed.
    WROTE=$(grep -o '^wrote .*' "$LOG" 2>/dev/null | awk '{print $2}')
    if [ -n "$WROTE" ]; then
        for JSON in $WROTE; do
            RECORD_SUMMARY=$(summarise_bench "$JSON")
            if [ -n "$SUMMARY" ]; then
                SUMMARY="$SUMMARY
$RECORD_SUMMARY"
            else
                SUMMARY=$RECORD_SUMMARY
            fi
        done
    else
        # Nothing was written -- the run died before its first record -- so report
        # where it would have gone rather than nothing at all.
        SUMMARY=$(summarise_bench "$(value_of_flag --out "$DEVICE_DIR/bench.json" ${ARGS[@]+"${ARGS[@]}"})")
    fi
else
    SUMMARY=$(summarise)
fi
printf '%s\n' "$SUMMARY" | tee -a "$LOG"
printf 'exit status        : %s\n' "$STATUS" | tee -a "$LOG"
exit "$STATUS"
