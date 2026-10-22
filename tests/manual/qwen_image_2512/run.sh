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
# Run one Qwen-Image-2512 manual test and report the readings that matter on a
# chip that has not run it before. Everything the run prints is kept, including
# the two logs the backend writes to stderr, so the numbers under the summary
# are a measurement of this run rather than a recollection of an earlier one.
#
# Usage:
#   tests/manual/qwen_image_2512/run.sh infer  [--stage ... --device ...]
#   tests/manual/qwen_image_2512/run.sh sweep  [--device ... --only ...]
#   tests/manual/qwen_image_2512/run.sh memprobe [--device ... --stage vae|full]
#   tests/manual/qwen_image_2512/run.sh side-by-side --left cuda --right flagos
#   tests/manual/qwen_image_2512/run.sh compare --a REF.png --b OUT.png
#   tests/manual/qwen_image_2512/run.sh census [LOG]     re-read a saved log
#
# Images land in one directory per backend under OUT_DIR -- <OUT_DIR>/flagos,
# <OUT_DIR>/musa, <OUT_DIR>/cuda -- so the vendor run and the flagos run of the
# same sweep do not overwrite each other, and side-by-side pairs them into
# <OUT_DIR>/compare.
#
# Environment:
#   PYTHON    interpreter to run; it must be the environment that has torch_fl
#             for the chip under test (default: python3)
#   HF_HOME   the cache that holds the model. Not defaulted here: leaving it
#             unset means the huggingface_hub default, which is usually NOT
#             where a shared model cache lives, and pointing it at an empty
#             directory looks exactly like a missing model.
#   OUT_DIR   where images and logs land (default: ./qwen-image-out)
#   LOG       log path override (default: $OUT_DIR/<mode>-<timestamp>.log)
#
#   FLAGOS_LOG=fallback,dispatch is exported for the run: `fallback` is the
#   cpu_fallback census, `dispatch` the per-op backend census.
#   HF_HUB_OFFLINE is left alone -- set it to 1 yourself once the cache is
#   complete and the box has no proxy.
#
# The procedure this drives, with the per-stage expectations:
#   tests/manual/qwen_image_2512/README.md

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SUBDIR=tests/manual/qwen_image_2512
PYTHON=${PYTHON:-python3}
OUT_DIR=${OUT_DIR:-$PWD/qwen-image-out}

# Both diagnostics are named in the one list. An inherited FLAGOS_LOG wins, so
# a narrower run (FLAGOS_LOG=dispatch) is not widened back to both.
export FLAGOS_LOG=${FLAGOS_LOG:-fallback,dispatch}

usage() {
    echo "usage: $(basename "${BASH_SOURCE[0]}") <infer|sweep|memprobe|side-by-side|compare|census> [args...]" >&2
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
    echo "libentry failures  : $(count 'failed to compile')"
    echo "dispatch records   : $(count 'flagos dispatch')"
    echo "distinct ATen ops  : $(grep -o 'dispatch\] [^ ]*' "$LOG" 2>/dev/null | sort -u | wc -l)"

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

MODE=${1:-}
[ -n "$MODE" ] || usage
shift || true

case "$MODE" in
infer) SCRIPT=$SUBDIR/infer.py ;;
sweep) SCRIPT=$SUBDIR/sweep.py ;;
memprobe) SCRIPT=$SUBDIR/memprobe.py ;;
compare) SCRIPT=$SUBDIR/compare.py ;;
side-by-side) SCRIPT=$SUBDIR/side_by_side.py ;;
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
if [ "$MODE" = "side-by-side" ] && ! has_flag --root "$@"; then
    ARGS=(--root "$OUT_DIR" ${ARGS[@]+"${ARGS[@]}"})
fi

LOG=${LOG:-$OUT_DIR/$MODE-$(date +%Y%m%d-%H%M%S).log}
echo "python : $PYTHON"
echo "script : $ROOT/$SCRIPT"
case "$MODE" in
infer | sweep | memprobe) echo "output : $DEVICE_DIR" ;;
esac
echo "log    : $LOG"
echo

(cd "$ROOT" && "$PYTHON" "$SCRIPT" ${ARGS[@]+"${ARGS[@]}"}) 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

summarise
echo "exit status        : $STATUS"
exit "$STATUS"
