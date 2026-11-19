# FlagGems-first Enflame validation plan

Date: 2026-11-18. This continues the native-host experiment on the measured
FlagGems 5.4.0 / FlagTree 0.7.0+enflame3.6 / TOPS 1.7.2.2402 S60 cohort.

## Evidence and objective

The previous vendor-only result does not qualify FlagGems model performance.
A metadata census of the complete Qwen3-0.6B eager model identifies its actual
operators, shapes, strides and dtypes. Individual substitutions, followed by a
combined full-prefill/eight-decode comparison, pass for nine FlagGems operators:
`mul.Tensor`, `add.Tensor`, `neg`, `cos`, `sin`, `where.self`, `silu`, `_softmax`
and `bmm`. The combined diagnostic records 7,389 successful kernel submissions.
Separate substitutions fail precision for `mean.dim`, `pow.Tensor_Scalar`,
`rsqrt` and `mm`; `cat` fails compilation on its mixed cache-concatenation sizes.
SiLU alone passes: the earlier combined mean/SiLU hold was too broad.

An ordinary dispatcher/config trial also passes all nine logit comparisons with
maximum absolute error zero and no Python tensor readback inside forwards.
One measured repetition reports 161.87/152.18 ms prefill/decode, so it does not
reach the 80% target. A separate two-forward API diagnostic records 1,684
Triton submissions, 2,582 stream synchronizes, and high pointwise host costs.
Instrumentation timings are not used for performance acceptance.

## Selected implementation

Extend the existing prebuilt C++ pointwise host machinery to the published
GCU300 wrapper ABI. Preserve the exact DMA predicate, both generated device
variants, dtype-dependent tile budget, MMU bound, warp rule, strides, extra tile
count and `ENABLE_I64` argument. Reuse the existing pointwise argument preparation
and dtype cache for supported tensors. Keep the compiled published preparation
and Cython schedule as the conservative fallback. No numerical device source,
dependency wheel, model layer, math setting or acceptance tolerance changes.

The alternative of simply calling published generic functions is insufficient:
the generic multiply's device-name gate redirects flagos tensors to ATen and
can report success with zero FlagGems submissions. Generic cat and bmm also
reproduce compiler/host incompatibilities on this cohort. Successful queue
submissions must therefore be counted separately from launcher construction.

## Verification and routing

Compare generated launch plans against the unchanged published wrapper over
dense, transposed, sliced and expanded inputs, scalar arguments and dtypes.
Exercise live outputs, cold/warm calls, nondefault streams and mixed producers.
Generate platform routes only after qualifying their implementations. Preserve
vendor paths for measured dtype/shape/compiler/precision gaps, with a concrete
reason for each remaining model route. Do not introduce CPU numerical fallback.

Rerun the installed wheel through ordinary strict eager execution, with no
diagnostic dispatch mode or preload. Require real FlagGems submission counts,
full prefill and every requested decode logit, a complete transfer audit, and
seven paired timing repetitions against isolated vendor PyTorch. Keep the
vendor-only result as historical evidence. If the first native-host change is
insufficient, use the new profile to select the next optimization.

## Measured follow-up

C++ preparation reduces the diagnostic trial to 94.27/86.56 ms. Admitting
source-derived pointwise entries and unrolling the published read-only variadic
dtype predicate reduces it to 78.08/70.25 ms. All nine comparisons remain exact.
A separate, temporary event-interoperability prototype passes 63 hardware cases
and the complete model, records 37,120 producer transitions, and reports
61.24/55.48 ms. Integrate this as explicit interop ABI 2: keep the existing
untracked pre/post drains and ABI 1 transition drains; serialize submissions,
record/wait a per-stream device event, release the lock on errors and destroy
the event when forgetting its stream. Final acceptance must use the installed
wheel without this prototype/preload.

The published FP32 `pow(2)` kernel fails, but an isolated composition using the
published FlagGems multiply passes all nine comparisons with 1,017 successful
submissions. Extend the GCU generator's scalar-pow category to express FP32
square through the selected multiply route. Other exponents and dtypes retain
the vendor implementation. Check special values and layouts on hardware.
