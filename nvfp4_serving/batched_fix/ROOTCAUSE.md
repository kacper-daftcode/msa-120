# Root cause: multi-group SFA-offset bug in `group_gemm_nvfp4_nt_groupwise` (flashinfer 0.6.12, SM120)

## TL;DR
The grouped NVFP4 GEMM computes the **per-group base offset of the activation
scale-factor tensor** (SFA) with a formula that is only correct for groups
`i = 0` and `i = 1`. For every group `i >= 2` it over-advances the SFA pointer
by `floor(i*127 / 128) * 128` swizzled rows (one extra 128-row block at `i=2,3`,
two at `i>=4`, ...). Groups `>= 2` therefore read the **wrong activation scale
rows**, producing garbage. The single-group path (`num_groups == 1`, `i==0`) is
unaffected, which is exactly why the validated per-expert loop is numerically
correct while the batched call is not.

This is **not** an `m_indptr` / `nvfp4_block_scale_interleave` /
`convert_sf_to_mma_layout` mismatch on the caller side. The caller-side scale
layout is fine. The bug is a hard-coded, mathematically wrong offset expression
**inside the CUDA arg-setup kernel**.

## Offending source location
File (inside image `vllm/vllm-openai:minimax-m3`):

```
/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/gemm/group_gemm_nvfp4_groupwise_sm120.cuh
```

Kernel `compute_sm120_cutlass_nvfp4_group_gemm_args(...)`, lines **74-76**:

```cpp
constexpr size_t alignment_swizzled_mn = 128;                 // line 59
...
// This formulation ensures that sf_m_offset_next - sf_m_offset >= m_offset_next - m_offset
size_t sf_m_offset =
    (static_cast<size_t>(m_offset) + static_cast<size_t>(i) * (alignment_swizzled_mn - 1)) /
    alignment_swizzled_mn * alignment_swizzled_mn;            // lines 74-76
```

This `sf_m_offset` is used as the per-group base of the activation scale tensor:

- **SwapAB = true** (the default, `swap_ab=True`; what the validated path uses):
  the activation `A` is the kernel's *B* operand and the activation scale `SFA`
  is the kernel's *SFB*. Line **93**:
  ```cpp
  SFB_ptr[i] = safe_inc_ptr(SFB, sf_m_offset * sf_k);
  ```
- **SwapAB = false**: same `sf_m_offset` feeds the activation scale `SFA`,
  line **105**:
  ```cpp
  SFA_ptr[i] = safe_inc_ptr(SFA, sf_m_offset * sf_k);
  ```

The *weight* scale (the kernel's other SF operand) is offset by
`i * sf_n * sf_k` (lines 89-90 / 108-109) — a clean per-group stride — and is
correct. Only the **activation** SF uses the broken `sf_m_offset`.

The sibling kernel
`group_gemm_mxfp4_groupwise_sm120.cuh` contains the **identical** broken
formula (same lines), so MXFP4 grouped GEMM has the same bug; this is a shared
arg-setup bug, not an NVFP4 typo.

## The offset math: what's wrong vs. what's correct

`m_indptr` entries are padded to a multiple of 4 (kernel requirement). So
`m_offset = m_indptr[i] = sum_{j<i} pad4(m_j)`.

The activation scale tensor produced by
`fp4_quantize(..., is_sf_swizzled_layout=True)` for an `m x K` chunk is one
**128-row-padded swizzled block** of shape `ceil(m/128)*128` rows by `K/16`
cols. The natural batched layout concatenates one such block per group, so the
**correct** base row of group `i` is the cumulative sum of 128-padded sizes:

```
correct_base(i) = sum_{j<i} ceil(m_j / 128) * 128          (units: swizzled rows)
SFA_ptr[i]      = SFA + correct_base(i) * sf_k
```

The kernel instead computes:

```
sf_m_offset(i) = floor( (m_offset[i] + i*127) / 128 ) * 128
```

Decompose: `m_offset[i] = 128*K_i + r_i` (`0 <= r_i < 128`). Then

```
sf_m_offset(i) = floor( (128*K_i + r_i + 127*i) / 128 ) * 128
              = (K_i + floor((r_i + 127*i)/128)) * 128
```

The intended value was `(K_i + ceil(r_i/128)) * 128` = round `m_offset[i]` **up**
to a 128 multiple (rounding each prior group's tail up to a full SF block). The
correct "round-up" needs `+127` **once for the current offset's own remainder**,
*not* `+127` **per preceding group**. By scaling the slack with the group index
`i`, the formula injects `floor(127*i/128)` spurious extra 128-blocks:

| group i | spurious extra 128-blocks = floor(127·i/128) |
|--------:|---------------------------------------------:|
| 0       | 0  (correct) |
| 1       | 0  (correct) |
| 2       | 1  (off by +128 rows) |
| 3       | 2  (off by +256 rows) |
| 4       | 3  (off by +384 rows) |

Concrete demonstration (group sizes all exactly 128, so `m_offset = 128*i`):

```
i=2: m_offset=256, (256 + 2*127)=510, 510//128*128 = 384   (correct: 256)  -> +128
i=3: m_offset=384, (384 + 3*127)=765, 765//128*128 = 640   (correct: 384)  -> +256
```

Even with perfectly 128-aligned groups the formula is wrong from `i=2`. With
ragged groups it is also wrong (see `repro_multigroup_sfa.py`).

### Why single-group works (validates the diagnosis)
`num_groups == 1` => only `i == 0` => `sf_m_offset = floor(m_offset[0]/128)*128 =
floor(0/128)*128 = 0`. The activation SF base is 0, i.e. exactly the start of the
buffer. Correct. This is precisely the per-expert path in
`unfused_moe/unfused_moe.py::_nvfp4_gemm_one` (it passes `m_indptr=[0, mpad]`,
one group), which is why it matches the bf16 reference at the NVFP4 noise floor.

## Two ways to fix

### (Preferred) Python-level fix — no kernel rebuild
The broken offset is **deterministic** and depends only on `(m_indptr, i)`, which
the caller also knows. And critically, the kernel's per-group SFA blocks are
always **>= the 128-padded rows each group needs** (the `i*127` term only ever
*grows* the gaps; verified exhaustively over random raggedness — never
truncates). Therefore the caller can **pre-scatter** the activation scale-factor
buffer so that group `i`'s swizzled SF block physically begins at the row the
kernel will read, i.e. at:

```
kernel_sf_m_offset(i) = ((m_indptr[i] + i*127) // 128) * 128
```

Build `a_scale_padded` of `total_sf_rows = kernel_sf_m_offset(G-1) + ceil(m_{G-1}/128)*128`
rows x `sf_k` cols, zero-filled, and copy each group's
`fp4_quantize(is_sf_swizzled_layout=True)` swizzled block into rows
`[kernel_sf_m_offset(i) : kernel_sf_m_offset(i)+ceil(m_i/128)*128]`. Pass that as
`a_scale` (flattened). The kernel then reads each group's SF from the correct
place. The weight-scale side (`i*sf_n*sf_k` stride) is already correct and needs
the standard per-expert `nvfp4_block_scale_interleave -> convert_sf_to_mma_layout`
stacked along `num_groups`.

This is what `batched_nvfp4_moe.py::batched_swigluoai_nvfp4_moe` implements. It
yields ONE batched `group_gemm_nvfp4_nt_groupwise` call over all experts and is
numerically identical (bit-compatible scales) to the per-expert loop.

Caveat: it wastes a little SFA memory (the spurious gaps), bounded by
`O(num_groups * 128 * sf_k)` bytes of e4m3 — negligible (for E<=128, sf_k<=96,
that is < ~1.5 MB), and the activation FP4 data / output are unchanged
(those use the correct `m_offset` stride).

### (Alternative) Two-line CUDA fix — if a rebuild is acceptable
Replace lines 74-76 with a correct per-group cumulative 128-padded base. Since
the kernel only has `m_indptr` (padded to 4, NOT to 128) it cannot reconstruct
`sum ceil(m_j/128)*128` from `m_offset` alone in O(1); the robust fix is to make
the **whole pipeline** pad `m_indptr` entries to multiples of **128** (instead of
4) and then use the exact-passthrough offset:

```cpp
// Requires: every (m_indptr[i+1]-m_indptr[i]) is a multiple of 128,
//           and a_scale is laid out with that same 128-padded m.
size_t sf_m_offset = static_cast<size_t>(m_offset);   // already a 128 multiple
```

Diff:
```diff
-  // This formulation ensures that sf_m_offset_next - sf_m_offset >= m_offset_next - m_offset
-  size_t sf_m_offset =
-      (static_cast<size_t>(m_offset) + static_cast<size_t>(i) * (alignment_swizzled_mn - 1)) /
-      alignment_swizzled_mn * alignment_swizzled_mn;
+  // m_indptr entries are padded to a multiple of 128 by the caller, so the
+  // activation scale base is simply m_offset (no per-group slack).
+  size_t sf_m_offset = static_cast<size_t>(m_offset);
```
Apply the identical change in `group_gemm_mxfp4_groupwise_sm120.cuh`. This is
cleaner but requires (a) recompiling the SM120 gemm module and (b) padding
`m_indptr` to 128 everywhere (and the activation FP4 `A`/`D` strides, which key
off `m_offset`, stay consistent because they also use `m_offset`). Because
recompiling inside the shipped image is heavyweight and the Python pre-scatter is
exact, the Python fix is the recommended integration.
