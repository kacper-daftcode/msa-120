# MSA-SM120: FlashAttention for NVIDIA Blackwell Consumer GPUs (RTX 5090 / RTX PRO 6000)

> **One-liner:** Drop-in FlashAttention BF16 + FP8 kernels for SM120, using per-warp HMMA/QMMA.SF with hardware FP6/FP4 unpack — no TMA, no TMEM, works today on RTX 5090.

---

# Plan portu MSA na SM120 (RTX 5090)

**Data:** 2026-06-12
**Kontekst:** Hardware probes potwierdziły że SM120 ma działające per-warp QMMA.SF,
LDSM packed types, cp.async, ale NIE ma TMA (mbarrier TX broken) ani TMEM.

## Stan obecny MSA

MSA ma dwa stacki:
- **csrc JIT** — dense FlashAttention z CUTLASS, ~2000 linii C++ per kernel
- **CuTe-DSL** — sparse attention, ~3000 linii Python per kernel

Oba stacki głęboko polegają na SM100 infrastrukturze:

### Kluczowe SM100 zależności

| Prymityw SM100 | Gdzie w MSA | Status SM120 |
|---|---|---|
| `tcgen05.mma` (inline PTX) | `blackwell_helpers.py` L182 | BROKEN (wymaga TMEM) |
| Akumulator w TMEM | `atten_fwd.py` L145-161 | BROKEN (controller uninit) |
| TMA `cp.async.bulk.tensor.2d` | `tma_utils.py` L55 | BROKEN (mbarrier TX) |
| `mbarrier.arrive.expect_tx` | `pipeline.py` (PipelineTmaUmma) | BROKEN |
| SMEM descriptor (64-bit) | `mma_sm100_desc.py` L220 | N/A (per-warp nie potrzebuje) |
| MMA instruction descriptor | `mma_sm100_desc.py` L115 | N/A (per-warp nie potrzebuje) |
| `nvcc -gencode=compute_100a` | `jit.py` L186 | Zmiana na compute_120f |

### Działające zamienniki SM120

| SM120 feature | Zweryfikowane | Zamienia SM100... |
|---|---|---|
| QMMA.SF (block-scaled FP8/FP6) | ✅ D=[32,32,32,32] | tcgen05.mma kind::f16/f8f6f4 |
| QMMA.SF.SP (sparse 2:4) | ✅ D=[32,32,32,32] | tcgen05.mma.sp |
| HMMA (FP16/BF16 m16n8k16) | ✅ (standardowe) | tcgen05.mma kind::f16 |
| LDSM.U6x16P32TO8 | ✅ bit-perfect | Nie istnieje na SM100 (per-warp) |
| LDSM.U4x16P64TO8 | ✅ bit-perfect | j.w. |
| cp.async (LDGSTS) | ✅ 128/128 match | TMA cp.async.bulk.tensor |
| UBLKPF.L2 (bulk prefetch) | ✅ no crash | TMA prefetch |
| mbarrier basic | ✅ działa | mbarrier TX (broken) |
| stmatrix m16n8.x2.trans | ✅ działa | j.w. |

## Strategia portu

### Opcja A: Głęboki redesign kerneli (NIE REKOMENDOWANE na start)

Przepisać `atten_fwd.py` od zera z per-warp modelem:
- Warp-cooperative pipeline zamiast TMA+UMMA async
- Akumulator w rejestrach zamiast TMEM
- cp.async zamiast TMA

**Szacunek:** 2-4 tygodnie, wymaga gruntownej wiedzy o FlashAttention.

### Opcja B: Warstwa abstrakcji SM120 w CuTe-DSL (REKOMENDOWANE)

1. Stworzyć `src/sm120/` mirror `src/sm100/` z podmienionym:
   - MMA: HMMA/QMMA.SF per-warp zamiast tcgen05.mma
   - Loads: cp.async zamiast TMA
   - Pipeline: cp.async.commit/wait_group zamiast mbarrier TX
   - Akumulator: rejestry zamiast TMEM

2. Zachować algorytmiczną strukturę (softmax, block tiling, sparse scheduling)
   która jest arch-agnostic.

3. Dodać dispatch w `interface.py` / `api.py` na podstawie SM version.

**Szacunek:** 1-2 tygodnie.

### Opcja C: Minimalny PoC — dense FMHA tylko (NAJSZYBSZY START)

1. Napisać minimalny SM120 FlashAttention kernel:
   - HMMA BF16 (m16n8k16) — per-warp, dobrze znane z SM80/SM90
   - cp.async (LDGSTS) dla Q/K/V load
   - Standardowy online-softmax w rejestrach
   - Brak sparse, brak FP8, brak paging

2. Zintegrować go z MSA API (`fmha_sm100_plan`, `fmha_sm100`).

3. Potem iteracyjnie dodawać: FP8 (QMMA.SF), sparse (QMMA.SF.SP), paging.

**Szacunek:** 3-5 dni na PoC, potem 1-2 tygodnie na pełny feature parity.

## Rekomendowany plan wykonania

### Faza 0: Infrastruktura (kilka godzin)
- [x] Dodać `-gencode=compute_120f,code=sm_120f` do jit.py
- [ ] Dodać SM120 arch detection w `api.py` / `interface.py`
- [ ] Stworzyć `src/sm120/` directory z `__init__.py`

### Faza 1: Dense BF16 FlashAttention na SM120 ✅ DONE
- [x] Kernel `csrc/sm120_fmha_fwd.cu`: HMMA m16n8k16, cp.async, online softmax
- [x] torch extension `csrc/sm120_launch.cu` + test `tests/test_sm120_fmha.py`
- [x] Poprawność: max_err < 0.001 dla production sizes (64×64 .. 4096×4096)
- [x] GQA: 4/4 PASS (heads_q/heads_kv = 8/4, 16/1, 4/2, 32/8)
- [x] Determinism: bit-exact across runs
- [ ] Known limitation: partial KV tiles < 8 tokens (rows 8-15 output zeroed)
- [ ] Known limitation: extreme magnitude (>2x std) precision loss in softmax

### Faza 1 ORIGINAL PLAN (completed differently):
- [SKIP] Nowy kernel `src/sm120/fwd/atten_fwd.py`:
  - HMMA BF16 m16n8k16 per-warp (well-known from FA2/SM80)
  - cp.async GMEM→SMEM + commit/wait_group pipeline
  - Standardowy 2-stage pipeline (load next tile while computing current)
  - Online softmax w rejestrach
  - M=128, N=128, D=128 tiling
- [ ] Hookup do CuTe-DSL compile interface
- [ ] Testy poprawności vs SM100 reference output

### Faza 2: FP8 FlashAttention z QMMA.SF (1 tydzień)
- [ ] Zamienić HMMA na QMMA.SF (FP8 E4M3, block-scaled)
  - mma.sync.aligned.m16n8k32.kind::mxf8f6f4.block_scale...
  - UE8M0 scales inline w instrukcji MMA
- [ ] FP8 Q/K/V loading z LDSM packed types
- [ ] Testy numeryczne (tolerancja argmax, nie bit-exact)

### Faza 3: Sparse attention z QMMA.SF.SP (1 tydzień)
- [ ] Dodać 2:4 sparse path z QMMA.SF.SP
  - mma.sp::ordered_metadata K=64
  - Metadata loading + layout
- [ ] CSR scheduling (reuse z SM100)
- [ ] Sparse topk indexer (csrc JIT, proste dodanie sm_120f)

### Faza 4: Paging i decode (1 tydzień)
- [ ] Paged K/V support
- [ ] Decode kernel (różne tiling, batch-first)
- [ ] FP4/NVFP4 opcjonalnie (OMMA.SF + LDSM.U4x16P64TO8)

### Faza 5: Optymalizacja (ciągła)
- [ ] Tuning: tile sizes, pipeline depth, warp schedule
- [ ] Benchmarking vs SM100 native
- [ ] Profile z ncu/nsys

## Kluczowe decyzje architektoniczne

### 1. Akumulator: rejestry vs SMEM
SM100 trzyma akumulator w TMEM (256 KB per SM). SM120 musi trzymać go
w rejestrach (standard per-warp). To zmniejsza effective M-tile bo
rejestry są ograniczone per warp. FlashAttention v2/v3 na SM80/SM90
pokazały że to działa z M=128, D=128 w rejestrach.

### 2. Warp layout
SM100 MSA: 16 warpów w wyspecjalizowanych rolach (MMA, softmax, load, store).
SM120: Prostszy model — 4-8 warpów, każdy robi load+compute+softmax.
Albo: zachować warp-specialized ale z per-warp MMA zamiast 1-warp UTC.

### 3. Pipeline depth
SM100: Deep pipeline TMA→SMEM→TMEM (asynchroniczny, 2+ stages).
SM120: Shallower pipeline cp.async→SMEM→registers (2 stages wystarczy).

### 4. Data types
Priorytet: BF16 → FP8 (QMMA.SF) → FP6 (QMMA.SF E2M3) → FP4 (OMMA.SF)
BF16 jest najprostszy do zaimplementowania i zweryfikowania.
