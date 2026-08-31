# `tests_jtoj24_scan` — tests for the PLANT escape scripts

Covers the three escape entry points under `scripts/JtoJ24_scan/`:

| Module | What is tested |
| --- | --- |
| `plant_order_scan.py` | the stationary-point escape geometry (`escape_basis`, `build_escape_tables`), the label placer, the colour ramps, and the three figures |
| `plot_plant_escape.py` | the replotter's CLI, end to end on a synthetic run directory |
| `plant_population_escape.py` | date parsing, the immune weighting, the cross-immunity kernels, the escape score, the single/pair decomposition, and the CLI |

Nothing here loads PLANT, ESM or torch. The default run is fully synthetic and
offline; the tests that touch the real 150k-sequence background CSV and the
committed `Results/JtoJ.2.4_scan/plant` run are opt-in.

```bash
./run_tests.sh              # fast, offline, fully synthetic
./run_tests.sh -s           # + the real-run / real-background tests
./run_tests.sh -c           # + coverage of the three modules
./run_tests.sh -t test_population_weights.py
```

Or directly, from any working directory:

```bash
/home3/oml4h/miniconda3/envs/plm_entropy/bin/python -m pytest \
    /home3/oml4h/PLM_SARS-CoV-2/tests_jtoj24_scan
```

This directory carries its own `pytest.ini` and is **not** under `tests/`, for
the same reason `tests_prescott_iav` is not: the suites cover different code and
neither may be able to break the other.

## The design rule

Copied from `tests_prescott_iav/conftest.py`, because it is what makes a test
suite worth having:

> A fixture whose expected values have to be computed by the code under test is
> worthless: it can only ever assert that the code agrees with itself.

So every expectation here is a literal, or one line of arithmetic a reader can
redo. Two fixtures carry the whole suite.

### The synthetic geometry

Root at the origin, three mutations, coordinates chosen so the endpoint axis is
`(1, 2, 2)` — whose length is exactly **3**, so the unit vector is
`(1/3, 2/3, 2/3)` and every projection is a third of an integer:

| genotype | coordinates | \|Δ\| | along axis | off axis | fraction |
| --- | --- | --- | --- | --- | --- |
| `N122D` | (1, 0, 0) | 1 | 1/3 | 2√2/3 | 1/9 |
| `T135K` | (0, 2, 0) | 2 | 4/3 | 2√5/3 | 4/9 |
| `K189R` | (0, 0, 2) | 2 | 4/3 | 2√5/3 | 4/9 |
| `N122D+T135K` | (1.5, 2, 0) | 2.5 | 5.5/3 | — | — |
| `N122D+K189R` | (1, 0, 2) | √5 | 5/3 | — | — |
| `T135K+K189R` | (0, 2, 2) | 2√2 | 8/3 | — | — |
| endpoint | (1, 2, 2) | 3 | 3 | 0 | 1 |

Pairwise epistasis along the axis is therefore exactly `1/6` for
`N122D+T135K` (from a planted `(0.5, 0, 0)`) and exactly **0** for the other
two — not "small", zero, so the assertions use `abs=1e-12`.

There is a deliberate trap in it: the **triple is exactly additive while one
pair is not**. Anything that infers pairwise terms from the endpoint instead of
measuring each double mutant passes every other test and fails
`test_pairwise_epistasis_is_not_inferred_from_the_endpoint`.

`T135K` and `K189R` are also deliberately interchangeable under the stationary
measure — same `|Δ|`, same on-axis component — which is what
`test_stationary_escape_ignores_the_landscape_entirely` uses to show the
population measure is not redundant.

### The synthetic immune landscape

Four sequences from 2020 and one from 2022, scored as of 2023.0 with a 1-year
half-life, so the raw recency weights are `0.5**3 = 0.125` and `0.5**1 = 0.5`:

| `--normalise-by` | 2020 mass | 2022 mass | split |
| --- | --- | --- | --- |
| `none` | 4 × 0.125 = 0.5 | 0.5 | **50 / 50** |
| `year` | 0.125 | 0.5 | **20 / 80** |

That contrast *is* the surveillance-effort correction, in numbers you can check
without running anything. Two of the four 2020 sequences also sit on identical
coordinates, so 2020 holds 3 distinct antigenic positions among 4 sequences and
`--within-period unique` splits its 0.2 three ways rather than four — giving
`0.2/3` to each isolated strain and `0.2/6` to each half of the tied pair.

## Layout

| File | Contents |
| --- | --- |
| `conftest.py` | the two fixtures above, the ground-truth literals, and the marker-driven skips |
| `test_escape_geometry.py` | closed-form components, orthogonality, planted epistasis, and translation / rotation / scaling invariance |
| `test_escape_figures.py` | ramps, the diverging colormap, `place_labels`, and that every figure is a real PNG on every degraded path |
| `test_plot_plant_escape_cli.py` | the replotter's arguments, label resolution, failure modes, and that the CSV values survive the round trip |
| `test_population_dates.py` | both date formats in `backgrounds.csv`, leap years, malformed input, and `load_backgrounds` |
| `test_population_weights.py` | recency, strictly-before, `--max-age`, period normalisation, the three `--within-period` modes, and the diagnostics |
| `test_population_kernels.py` | each kernel's closed forms, plus the local-impact property as an assertion rather than a claim |
| `test_population_escape_math.py` | the chunked distance computation against a naive reference, and the single/pair decomposition |
| `test_population_cli.py` | the full CLI: outputs per date, metadata, every flag actually changing the answer, and every failure mode |
| `test_real_data.py` | opt-in: golden numbers from the committed run, and that the genotype and background frames really are shared |

## Markers

```
what a test IS      unit  integration  cli  figure  slow
what a test NEEDS   requires_real_run  requires_real_backgrounds
```

`slow` is skipped unless `--run-slow`; the `requires_*` markers skip themselves
when the input is missing, so the suite is green on a machine that has neither
the `Results/` tree nor the PLANT download.

## Things worth knowing when a test fails

- **`test_the_frames_really_are_shared`** failing means the genotype embeddings
  and `backgrounds.csv` came from different PLANT checkpoints. Every population
  escape number is meaningless until that is fixed; nothing else in the suite
  will tell you, because the synthetic fixtures share a frame by construction.
- **The golden numbers in `test_real_data.py`** were read off the committed
  `genotype_embeddings.csv`. They are a regression guard, not a claim of
  correctness — if the checkpoint or the scan changes, update them deliberately,
  in a commit that says so.
- **`test_diagnostics_are_json_safe_once_the_table_is_removed`** exists because
  the metadata writer does `json.dumps({..., **diagnostics})`. Adding any
  non-serialisable value to the diagnostics dict breaks the CLI at the last
  step of a long run; this catches it in milliseconds.
