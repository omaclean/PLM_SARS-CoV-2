# Coverage and scope — `tests_prescott_iav`

What this suite tests, what it deliberately does **not** test and why, and the list of
real defects the build-plus-two-adversarial-passes exercise uncovered.

Companion to `README.md` (which covers *how to run* the suite). This file covers *what
the numbers mean*.

---

## Headline

| | |
|---|---|
| Tests, default run | **2287 passed, 26 skipped, 0 failed** (~69 s serial, ~34 s with `-n 8`) |
| Tests, `--run-slow` | **2313 passed, 0 skipped, 0 failed** (~94 s with `-n 8`) |
| Coverage, `--run-slow` | **98.31 %** — 3654 statements / 33 missed, 1324 branches / 51 partial |
| Coverage, default run | **98.25 %** — identical missed-line set, 3 more partial branches |
| Repo-level `tests/` | **17 failed, 112 passed, 3 skipped — pre-existing, see below** |

Measured with:

```bash
cd /home3/oml4h/PLM_SARS-CoV-2/tests_prescott_iav
/home3/oml4h/miniconda3/envs/PRESCOTT/bin/python -m pytest . --run-slow \
    --cov=prescott_iav --cov=run_prescott_diversity --cov-branch \
    --cov-report=term-missing
```

### Per module (`--run-slow`)

| module | stmts | miss | branch | brpart | cover |
|---|---:|---:|---:|---:|---:|
| `constants.py` | 43 | 0 | 12 | 0 | **100.0 %** |
| `common.py` | 215 | 0 | 86 | 1 | **99.7 %** |
| `run_escott.py` | 680 | 4 | 242 | 5 | **99.0 %** |
| `leakage_check.py` | 668 | 7 | 220 | 3 | **98.9 %** |
| `jet_surrogate.py` | 538 | 4 | 178 | 5 | **98.7 %** |
| `prepare_inputs.py` | 462 | 5 | 168 | 6 | **98.3 %** |
| `run_prescott_diversity.py` | 1048 | 13 | 418 | 31 | **97.0 %** |
| **TOTAL** | **3654** | **33** | **1324** | **51** | **98.31 %** |

### Tests per file

| file | tests | what it holds |
|---|---:|---|
| `test_common.py` | 423 | shared helpers, `constants`, label/key identity, reference→alignment column map |
| `test_driver_cli.py` | 347 | the driver's argparse surface, validation, flag forwarding to stage 1 |
| `test_driver_analysis.py` | 317 | alpha sweep, metrics, temperature resolution, best-alpha selection, figures |
| `test_run_escott.py` | 317 | ESCOTT/PRESCOTT invocation, score matrices, frequency equations, clipping |
| `test_leakage_check.py` | 294 | BLAST screening, purge bookkeeping, FASTA parsing |
| `test_jet_surrogate.py` | 261 | `jet.res` construction, trace/pc/cv/ss, DSSP, SASA, the manifest cache |
| `test_prepare_inputs.py` | 194 | guide parsing, CDS→protein, structure prep, MSA, frequency files |
| `test_regressions_numerics.py` | 65 | the numerics hunt (bugs 6–10 below) + properties proven sound |
| `test_regressions_coordinates.py` | 47 | the coordinate/identity hunt (bugs 11–12 below) |
| `test_scaffolding.py` | 35 | the suite's own fixtures, markers and PATH contract |
| `test_import_fallbacks.py` | 13 | every arm of the `constants`/`common` import chains resolves to one module |
| **total** | **2313** | |

---

## What is tested

**The coordinate chain, end to end, on real production data.** CDS → 566-aa HA0 for all
five lineages; the 16-residue signal peptide; 6WXB author numbering → query numbering at
offset **+16** (84.5 % identity, versus 9.5 % for the runner-up and <15 % for a ±1 slip);
`has_structure` being exactly the covered set; and `common.map_reference_to_alignment_columns`
being **byte-identical** to `Functions_HuggingFace.build_reference_to_alignment_column_map`
on the real K / J.2.4 / J.2_int panels. That last equivalence is the contract that keeps
the frequency prior aligned with the observed diversity, and it was previously untested.

**The parent map.** Default preset `clade_evidence` encodes the single linear ladder
`G.1 → J_int → J.2_int → J.2.4 → K`. The old `K ← J.2_int` edge survives **only** as the
explicitly labelled `brief_as_stated` sensitivity preset, which is what `--parent-sensitivity`
exists to compare. Both presets, and the wire format that carries them to stage 1, are pinned.

**The numerics.** Softmax temperature resolution (including the `match-plm` solve), the
alpha sweep, the frequency-penalty equations, clipping accounting, and the CSV round trip
(exact to 4.8e-263, columns summing to 1).

**Degenerate and adversarial inputs.** Flat score matrices, single-class AUROC, constant
columns, zero-depth panels, `nan`/`inf` everywhere a float is parsed, malformed FASTA
deflines, empty guide cells, duplicate mutant keys, wrong-frame frequency files.

**Identity and filename safety.** All five lineage labels stay distinct under `safe_label`,
`dot_free_key`, `variant_parent_token` and `lineage_tag`; `build_variant_name` agrees with
`stage1_variant_name` over a 300-point grid; slash-bearing FASTA headers never reach a filename.

**Import fallbacks.** `run_escott.py` and `jet_surrogate.py` each reach `constants` three
ways. Only one arm fires per interpreter, so a normal run leaves two unexecuted.
`test_import_fallbacks.py` forces each arm and compares the resolved values against the
literals — because an arm resolving to a *different* `constants` would silently
desynchronise the parent map with no error anywhere.

---

## What is deliberately NOT tested, and why

### The 33 uncovered statements

Every one is in one of three categories. None is a live untested path.

**1. `sys.path` bootstraps and `__main__` guards (13 statements).**
`jet_surrogate.py:196, 1529`; `leakage_check.py:177, 2067–2071`; `prepare_inputs.py:52, 1194`;
`run_escott.py:155, 1903`; `run_prescott_diversity.py:73, 3174`. These run at import or at
process exit under the real interpreter and cannot fire under pytest, which has already
imported the module. Their *effect* is tested — `test_import_fallbacks.py` proves the path
manipulation lands on the right module, and the CLI tests drive `main()` directly rather
than through `sys.exit(main())`.

**2. Defensive `raise` guards proven dead by construction (12 statements).**
`prepare_inputs.py:474` (escott would truncate the header), `:757` (alignment-length
disagreement), `:813` (duplicate mutant keys); `run_escott.py:622` (softmax columns do not
sum to 1), `:1062` (a dot in the prescott `-o` stem); `jet_surrogate.py:378` (a column with
zero residue types), `:1090` (`traceMax` emitted); `leakage_check.py:1132` (purge
bookkeeping error). Each is an internal-consistency assertion whose precondition is
established elsewhere in the same function. **Each has a named test pinning the invariant
that makes the guard dead**, rather than a test that fakes the branch — the guard is a
tripwire for a future edit, and faking it would test nothing. Deliberate.

**3. Optional-output false arms in the heavy scoring loop (8 statements).**
`run_prescott_diversity.py:2710, 2770, 2884, 3044, 3059, 3061` and the two
`if idx is None: continue` arms at `:3006, 3029`. The *decisions* are covered by unit tests
on `best_alpha_index` (5 of them); only the trivial `continue` inside `run_analysis`'s loop
is unhit, and it is unreachable in a normal run because a non-empty `alpha_df` always has
grid rows. The `--diagnostic-exports` arms are exercised under `--run-slow` in one direction
only.

### Whole capabilities gated behind markers

26 tests skip by default. They are **opt-in, not disabled** — the three `pytest.mark.skip`
occurrences in the suite are all inside `conftest.pytest_configure`, applied programmatically
for `--run-slow` and for capability gating. No test is hard-disabled anywhere.

- `slow` / `requires_real_data` — need the multi-GB production inputs. Run with `--run-slow`.
- `requires_*` (blast, mafft, muscle, dssp, r, prody, freesasa, scipy, torch, escott,
  blat_reference, prescott_python) — auto-skip when the dependency is absent, so the suite
  stays green on a machine that lacks one. All of them are present in the PRESCOTT env, so
  `--run-slow` there skips nothing.

### Out of scope by instruction

- **`scripts/run_mutational_accessibility.py` and `Functions_HuggingFace.py`** are the PLM
  equivalents this pipeline mirrors. They are covered by the pre-existing `tests/` suite and
  were not to be modified. **They contain a live instance of bug 5 below** — the identical
  `idxmax`-over-the-baseline-row pattern at `run_mutational_accessibility.py:5082-5083` and
  `5108-5109`. It was left in place deliberately; it should be fixed there too.
- **The repo-level `tests/` suite** has **17 pre-existing failures**. These are not caused by
  this work and were proven independent twice over: `git diff --stat HEAD` is empty (no
  tracked file was modified by any pass), and re-running `tests/` with
  `scripts/prescott_iav/`, `scripts/run_prescott_diversity.py` and `tests_prescott_iav/`
  physically moved out of the tree produces a **byte-identical failure set** (`diff` clean).
  Nothing under `tests/` references either module.
- **Real JET2** is not installed; `jet_surrogate.py` is the surrogate. It is validated against
  the one real JET2 output PRESCOTT ships (`BLAT_jet.res`, rho = 0.877 on `trace`) rather than
  against JET2 itself.

### Known honest limits, recorded rather than hidden

- The surrogate leaves ~8 % of BLAT positions at `trace == 0` where real JET2 leaves 1 %.
  `pred.R:487` multiplies each ESCOTT column by `trace[i]`, so those sites become uniform
  noise. The pipeline **warns** with the measured best `--trace-top-fraction`; it is a
  modelling limitation, not a bug.
- `--frequency-cutoff-k` is depth-free exactly at the anchors (a singleton costs 0, a fixed
  variant costs `c`) but **not** between them: a doubleton costs 0.128·c in G.1 (N=229) and
  0.068·c in J.2_int (N=27452), a 1.88x gap. Pinned as a test so it is not mistaken for
  depth invariance.
- Equation 1's penalty is inverted relative to intuition (a fixed variant gets *zero*
  penalty, a rare one gets `c`). Verified line by line against upstream `prescott.py:747-757`
  — a **faithful reproduction**, not our bug. Equation 2 is the default.

---

## Real bugs found and fixed

Twelve. Each was **revert-verified**: reverted in the real source, the named test confirmed
failing, then restored and checksum-checked.

### Build pass

| # | Bug | Guarded by |
|---|---|---|
| **1** | `leakage_check.accession_of` — `IndexError` on a `>`-only header. The truthiness test read the *original* string but the index read the *`>`-stripped* one, so a `>>` defline (which `read_fasta` yields as `">"`) killed the entire leakage stage over one malformed record. | `test_leakage_check.py::test_accession_of_a_gt_only_header_must_not_crash` and `::test_a_gt_gt_defline_reaches_accession_of_through_the_real_reader` |
| **2** | `jet_surrogate` — `--structure-chain` was neither recorded in the manifest nor compared, so a chain-B rerun was served chain A's table with exit 0. On a real trimer that is the wrong subunit's `cv`, `pc` and DSSP. | `test_jet_surrogate.py::test_structure_chain_invalidates_the_cache` |
| **3** | `jet_surrogate` — a cache hit returned before the `--out-components` / `--out-dssp` blocks, so a rerun that newly requested them exited 0 producing nothing. This bit the driver, which requests the components TSV on reruns. | `test_jet_surrogate.py::test_cache_hit_still_writes_requested_side_outputs` |
| **4** | `prepare_inputs` — an empty `reference` guide cell became `Path("")` = `PosixPath(".")`, whose `.exists()` is `True`, bypassing the `FileNotFoundError` guard and dying deep inside `load_reference_cds` as `IsADirectoryError('.')`, naming neither the lineage nor the guide. | `test_prepare_inputs.py::test_empty_reference_is_refused` and `::test_guide_row_with_an_empty_reference_column_should_name_the_lineage` |
| **5** | `run_prescott_diversity` — `best_alpha = NaN`. Both `idxmax` calls ran over the whole sweep table, which includes the mutation-only baseline row carrying `alpha = NaN`. Whenever the codon model alone out-ranked the grid, `best_alpha_two_methods.tsv` reported `NaN`, reading as a failed fit rather than "the baseline won". | `test_driver_analysis.py::test_best_alpha_is_never_nan` plus 5 tests on `best_alpha_index` |

### Numerics pass

| # | Bug | Guarded by |
|---|---|---|
| **6** | `resolve_escott_temperature` — `match-plm` calibrated against the **wrong spread**, missing its target by 30–45 %. It solved `T = sd(E)/sd(log plm_ref)` using the *total* sd, but `plm_prob` is a **per-column** softmax, whose per-column constant annihilates all between-column variance. On real ESCOTT: total sd 1.8155 vs within-column 0.9005. Targeting sd 1.0 achieved 0.577. `match-plm` exists solely to make `best_alpha` comparable with a PLM run, and alpha is not scale-free. Replaced with a bisection solve (`sd_log_softmax` + `solve_softmax_temperature`), exact to 1.4e-13. | `test_regressions_numerics.py::TestMatchPlmTemperatureCalibration` (14 tests), anchored on a closed-form invariant needing no measured number: adding a constant to a whole column is a no-op for the softmax, so it must be a no-op for the matched `T`. |
| **7** | `count_flat_columns` tested `abs(value) <= 1e-12`, seeing only the `trace == 0` route. **Full conservation gives a constant *non-zero* column** — same uniform 1/20 softmax, same total absence of rank information, but counted as alive. Real MLH1: 8 of 756 columns constant, 0 all-zero, so the pipeline wrote `n_flat_columns = 0` for a protein with 1.06 % of its positions dead. | `test_regressions_numerics.py::TestUniformColumnsAreCountedAsDead` (11 tests) |
| **8** | `nan` / `inf` temperatures walked through every guard, because `nan <= 0` is `False`. `--escott-temperature nan` produced an all-NaN matrix surfacing as a misleading assertion; **`inf` raised nothing at all** — `E/inf == 0` yields an exactly uniform matrix (every site dead) and the run completes with meaningless metrics. Same hole in `--alpha-step` and `--coefficient-grid`. | `test_regressions_numerics.py::TestNonFiniteTemperature` (12 tests) |
| **9** | The `exp()` underflow diagnostic named neither the temperature nor a way out. On real ESCOTT the widest column spans 6.03 units, so any `T < ~0.0082` underflows — and `--escott-temperature 0.001` is an ordinary thing to try. | `test_regressions_numerics.py::TestUnderflowTemperatureDiagnostic` (3 tests, incl. the real-data boundary at 6.03) |
| **10** | `load_frequency_file` accepted `nan` and `inf`, because `(frequency <= 0).any()` is NaN-transparent. The NaN case fails **silently**: the mutant is counted in `n_mutants_with_frequency` yet receives no penalty from any equation, because both `nan > Fc` and `nan <= Fc` are `False`. | `test_regressions_numerics.py::TestFrequencyFileValidation` (12 tests) |

### Coordinate / identity pass

| # | Bug | Guarded by |
|---|---|---|
| **11** | `jet_surrogate` — the structure cache compared **paths, not content**. Its own comment said the guard existed because "a second run with a different structure writes to the same path", but the path string is exactly what does not change there: `prepare_inputs` rewrites `inputs/structure/<stem>_chain<C>_qnum.pdb` every run, so the path is invariant under a changed `--structure-offset`, an edited structure, or two `--structure` files whose stem token collides (`6WXB.cif` and `6WXB-assembly1.cif` both map to `6WXB`). Demonstrated on real data: offset `auto` (+16) then `10` gave a **cache hit, exit 0**, with every `pc`/`cv`/`ss` belonging to different residues than ESCOTT would score. The cache hit also skipped `build_jet_table`'s own frame check, which refuses the same file at 5.8 % identity. | `test_regressions_coordinates.py::TestJetCacheKeysOnStructureContent` (7 tests) + `::TestStructureStemTokenIsLossy` |
| **12** | `run_escott` — a frequency file in the **wrong frame** made every PRESCOTT variant a numerical clone of ESCOTT, silently. `build_log10_frequency_matrix` drops records whose position/WT letter disagree; with *nothing* landing, `prescott_v2_scores` collapses to the identity (measured max abs(PRESCOTT − ESCOTT) = 5.6e-17). Stage C still wrote each grid point under its own name and stage D reported it as a separate model with its own "best alpha" row. Only trace: a `print`. This is the most likely coordinate accident in the domain — HA is universally published in **mature (H3) numbering** while every frame here is the 566-aa HA0 translation, so a file one signal peptide (16 residues) out matches 0 %. New `assert_frequency_frame` with `MIN_FREQUENCY_MATCH_FRACTION = 0.5`: a self-consistent tree matches 100 %, a whole-frame error 0 %, a one-residue shift ~5 %. | `test_regressions_coordinates.py::TestFrequencyFileMustBeInTheEscottFrame` (7) + `::TestProcessLineageRefusesAWrongFrameFrequencyFile` (3) |

### Investigated and found sound

`test_regressions_numerics.py::TestNumericPropertiesThatHold` pins these so nobody re-hunts
them: frequency exactly 1.0 not colliding with `NO_FREQUENCY_SENTINEL` (999.0); frequency
exactly 0 rejected at both writer and reader; zero-depth panel columns short-circuiting
before the division; `c = 0` being the identity to 1.07e-14 on real data; single-class AUROC
handled by `rma.safe_auroc`; and column-depth spread within real panels being negligible
(worst-case singleton penalty 0.001·c), which is what makes `median_depth` a fair stand-in
for `N`.

Dropped as latent-only, with no demonstrated concrete failure: `constants.variant_parent_token`
not being byte-identical to `run_prescott_diversity.variant_token` as its docstring claims
(identical for all five production labels, and `reconcile_variant_plan` matches on the parent
*label*); `prepare_inputs` keying `frequency_report.json` by label while
`run_escott.resolve_frequency_cutoff` looks it up by key (measured Fc drift on the real G.1
panel: 0.000000); `prescott_parity_check` lacking a `position < 1` guard (unreachable);
`prepare_structure` raising an opaque `prody.SelectionError` on a blank chain ID (a crash,
not corruption); and `compare_to_reference` aligning by row order without checking `pos`/`AA`
(the shipped `BLAT_jet.res` is 1..286 and matches).

---

## End-to-end verification

After all twelve fixes, the driver was run on the **real** guide
(`Sequences/IAV_lineage_guide.csv`, real GISAID panels, real 6WXB) in `--test-mode` and
completed with **exit 0**, producing 4 score matrices, 99 figures and the full table set.
Three of the fixes are visible in its own log and outputs, on real data:

```
@> Parent map (clade_evidence): {'J_int': 'G.1', 'J.2_int': 'J_int',
                                 'J.2.4': 'J.2_int', 'K': 'J.2.4'}
@> Structure 6WXB-assembly1.cif: offset +16, 485/566 covered (85.7%)
@> [J_int] 18/566 positions have an all-zero ESCOTT column (trace == 0)
@> [J_int] 19/566 positions softmax to a uniform 1/20 (18 from trace == 0,
           1 from a constant non-zero column)          <- bug 7, caught in production
@> [J_int] frequency file (G.1) J_int_parent_frequency.txt:
           {'n_frequency_records': 61, 'n_matched': 61, 'n_unmatched': 0}
                                                        <- bug 12's guard, 100% as predicted
```

- That constant-non-zero column at J_int is a site the pre-fix pipeline would have reported
  as alive. Bug 7 was found on the PRESCOTT distribution's MLH1 sample and reproduces on
  this project's own influenza data.
- `best_alpha_two_methods.tsv` contains no `NaN` (bug 5), and the PRESCOTT variants score
  distinctly from the ESCOTT baseline (Method B: 0.32018 vs 0.31999), so the frequency prior
  is genuinely landing rather than collapsing to the identity (bug 12).
