# `tests_prescott_iav` — test suite for the PRESCOTT/ESCOTT influenza pipeline

Covers `scripts/prescott_iav/{constants,common,prepare_inputs,jet_surrogate,run_escott,leakage_check}.py`
and `scripts/run_prescott_diversity.py`.

This directory is **self-contained**. It is not part of `tests/`, it carries its own
`pytest.ini`, and it neither reads nor depends on the repository-level `pytest.ini`.
The two suites are independent: neither can break the other.

---

## Running it

```bash
/home3/oml4h/miniconda3/envs/PRESCOTT/bin/python -m pytest \
    /home3/oml4h/PLM_SARS-CoV-2/tests_prescott_iav
```

That works from **any** working directory and needs no repo-level config: pytest takes
its rootdir config from the first ancestor of the given path that contains one, and
stops at `tests_prescott_iav/pytest.ini`.

Always use that interpreter. **Never** a bare `python`, `pytest` or `Rscript` — the
system ones lack this project's dependencies, and the system `makeblastdb` (see below)
does not merely fail, it hangs.

Useful variants:

```bash
# one module, verbose, stop at the first failure
… -m pytest tests_prescott_iav/test_run_escott.py -x -v

# only the fast, dependency-free tests
… -m pytest tests_prescott_iav -m unit

# in parallel (pytest-xdist is installed)
… -m pytest tests_prescott_iav -n 8

# include the opt-in tests: minute-scale work and the real production data
… -m pytest tests_prescott_iav --run-slow

# with coverage (opt-in; roughly doubles the runtime)
… -m pytest tests_prescott_iav \
      --cov=prescott_iav --cov=run_prescott_diversity --cov-report=term-missing
```

The default run should stay in the tens of seconds and must work **offline**.

---

## PATH: the one piece of magic

`conftest.py` prepends `/home3/oml4h/miniconda3/envs/PRESCOTT/bin` to `os.environ["PATH"]`
at import, for this process and everything it spawns.

This is load-bearing, not tidiness. The modules under test resolve their external tools
by bare name (`makeblastdb`, `blastp`, `mafft`, `mkdssp`, `Rscript`, `escott`,
`prescott`). On this machine the bare names otherwise resolve to
`/software/blast-v2.11.0/bin`, and **that `makeblastdb` hangs indefinitely** inside
`leakage_check.blast_records`. A suite that inherits an unprepared PATH does not go red,
it stops responding. With the wiring in place the leakage tests finish in ~15 s.

`conftest.py` also prepends the repo root and `<repo>/scripts` to `sys.path` (exactly
once) and sets `MPLBACKEND=Agg`, so test modules can simply do:

```python
from prescott_iav import common, constants, jet_surrogate, run_escott
from tests_prescott_iav.conftest import QUERY_PROTEIN, EXPECTED_PARENT_MAP
```

`scripts/run_prescott_diversity.py` is a script, not an importable module; use the
`driver_module` fixture, which loads it by file location.

---

## Marker taxonomy

Two orthogonal axes. Declare both where they apply.

### What a test *is*

| marker | meaning |
|---|---|
| `unit` | pure and fast; no subprocess, no external binary |
| `integration` | several modules together, still fully synthetic |
| `cli` | exercises a module's argparse surface / `main()` |
| `slow` | minute-scale. **Opt-in**: skipped unless `--run-slow` |

### What a test *needs*

Each of these is auto-skipped, with a reason, when its dependency is absent. The probe
runs once per session and is cached, so marking 500 tests costs nothing.

| marker | skipped when |
|---|---|
| `requires_escott` | `escott` / `prescott` are not on PATH |
| `requires_blast` | `blastp` / `makeblastdb` are not on PATH |
| `requires_mafft` | `mafft` is not on PATH |
| `requires_muscle` | `muscle` is not on PATH |
| `requires_dssp` | `mkdssp` is not on PATH |
| `requires_r` | `Rscript` is not on PATH |
| `requires_prody` | `prody` is not importable |
| `requires_freesasa` | `freesasa` is not importable |
| `requires_scipy` | `scipy` is not importable |
| `requires_torch` | `torch` is not importable |
| `requires_rma` | `run_mutational_accessibility.py` is not importable (needs torch) |
| `requires_prescott_python` | the PRESCOTT env interpreter is missing |
| `requires_blat_reference` | the shipped PRESCOTT BLAT reference data is missing |
| `requires_real_data` | the multi-GB production inputs are missing. **Opt-in**: also needs `--run-slow` |

`blast` is a legacy alias for `requires_blast` and behaves identically.

**Opt-in markers are `slow` and `requires_real_data`.** They are the two things that stop
the default suite being fast and offline, so they are skipped unless `--run-slow` is
given. Everything else runs by default and skips itself only when genuinely unavailable.

In the current PRESCOTT env **every** capability above is present, so nothing is
auto-skipped here today; the markers exist so the suite degrades honestly elsewhere.

Adding a marker to `pytest.ini` alone is not enough — add its probe to
`conftest.CAPABILITIES` too, or it will never skip anything. `--strict-markers` and
`--strict-config` are on, so a typo is an error rather than a test that silently never
runs.

---

## Fixtures

The design rule, and the reason this file is worth reading before writing a test:

> **A fixture whose expected values have to be computed by the code under test is
> worthless.** It can only assert that the code agrees with itself.

So every fixture ships independently-derived ground truth: a literal, arithmetic worked
by hand in the docstring, or a closed form you can check in one line. Denominators are
chosen to make that possible — the panels hold exactly 100 records, so a planted count
*is* a percentage.

### Sequences

| fixture / constant | what it gives you |
|---|---|
| `QUERY_PROTEIN` | 72 aa of HA-like protein (a literal) |
| `QUERY_CDS` | 72 codons + `TGA`; translates back to `QUERY_PROTEIN` |
| `PARENT_PROTEIN` | the same protein with **T40I** — one difference, on purpose |
| `query_protein_fasta` | single-record FASTA; header `HAK` → escott token `HAK` |
| `query_cds_fasta` | single-record nucleotide CDS with a realistic GISAID-style header |

### Alignments

| fixture | ground truth |
|---|---|
| `tiny_msa` | 12 × 72, query first and ungapped. Per-column counts, occupancy counts, residue-type counts and a `column_class` map. Conserved columns (1 type, occupancy exactly 1.0), all-gap columns (only the query has a residue), hypervariable columns (12 distinct residues), semi-conserved (9 + 3) and mild (11 + 1) |
| `uniform_msa` | 12 identical rows → Henikoff weights **exactly 1.0**, occupancy **exactly 1.0** |
| `handworked_msa` | 4 × 4; weights **(7/6, 5/6, 5/6, 7/6)**, derived term by term in the docstring |
| `gapped_query_msa` | first row has gaps — `build_jet_table` must refuse it |

### Structures

Geometry is chosen so the answers are exact, not approximate.

| fixture | ground truth |
|---|---|
| `cv_ladder_pdb` | 8 CA atoms, 3.8 Å apart, along +x. At radius 7.0 an interior residue sees exactly two opposite neighbours → circular variance **exactly 1.0**; a terminus sees one → **exactly 0.0** |
| `cv_context_pdb` | the same chain A plus one chain-B atom at x = −3.8. Residue 1 goes 0.0 → **1.0**; nothing else moves. This is the monomer-vs-trimer question in miniature |
| `sasa_monomer_pdb` | two isolated glycines. A lone CA sphere has SASA ≥ 109 Å² against Tien's 104 Å² maximum for glycine, so RSA clips to **exactly 1.0** under any radius set |
| `sasa_context_pdb` | residue 1 enclosed by a 12-atom icosahedral shell at 3.0 Å → RSA ≈ 0; residue 2, 100 Å away, still exactly 1.0 |
| `query_numbered_pdb_factory` | CA backbone in query numbering 1..72; `covered=` makes a partial-coverage structure, `chains=` makes a trimer |

### Panels

| fixture | ground truth |
|---|---|
| `frequency_panels` | parent and target panels, 100 records each, with planted per-position counts. Ships the **exact** mutant set and frequency `build_parent_frequency_file` must emit at `--min-count` 1 and 2, the three mutants that must be dropped and *why* each is dropped, the median depth (100.0), and closed-form entropy for the target panel |
| `leakage_panels` | target / parent / deep sets with **one planted duplicate each**, at documented row indices. Deep rows carry the 16-aa signal peptide that the panels lack, so a genuine duplicate is *not* byte-identical — hashing cannot see it, only alignment can. `deep_clean_fasta` is the same set with the leak removed |
| `panel_factory` | build any panel from a `{position: {residue: count}}` spec; returns the path **and** the counts/depths/frequencies derived from the literal spec |

The `frequency_panels` parent spec is designed so each planted site exercises exactly one
branch:

| position | planted | branch it exercises |
|---|---|---|
| 10, 15, 35 | 0.10 / 0.25 / (0.30 + 0.20) | ordinary standing variation, kept |
| 20 | 19 of 95 (5 gaps) | depth ≠ record count |
| 25 | count 1 | kept at `--min-count 1`, dropped at 2 |
| 30 | 0.98 | ≥ `freq_max` 0.95: ancestral residue at a defining site |
| 40 | 0.12 | parent reversion. Far below `freq_max`, so **only** `drop_parent_reversions` can catch it |

Closed forms for the target panel: position 18 is 50/50 → entropy **exactly 1.0 bit**;
position 5 is 80/20 → **0.7219280948873623 bits**.

### Pipeline artefacts

| fixture | notes |
|---|---|
| `fake_jet_res` / `jet_res_factory` | valid `.res` in escott's exact column layout. The default has 7 of 72 zero-trace positions = 9.72%, deliberately **inside the warn band** (above 5%, below the 10% refusal ceiling). `jet_res_factory(n_zero=30)` sits above the ceiling |
| `fake_escott_matrix` / `escott_matrix_factory` | a real R `write.table` file: quoted column names, quoted row names, L fields in the header and L+1 per row, `NA` on the wild-type cell. Verified against real `HAJ_normPred_evolCombi.txt`. Ships the values it wrote, so a parser test is a genuine round trip. Flat (zero-trace) columns softmax to **exactly 1/20** at any temperature. Pass `values=` to write a deliberately malformed matrix |
| `frequency_file_factory` | PRESCOTT custom frequency file, `<MUTANT> <freq>`, no header, `.txt` (never `.csv` — prescott switches parsers on that suffix) |
| `score_matrix_factory` | `plm_probability_profile.csv` layout: a `sequence` row then 20 rows in PLM cache order, no header |

### Trees and environment

| fixture | notes |
|---|---|
| `output_dir_factory` | fresh output tree with `tables/`, `tables/diagnostics/`, `figures/`, `scores/`, `inputs/` |
| `guide_factory` / `five_lineage_guide` | a MONTHLY_GUIDE CSV (`month,fasta,reference`) plus the panels and nucleotide references it names. `five_lineage_guide` is the production topology in miniature, so parent-map validation has real rows to validate against |
| `prepared_inputs_tree` | a complete stage-A/B `inputs/` tree — query, MSA, jet, primary and sensitivity frequency files, structures and a full `inputs_manifest.json` with `frequency_index` — **without running stage A or B**. `run_escott.resolve_lineage_inputs` and `resolve_alternate_frequency_paths` both work against it |
| `prescott_modules` | the six stage-1 modules, imported once per session through the package |
| `driver_module` | `run_prescott_diversity.py`, loaded by file location (pulls in torch; session-scoped) |
| `prescott_python`, `subprocess_env`, `run_module_cli` | the right interpreter, an env with the right PATH/PYTHONPATH, and a helper that runs a module's CLI in a subprocess |
| `expected_constants` | literal copies of the pinned constants, for testing `prescott_iav.constants` **against** rather than **with** |

### The literals that must stay literal

`EXPECTED_PARENT_MAP`, `EXPECTED_SENSITIVITY_PARENT_MAP`, `EXPECTED_TRACE_TOP_FRACTION`,
`EXPECTED_LINEAGE_TAGS`, `ESCOTT_ROW_ORDER`, `PLM_CACHE_ROW_ORDER` and friends are
written out by hand in `conftest.py` and **deliberately not imported** from
`prescott_iav.constants`.

The corrected ladder is a single line:

```
G.1 -> J_int -> J.2_int -> J.2.4 -> K
```

`K` descends from **J.2.4**, not from `J.2_int`. A test that read the map out of the
module and compared it with itself would pass happily if the module regressed to the old
edge — which is exactly the regression these literals exist to catch. `brief_as_stated`
(`K ← J.2_int`) is retained only as the labelled sensitivity alternative and must never
be the default.

---

## `test_scaffolding.py`

Validates the scaffolding itself, not the pipeline: that the pinned literals still match
`constants`, that the synthetic CDS still translates, that the hand-worked weights and
the ladder's circular variances are still exact, that the R-format matrix still
round-trips through the real parser, and that the planted panel counts still produce the
expected frequency file.

Please do not extend it with pipeline tests — put those in your own module. Add to it
only when you add a fixture whose ground truth needs pinning.

---

## House rules

* **Never** create, move or modify anything under `/home3/oml4h/PLM_SARS-CoV-2/tests/`,
  and never modify the repo-level `pytest.ini`.
* Every subprocess gets `subprocess_env` (or `run_module_cli`). Never a bare binary name
  from an unprepared environment.
* Prefer an exact assertion over `approx`. If you need `approx`, say in a comment why the
  value cannot be exact.
* If you need a new fixture, put it in `conftest.py` with its ground truth and its
  derivation, and pin it in `test_scaffolding.py`.
* Tests must not write outside `tmp_path`, and must not depend on execution order.
